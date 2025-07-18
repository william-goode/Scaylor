import os
import time
import json
import logging
import getpass
import os

from datetime import datetime
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_progress.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingCheckpointer:
    """
    Handles checkpointing and resuming of embedding processes
    """
    
    def __init__(self, checkpoint_dir: str = "embedding_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, 
                       batch_idx: int, 
                       vectorstore: FAISS,
                       processed_chunks: List[str],
                       metadata: dict):
        """Save checkpoint with current progress"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_batch_{batch_idx}.pkl"
        metadata_path = self.checkpoint_dir / f"metadata_batch_{batch_idx}.json"
        
        # Save vectorstore
        vectorstore.save_local(str(self.checkpoint_dir / f"vectorstore_batch_{batch_idx}"))
        
        # Save metadata
        checkpoint_data = {
            'batch_idx': batch_idx,
            'processed_chunks': processed_chunks,
            'timestamp': datetime.now().isoformat(),
            'total_chunks_processed': len(processed_chunks)
        }
        checkpoint_data.update(metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved at batch {batch_idx}")
        
    def load_checkpoint(self, batch_idx: int) -> tuple:
        """Load checkpoint from specific batch"""
        vectorstore_path = self.checkpoint_dir / f"vectorstore_batch_{batch_idx}"
        metadata_path = self.checkpoint_dir / f"metadata_batch_{batch_idx}.json"
        
        if not vectorstore_path.exists() or not metadata_path.exists():
            return None, None
        
        # Load vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07",
            task_type="QUESTION_ANSWERING") 
            # forgot to include task_type initially, current vectorstore has defaulkt
        
        vectorstore = FAISS.load_local(
                        str(vectorstore_path), 
                        embeddings, 
                        allow_dangerous_deserialization=True
                        )
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Checkpoint loaded from batch {batch_idx}")
        return vectorstore, metadata
    
    def find_latest_checkpoint(self) -> Optional[int]:
        """Find the latest checkpoint batch index"""
        checkpoint_files = list(self.checkpoint_dir.glob("metadata_batch_*.json"))
        if not checkpoint_files:
            return None
        
        latest_batch = max([
            int(f.stem.split('_')[-1]) 
            for f in checkpoint_files
        ])
        return latest_batch
    
    def cleanup_old_checkpoints(self, keep_last: int = 2):
        """Clean up old checkpoints, keeping only the last N"""
        checkpoint_files = list(self.checkpoint_dir.glob("*_batch_*.json"))
        if len(checkpoint_files) <= keep_last:
            return
        
        batch_indices = sorted([
            int(f.stem.split('_')[-1]) 
            for f in checkpoint_files
        ])
        
        # Remove old checkpoints
        for batch_idx in batch_indices[:-keep_last]:
            for pattern in [f"*_batch_{batch_idx}*", f"vectorstore_batch_{batch_idx}"]:
                for file_path in self.checkpoint_dir.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        import shutil
                        shutil.rmtree(file_path)
        
        logger.info(f"Cleaned up old checkpoints, kept last {keep_last}")

def load_pdf_documents(pdf_path: str) -> List[Document]:
    """Load PDF and return documents"""
    logger.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages from PDF")
    return documents

def split_documents(documents: List[Document], 
                   chunk_size: int = 1000, 
                   chunk_overlap: int = 100) -> List[Document]:
    """Split documents into chunks"""
    logger.info(f"Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def create_embeddings_with_checkpoint(
    pdf_path: str,
    batch_size: int = 10,
    delay_between_batches: float = 2.0,
    checkpoint_every: int = 5,
    resume_from_checkpoint: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    max_retries: int = 3,
    exponential_backoff: bool = True
) -> FAISS:
    """
    Create embeddings with checkpointing and progress tracking
    
    Args:
        pdf_path: Path to the PDF file
        batch_size: Number of chunks to process in each batch
        delay_between_batches: Seconds to wait between batches
        checkpoint_every: Save checkpoint every N batches
        resume_from_checkpoint: Whether to resume from existing checkpoint
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        max_retries: Maximum retries for failed API calls
        exponential_backoff: Use exponential backoff for retries
    """
    
    # Initialize checkpointer
    checkpointer = EmbeddingCheckpointer()
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check for existing checkpoint
    start_batch = 0
    vectorstore = None
    processed_chunks = []
    
    if resume_from_checkpoint:
        latest_checkpoint = checkpointer.find_latest_checkpoint()
        if latest_checkpoint is not None:
            logger.info(f"Found checkpoint at batch {latest_checkpoint}")
            try:
                vectorstore, metadata = checkpointer.load_checkpoint(latest_checkpoint)
                if vectorstore is not None:
                    start_batch = latest_checkpoint + 1
                    processed_chunks = metadata.get('processed_chunks', [])
                    logger.info(f"Resuming from batch {start_batch}, {len(processed_chunks)} chunks already processed")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting fresh...")
    
    # Load and split documents
    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    
    # Filter out already processed chunks if resuming
    if processed_chunks:
        original_count = len(chunks)
        chunks = [chunk for chunk in chunks if chunk.page_content not in processed_chunks]
        logger.info(f"Filtered out {original_count - len(chunks)} already processed chunks")
    
    if not chunks:
        logger.info("All chunks already processed!")
        return vectorstore
    
    # Calculate total batches
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    # Create progress bar
    progress_bar = tqdm(
        total=len(chunks),
        desc="Creating embeddings",
        unit="chunk",
        initial=len(processed_chunks),
        position=0,
        leave=True
    )
    
    # Process batches
    batch_idx = start_batch
    chunks_processed_in_session = 0
    
    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_success = False
            retry_count = 0
            
            while not batch_success and retry_count < max_retries:
                try:
                    logger.info(f"Processing batch {batch_idx + 1}/{total_batches + start_batch} "
                              f"(chunks {i+1}-{min(i+batch_size, len(chunks))}/{len(chunks)})")
                    
                    # Create embeddings for batch
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(batch, embeddings)
                    else:
                        batch_vectorstore = FAISS.from_documents(batch, embeddings)
                        vectorstore.merge_from(batch_vectorstore)
                    
                    # Update progress tracking
                    for chunk in batch:
                        processed_chunks.append(chunk.page_content)
                    
                    chunks_processed_in_session += len(batch)
                    progress_bar.update(len(batch))
                    
                    batch_success = True
                    logger.info(f"Successfully processed batch {batch_idx + 1}")
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = f"Error processing batch {batch_idx + 1}, attempt {retry_count}/{max_retries}: {e}"
                    logger.error(error_msg)
                    
                    if retry_count < max_retries:
                        if exponential_backoff:
                            wait_time = (2 ** retry_count) * delay_between_batches
                        else:
                            wait_time = delay_between_batches * 2
                        
                        logger.info(f"Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exceeded for batch {batch_idx + 1}")
                        # Save checkpoint before failing
                        if vectorstore is not None:
                            checkpointer.save_checkpoint(
                                batch_idx,
                                vectorstore,
                                processed_chunks,
                                {
                                    'pdf_path': pdf_path,
                                    'batch_size': batch_size,
                                    'total_chunks': len(chunks) + len(processed_chunks),
                                    'error': str(e)
                                }
                            )
                        raise e
            
            # Save checkpoint periodically
            if (batch_idx + 1) % checkpoint_every == 0:
                checkpointer.save_checkpoint(
                    batch_idx,
                    vectorstore,
                    processed_chunks,
                    {
                        'pdf_path': pdf_path,
                        'batch_size': batch_size,
                        'total_chunks': len(chunks) + len(processed_chunks),
                        'chunks_processed_in_session': chunks_processed_in_session
                    }
                )
                # Clean up old checkpoints
                checkpointer.cleanup_old_checkpoints(keep_last=3)
            
            batch_idx += 1
            
            # Delay between batches (except for the last batch)
            if i + batch_size < len(chunks):
                time.sleep(delay_between_batches)
    
    finally:
        progress_bar.close()
    
    # Save final checkpoint
    if vectorstore is not None:
        checkpointer.save_checkpoint(
            batch_idx - 1,
            vectorstore,
            processed_chunks,
            {
                'pdf_path': pdf_path,
                'batch_size': batch_size,
                'total_chunks': len(processed_chunks),
                'chunks_processed_in_session': chunks_processed_in_session,
                'completed': True
            }
        )
    
    logger.info(f"Embedding process completed! Processed {len(processed_chunks)} total chunks")
    logger.info(f"{chunks_processed_in_session} chunks processed in this session")
    
    return vectorstore

def main():
    """Main function to run the embedding process"""
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
    
    # Configuration
    """Creates embedding for single PDF file with checkpointing"""
    PDF_PATH = "C:/Users/19368/Repos/Personal/Scaylor Coding Challenge/data/Q32024.pdf"  # Update this path
    BATCH_SIZE = 5  # Smaller batches for rate limiting
    DELAY_BETWEEN_BATCHES = 2.0  # 2 seconds between batches
    CHECKPOINT_EVERY = 3  # Save checkpoint every 3 batches
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Ensure PDF exists
    if not os.path.exists(PDF_PATH):
        logger.error(f"PDF file not found: {PDF_PATH}")
        logger.info("Please update the PDF_PATH variable with your actual PDF file path")
        return
    
    try:
        # Create embeddings with checkpointing
        vectorstore = create_embeddings_with_checkpoint(
            pdf_path=PDF_PATH,
            batch_size=BATCH_SIZE,
            delay_between_batches=DELAY_BETWEEN_BATCHES,
            checkpoint_every=CHECKPOINT_EVERY,
            resume_from_checkpoint=True,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            max_retries=3,
            exponential_backoff=True
        )
        
        # Save final vectorstore
        final_output_path = "final_vectorstore"
        vectorstore.save_local(final_output_path)
        logger.info(f"Final vectorstore saved to: {final_output_path}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Progress has been saved in checkpoints.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.info("Check the embedding_checkpoints/ directory for saved progress")

if __name__ == "__main__":
    main()