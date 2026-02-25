# First, uninstall the wrong fitz package and install the correct one
"""
pip uninstall fitz -y
pip install PyMuPDF
"""

import pymupdf  # Use pymupdf instead of fitz
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

class PDFChatAssistant:
    def __init__(self, pdf_path):
        """Initialize with robust error handling"""
        print("="*60)
        print("📚 PDF CHAT ASSISTANT")
        print("="*60)
        
        # Check for CUDA availability
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"💻 Using device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        
        # Load models with fallbacks
        self.load_models()
        
        # Process PDF
        self.chunks = []
        self.embeddings = None
        self.metadata = {}
        self.load_and_process_pdf(pdf_path)
        
    def load_models(self):
        """Load models with fallback options"""
        try:
            # Try better model first, fall back to lighter model if needed
            print("🔮 Loading embedding model...")
            try:
                self.embed_model = SentenceTransformer("all-mpnet-base-v2")
                print("   ✅ Using all-mpnet-base-v2")
            except:
                self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("   ⚠️ Using all-MiniLM-L6-v2 (lighter model)")
            
            print("🤖 Loading QA model...")
            try:
                self.qa_model = pipeline(
                    "question-answering",
                    model="deepset/roberta-base-squad2",
                    device=self.device
                )
                print("   ✅ Using deepset/roberta-base-squad2")
            except:
                self.qa_model = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=self.device
                )
                print("   ⚠️ Using distilbert-base-cased-distilled-squad (lighter model)")
            
            # Optional summarizer - only load if needed
            self.summarizer = None
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def load_and_process_pdf(self, pdf_path):
        """Extract and process PDF content"""
        try:
            print(f"\n📄 Loading PDF: {pdf_path}")
            
            # Open PDF with pymupdf
            doc = pymupdf.open(pdf_path)
            
            # Extract metadata
            self.metadata = {
                'pages': len(doc),
                'title': doc.metadata.get('title', 'Unknown'),
                'author': doc.metadata.get('author', 'Unknown'),
                'subject': doc.metadata.get('subject', 'Unknown')
            }
            
            # Extract text page by page
            full_text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Add page marker for context
                if page_text.strip():
                    full_text += f"\n[Page {page_num + 1}]\n{page_text}\n"
            
            doc.close()
            
            if not full_text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            # Clean and chunk text
            full_text = self.clean_text(full_text)
            self.chunks = self.smart_chunking(full_text)
            
            print(f"   ✅ Created {len(self.chunks)} text chunks")
            
            # Create embeddings
            print("🔮 Creating embeddings (this may take a moment)...")
            self.embeddings = self.embed_model.encode(
                self.chunks, 
                show_progress_bar=True,
                batch_size=32
            )
            print("   ✅ Embeddings created successfully")
            
            # Show document summary
            self.show_document_stats()
            
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            raise
    
    def clean_text(self, text):
        """Clean extracted text"""
        # Fix common PDF extraction issues
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)    # Remove single newlines
        text = re.sub(r'\n{3,}', '\n\n', text)          # Normalize multiple newlines
        text = re.sub(r'\s+', ' ', text)                # Remove extra spaces
        text = re.sub(r'[^\x20-\x7E\n]', '', text)      # Remove non-ASCII except newlines
        
        return text.strip()
    
    def smart_chunking(self, text, chunk_size=500, overlap=50):
        """Intelligent text chunking"""
        # Split by pages first
        pages = re.split(r'\[Page \d+\]', text)
        pages = [p.strip() for p in pages if p.strip()]
        
        chunks = []
        for page in pages:
            # Split page into paragraphs
            paragraphs = re.split(r'\n\s*\n', page)
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Split long paragraphs into sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                
                current_chunk = ""
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # Check if adding this sentence exceeds chunk size
                    if len((current_chunk + " " + sentence).split()) <= chunk_size:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        # Save current chunk
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # Start new chunk with overlap
                        if chunks and overlap > 0:
                            words = chunks[-1].split()
                            overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else chunks[-1]
                            current_chunk = overlap_text + " " + sentence
                        else:
                            current_chunk = sentence
                
                # Add last chunk
                if current_chunk:
                    chunks.append(current_chunk)
        
        return chunks
    
    def show_document_stats(self):
        """Display document statistics"""
        total_words = sum(len(chunk.split()) for chunk in self.chunks)
        print("\n📊 Document Statistics:")
        print(f"   - Pages: {self.metadata['pages']}")
        print(f"   - Chunks: {len(self.chunks)}")
        print(f"   - Total words: ~{total_words}")
        print(f"   - Title: {self.metadata['title']}")
        if self.metadata['author'] != 'Unknown':
            print(f"   - Author: {self.metadata['author']}")
    
    def find_relevant_chunks(self, question, k=5):
        """Find most relevant chunks for a question"""
        # Encode question
        question_embedding = self.embed_model.encode(question)
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, question_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(question_embedding)
        )
        
        # Get top k chunks
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return chunks with scores
        relevant = []
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Only keep reasonably similar chunks
                relevant.append({
                    'text': self.chunks[idx],
                    'score': similarities[idx],
                    'index': idx
                })
        
        return relevant
    
    def ask(self, question):
        """Answer a question about the PDF"""
        try:
            # Handle special commands
            if question.lower() in ['summarize', 'summary']:
                return self.summarize_document()
            elif question.lower() in ['stats', 'statistics']:
                return self.get_formatted_stats()
            
            # Find relevant chunks
            relevant_chunks = self.find_relevant_chunks(question)
            
            if not relevant_chunks:
                return "I couldn't find relevant information in the document. Try rephrasing your question."
            
            # Combine top chunks for context
            context = " ".join([chunk['text'] for chunk in relevant_chunks[:3]])
            
            # Truncate context if too long (model has max length)
            if len(context.split()) > 400:
                context = " ".join(context.split()[:400])
            
            # Get answer from QA model
            result = self.qa_model(
                question=question,
                context=context,
                handle_impossible_answer=True
            )
            
            # Check if answer is valid
            if result['score'] > 0.1 and result['answer'].strip():
                answer = result['answer'].strip()
                
                # Add confidence indicator
                if result['score'] > 0.5:
                    confidence = "✅ High confidence"
                elif result['score'] > 0.2:
                    confidence = "⚠️ Medium confidence"
                else:
                    confidence = "❓ Low confidence"
                
                return f"{answer}\n[{confidence}]"
            else:
                return "I couldn't find a specific answer. Try rephrasing your question."
                
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def summarize_document(self):
        """Generate a summary of the document"""
        if not self.summarizer:
            return "Summarization model not loaded. Install transformers with summarization support."
        
        try:
            # Take first few chunks for summary
            text_to_summarize = " ".join(self.chunks[:3])
            
            if len(text_to_summarize.split()) > 500:
                text_to_summarize = " ".join(text_to_summarize.split()[:500])
            
            summary = self.summarizer(
                text_to_summarize,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            
            return summary[0]['summary_text']
        except:
            return "Could not generate summary. The summarization model may not be available."
    
    def get_formatted_stats(self):
        """Get formatted document statistics"""
        stats = f"""
📊 Document Statistics:
   - Pages: {self.metadata['pages']}
   - Text chunks: {len(self.chunks)}
   - Total words: ~{sum(len(chunk.split()) for chunk in self.chunks)}
   - Title: {self.metadata['title']}
   - Author: {self.metadata['author']}
        """
        return stats.strip()
    
    def chat(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("💬 Start chatting with your PDF!")
        print("Commands: 'summarize', 'stats', 'help', 'exit'")
        print("="*60)
        
        while True:
            try:
                # Get user input
                question = input("\n📝 You: ").strip()
                
                # Check for exit
                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                # Check for help
                if question.lower() == 'help':
                    self.show_help()
                    continue
                
                # Skip empty questions
                if not question:
                    continue
                
                # Get answer
                print("🤖 Assistant: ", end="", flush=True)
                answer = self.ask(question)
                print(answer)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def show_help(self):
        """Show help information"""
        help_text = """
📚 PDF Chat Assistant Help
==========================
Commands:
  - ask any question about the PDF content
  - 'summarize' - get a brief summary
  - 'stats' - show document statistics
  - 'help' - show this help
  - 'exit' - quit the program

Tips:
  - Be specific in your questions
  - Ask about names, dates, concepts, etc.
  - The assistant will show confidence levels
        """
        print(help_text)

def main():
    """Main function"""
    import sys
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("📂 Enter PDF path: ").strip()
    
    try:
        # Initialize chat assistant
        assistant = PDFChatAssistant(pdf_path)
        
        # Start chatting
        assistant.chat()
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{pdf_path}'")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()