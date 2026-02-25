# talking-pdf


PDF
 ↓
Text extraction
 ↓
Chunking
 ↓
Embedding
 ↓
Vector database
 ↓
User question
 ↓
Find relevant text
 ↓
LLM generates answer



| Task           | Tool                        |
| -------------- | --------------------------- |
| PDF extraction | PyMuPDF, pdfplumber         |
| NLP model      | Transformers, OpenAI, LLaMA |
| Embeddings     | sentence-transformers       |
| Vector storage | ChromaDB, FAISS             |
| Full framework | LangChain                   |



PDF → embedding vector → usable for:
   • asking questions ❓
   • searching 🔎
   • summarizing 📄
   • classification 🧠
   • DSS recommendations ⚕️






model="google/flan-t5-xl"        # very good
model="mistralai/Mistral-7B-Instruct-v0.2"   # excellent
model="meta-llama/Meta-Llama-3-8B-Instruct"  # best quality