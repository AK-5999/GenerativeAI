
```markdown
# 1. Introduction to RAG (Retrieval-Augmented Generation)

**Theory:** RAG ek aesa framework hai jo LLM (jaise GPT-4) ko external data se connect karta hai. LLM ki knowledge limited hoti hai, RAG use "Open Book Exam" dene ki taqat deta hai.

**Layman Term:** Maan lijiye AI ek expert vakil hai, lekin use aapke personal case ki details nahi pata. RAG use case ki files la kar deta hai taaki wo sahi jawab de sake.

---

# 2. Crucial Components & Practical Implementation

## Step 1: Ingestion (Data Extraction)
**Options:** - `PyPDFLoader`: Simple text-based PDFs.
- `Unstructured`: Complex layout, tables, aur images ke liye.
- `WebBaseLoader`: Websites se data nikalne ke liye.

```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# PDF Loading
pdf_loader = PyPDFLoader("manual.pdf")
docs = pdf_loader.load()

# Web Loading
web_loader = WebBaseLoader("[https://example.com](https://example.com)")
web_docs = web_loader.load()

```

## Step 2: Chunking (Text Splitting)

**Options:**

* `RecursiveCharacterTextSplitter`: Sabse best, paragraph/sentence structure maintain karta hai.
* `SemanticChunking`: Meaning ke basis par tukde karta hai.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)

```

## Step 3: Embeddings (Text to Vectors)

**Options:**

* `OpenAIEmbeddings`: Paid, high accuracy.
* `HuggingFaceEmbeddings`: Free, local privacy ke liye.

```python
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

```

## Step 4: Vector Storage (Filing Cabinet)

**Options:**

* `Chroma` / `FAISS`: Local storage (K-Means clustering backend mein use karte hain).
* `Pinecone`: Cloud-based scalable storage.

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model,
    persist_directory="./my_db"
)

```

---

# 3. Advanced Retrieval Techniques

## Query Routing

**Theory:** Jab multiple databases hon, toh router decide karta hai query kahan jayegi.

```python
# LLM based Router (Pydantic)
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

class RouteQuery(BaseModel):
    datasource: Literal["finance_db", "hr_db"] = Field(..., description="Route query to specific DB")

router_llm = llm.with_structured_output(RouteQuery)

```

## Re-ranking (Cross-Encoders)

**Theory:** Bi-encoders (fast search) top results dete hain, Cross-encoders unhe refine karke best rank karte hain.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

reranker = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, 
    base_retriever=vectorstore.as_retriever()
)

```

---

# 4. Handling Different Data Types

| Data Type | Processing Method | Tools |
| --- | --- | --- |
| **Tables** | Markdown/HTML conversion | `Unstructured`, `Azure Doc Intelligence` |
| **Photos/Logos** | Vision Captioning (GPT-4o) | `langchain_openai.ChatOpenAI` |
| **Videos** | Audio Transcription (Whisper) | `openai-whisper`, `FFmpeg` |
| **Web Links** | HTML Scraping & Cleaning | `FireCrawl`, `BeautifulSoup` |

---

# 5. Real-World Challenges (Production Issues)

1. **Messy Layouts:** Tables normal parser se read nahi ho pati (Solution: Layout-aware parsing).
2. **Retrieval Failure:** Jab user numeric codes (e.g. "ERR_404") search kare bina semantic words ke.
* **Solution:** **Hybrid Search** (Keyword + Vector).


3. **Stale Data:** Database update nahi hota (Solution: Webhooks & Automated ETL).
4. **Privacy:** Secret data leak hona (Solution: Metadata filtering by user role).

---

# 6. Complete Pipeline Construction (LCEL)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 1. Prompt
template = "Answer based on context: {context}\nQuestion: {question}"
prompt = ChatPromptTemplate.from_template(template)

# 2. Model
llm = ChatOpenAI(model="gpt-4o")

# 3. Chain
rag_chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 4. Execution
response = rag_chain.invoke("What are the key findings?")
print(response.content)


```
