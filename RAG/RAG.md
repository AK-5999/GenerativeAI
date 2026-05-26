# 🚀 Ultimate RAG Pipeline Guide: From Concepts to Code

Yeh document ek complete guide hai RAG (Retrieval-Augmented Generation) pipeline ko LangChain ke saath samajhne aur implement karne ke liye.

---

## 1. RAG Core Pipeline (The 6 Pillars)

RAG pipeline ka standard flow: **Load → Split → Embed → Store → Retrieve → Generate**.

### Step 1: Ingestion (Data Extraction)
**Theory:** Raw data (PDF, Web, Image) ko clean text mein badalna.
**Options:**
- **Standard PDF:** `PyPDFLoader` (Simple text).
- **Complex Layout/OCR:** `UnstructuredPDFLoader` (Tables aur scanned docs).
- **Web Pages:** `WebBaseLoader` ya `FireCrawl` (JavaScript heavy sites).

```python
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

# Implementation
loader = PyPDFLoader("manual.pdf")
raw_docs = loader.load()

```

### Step 2: Chunking (Text Splitting)

**Theory:** Bade docs ko chote pieces mein todna taaki LLM ki context window mein fit ho sake.
**Options:**

* **Recursive Character:** Default aur best. Paragraphs aur sentences ko maintain karta hai.
* **Semantic Chunking:** Meaning change hone par break karta hai.
* **Token Splitting:** Exact word count ke hisab se.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Implementation
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(raw_docs)

```

### Step 3: Embeddings (Vectorization)

**Theory:** Text ko mathematical numbers (Vectors) mein badalna jo meaning capture karte hain.
**Options:**

* **OpenAI:** `text-embedding-3-small` (Top Accuracy).
* **HuggingFace:** `bge-large-en` (Local & Privacy-friendly).

```python
from langchain_openai import OpenAIEmbeddings

# Implementation
embeddings = OpenAIEmbeddings()

```

### Step 4: Vector Storage & Indexing

**Theory:** Vectors ko store karna aur search fast banane ke liye **K-Means Clustering** ya **IVF Indexing** ka use karna.
**Options:**

* **Local:** `Chroma`, `FAISS`.
* **Cloud:** `Pinecone`, `Weaviate`.

```python
from langchain_community.vectorstores import Chroma

# Implementation (K-Means backend indexing)
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="./db_index"
)

```

### Step 5: Advanced Retrieval Strategies

**Theory:** Sahi context dhoondhne ke alag-alag tarike.

| Strategy | Logic | Use Case |
| --- | --- | --- |
| **Similarity** | Semantic match | General Chat |
| **Hybrid Search** | Vector + Keyword | Exact codes (e.g. "IPC-420") |
| **MMR** | Diversity | Repetitive answers se bachne ke liye |
| **Re-ranking** | Cross-Encoder Check | High Accuracy requirements |

```python
# Implementation: MMR & Re-ranking
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

base_retriever = vectorstore.as_retriever(search_type="mmr")
reranker = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=base_retriever
)

```

### Step 6: Generation (LLM & LCEL)

**Theory:** Context aur Query ko milakar final answer banana.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Chain Setup
prompt = ChatPromptTemplate.from_template("Answer based on context: {context}\n\nQuestion: {question}")
llm = ChatOpenAI(model="gpt-4o")

rag_chain = (
    {"context": compression_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

```

---

## 2. Handling Diverse Input Formats

| Data Type | Backend Process | Best Tool |
| --- | --- | --- |
| **Tables** | HTML/Markdown mein badalna | `Azure Doc Intelligence` |
| **Photos/Logos** | Vision Model se text captioning | `GPT-4o Vision` |
| **Videos/Audio** | Speech-to-Text Transcription | `OpenAI Whisper` |
| **Links** | Boilerplate (Ads/Menu) cleaning | `FireCrawl` |

---

## 3. Real-World Challenges & Solutions

### Q1: "Semantic Word" missing ho toh kya hoga?

**Challenge:** Agar user sirf code search kare (e.g. "XYZ-123") jisme koi semantic meaning nahi hai, toh K-Means clustering galat cluster mein bhej degi.
**Solution:** **Hybrid Search** use karein jo Keyword (BM25) aur Vector search ko combine karta hai.

### Q2: Data Security kaise handle karein?

**Challenge:** Har employee ko har document ka access nahi hona chahiye.
**Solution:** **Metadata Filtering** use karein. Ingestion ke waqt role-based tags lagayein.

### Q3: Tables/Layout kharab hona?

**Challenge:** Normal parsers columns ko mix kar dete hain.
**Solution:** Layout-aware parsers use karein jo table structure ko HTML format mein save karein.

---

## 4. Query Routing (The Traffic Cop)

Bade systems mein, routing decide karti hai ki sawal kis database mein jayega.

```python
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field

class RouteQuery(BaseModel):
    destination: Literal["technical_db", "billing_db"] = Field(..., description="Route to DB")

router = llm.with_structured_output(RouteQuery)

```

---

## 5. Summary Cheat Sheet

* **Fastest Search:** Bi-Encoders + FAISS.
* **Accurate Answer:** Cross-Encoders (Re-ranking).
* **Big Dataset:** K-Means Clustering for indexing.
* **Dirty Data:** Unstructured/Vision Parsers.



---

## 6. RAG Component Deep-Dive & Implementation

## A. Ingestion Methods (Data Extraction)
Real-world data formats ke liye alag-alag loaders use hote hain.

| Method | Type | Tool/Class | Kab Use Karein? |
| :--- | :--- | :--- | :--- |
| **Standard PDF** | Text-based | `PyPDFLoader` | Simple structured PDFs ke liye. |
| **OCR/Layout** | Image/Complex | `UnstructuredPDFLoader` | Scanned docs ya multi-column layout ke liye. |
| **Web Scraping** | Live URLs | `FireCrawlLoader` | Clean web content extraction ke liye. |

**Practical Code:**
```python
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

# 1. Standard PDF
loader = PyPDFLoader("data.pdf")
docs = loader.load()

# 2. Complex/Scanned PDF (OCR)
# requires: pip install unstructured
loader_ocr = UnstructuredPDFLoader("scanned_data.pdf", mode="elements")
docs_complex = loader_ocr.load()

```

---

## B. Chunking Methods (Text Splitting)

Data ko todne ka tarika retrieval accuracy decide karta hai.

| Method | Logic | Tool/Class |
| --- | --- | --- |
| **Recursive** | Characters & Separators (`\n\n`, `\n`) | `RecursiveCharacterTextSplitter` |
| **Token-based** | LLM Token count (tiktoken) | `TokenTextSplitter` |
| **Semantic** | Meaning/Embedding change | `SemanticChunker` |

**Practical Code:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# 1. Standard (Recursive)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# 2. Semantic (Advanced)
semantic_splitter = SemanticChunker(OpenAIEmbeddings())
semantic_chunks = semantic_splitter.create_documents([docs[0].page_content])

```

---

## C. Retrieval Methods (Search Strategies)

Sirf simple search hamesha kaam nahi aati, isliye different retrieval types use hote hain.

| Method | Backend Logic | Kab Use Karein? |
| --- | --- | --- |
| **Similarity Search** | Distance between vectors | Default/General use-case. |
| **Hybrid Search** | Vector + Keyword (BM25) | Jab exact numeric codes (e.g., ERR_404) dhoondne hon. |
| **MMR Search** | Diversity + Relevance | Jab repetitive results se bachna ho. |
| **Multi-Query** | User query ke variations bana kar | Jab user ki query clear na ho. |

**Practical Code:**

```python
# 1. Maximum Marginal Relevance (MMR)
retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5})

# 2. Metadata Filtering (Direct Hit)
retriever_filter = vectorstore.as_retriever(
    search_kwargs={'filter': {'category': 'legal'}}
)

# 3. Multi-Query Retrieval
from langchain.retrievers.multi_query import MultiQueryRetriever
mq_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

```

---

