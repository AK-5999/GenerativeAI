
# 1. RAG Component Deep-Dive & Implementation

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

# 2. Handling Complex Inputs (Multimedia)

* **Tables:** Ingestion ke waqt `unstructured` use karke table ko HTML/Markdown mein badlein taaki structure save rahe.
* **Photos/Logos:** `GPT-4o` (Vision) se image ka text description nikaal kar use vector store mein daalein.
* **Videos/Audio:** `OpenAI Whisper` se transcription nikaalein aur timestamps ke saath chunking karein.

---

# 3. Real-World Challenges Summary

1. **Semantic Gap:** User ne query mein code dala (e.g., "IPC 302") par context nahi diya.
* *Solution:* **Hybrid Search** ya **Query Expansion**.


2. **K-Means Bottleneck:** Retrieval ke waqt data clusters mein hona fast hai par semantic matching weak ho sakti hai.
* *Solution:* Retrieval ke baad **Cross-Encoder Re-ranking** karein.


3. **Data Ingestion Mess:** Scanned tables ka data row/column mix ho jana.
* *Solution:* Vision-based OCR engines (Azure/AWS/Unstructured).



---

# 4. Final Advanced RAG Pipeline (LCEL)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Components setup
llm = ChatOpenAI(model="gpt-4o")
retriever = vectorstore.as_retriever(search_type="mmr") 

# Prompt template
template = """Answer the question based ONLY on the context. 
If not in context, say 'I don't know'.
Context: {context}
Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

# Chain execution
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Usage
result = chain.invoke("Explain Section 420 codes?")
print(result.content)

```

