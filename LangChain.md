# LangChain
**LangChain** is a framework for building applications using **large language models (LLMs)**. 
It helps in connecting **LLMs** to external data, APIs, and tools, enabling easier creation of **AI-powered applications** like chatbots and automation systems.

LangChain remains one of the best because of its:
  - **Extensive integration** with various tools and APIs.
  - Modular framework for **custom** workflows.
  - **Strong community support** and active development.
  - **Seamless** document handling and retrieval-augmented generation capabilities.

Key components:
1. **Chains**: Sequences of tasks or operations that can be executed together (e.g., combining LLMs with other tools).
2. **Agents**: Autonomous entities that decide how to interact with various tools or environments based on input.
3. **Memory**: Allows storing information across multiple interactions for context-aware conversations.
4. **Toolkits**: Pre-built integrations to connect **LLMs** with external APIs, databases, and services.
5. **Prompts**: Templates that guide the input to the LLMs for specific tasks or outputs.

Other Important Components:
## 1. **Text**: The natural language way to interact with LLMs
## 2. **Chat Messages**: Chat messages refer to the communication exchanged between the user and the AI model.
    - System: Helpful background context that tell the AI what to do
    - Human: Messages that are intented to represent the user
    - Messages: that show what the AI responded with
## 3. **Document**: An object that holds a piece of text and metadata (id, source, time)
## 4. **Function Calling Modles**: Function calling models enable LLMs to invoke external tools.
    - from langchain.llms import OpenAI
    - llm = OpenAI(model_name="text-ada-001", openai_api_key=openai_api_key)
## 5. **Embedding Models**: Embedding models convert text (or other data) into vector representations.
  - **OpenAI embeddings** -> text-embedding-ada-002.
  - **Hugging Face Embeddings** -> BERT, DistilBERT, RoBERTa.
    - from langchain.embeddings import OpenAIEmbeddings 
    - huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    - sentence_transformer_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    - embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    - text_embedding = embeddings.embed_query(text)
## 7. **Example Selectors**: Example selectors are used to choose or filter specific examples from a dataset or set of documents.

| **Example Selector**                               | **Description**                                                                 | **Use Case**                                                                                     | **Example**                                                                                                      |
|----------------------------------------------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **`SemanticSimilarityExampleSelector`**            | Selects examples based on **semantic similarity** to the query using embeddings. | Best for tasks requiring contextual relevance where the examples need to match the input query.  | Selecting similar examples for a query using **semantic embeddings** to ensure high relevance.                  |
| **`MaximalMarginalRelevanceExampleSelector`**      | Chooses examples based on a combination of **relevance** and **diversity**.       | Suitable when both relevance to the query and diversity of examples are important (e.g., for balanced training data). | Selecting diverse yet relevant examples to avoid redundancy and provide variety.                                  |
| **`ContextualExampleSelector`**                    | Selects examples based on **semantic similarity** along with additional **contextual information** (e.g., metadata). | Useful when you have additional metadata or contextual information that should influence example selection. | Using contextual metadata (e.g., source, author) to filter or prioritize examples along with their semantic relevance. |
| **`RandomExampleSelector`**                        | Randomly selects a few examples from the pool.                                    | When you need random sampling or want to avoid bias in selecting examples for testing or evaluation. | Selecting random examples from a pool for A/B testing, evaluation, or diversity in training.                   |

## 8. **Memoy**: the ability of the system to remember and maintain information across multiple interactions or queries.


| **Memory Type**                         | **Description**                                                                 | **Use Case**                                                                                   | **Example**                                                                                                      |
|-----------------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **`ConversationBufferMemory`**           | Stores a **buffer** of conversation history (inputs and outputs) during the session. | Basic chat applications where recent conversation context is needed.                          | Simple conversational agent where input/output history is maintained temporarily for ongoing interactions.       |
| **`ConversationBufferWindowMemory`**    | Stores a **sliding window** of recent conversations (fixed size).                 | Real-time conversation applications that only need recent context and don't require long-term memory. | Chat application where only the last 3-5 exchanges are stored for context.                                       |
| **`ConversationTokenBufferMemory`**     | Tracks conversation history in **tokens** to manage token limits in large models. | Scenarios where token management is crucial, and you need to stay within a token budget.        | Large conversational AI handling long dialogues without exceeding token limits.                                  |
| **`EntityMemory`**                      | Tracks **structured data** or entities like user preferences, names, or locations. | Personalized interactions where specific user data needs to be retained across conversations.   | Chatbot remembering user preferences like their name or favorite color.                                         |
| **`ChatMessageHistory`**                | Stores individual **chat messages** for detailed history tracking.               | Tracking message history in systems requiring granular access to conversation details.         | Debugging conversations, or systems requiring detailed inspection of message exchanges.                         |
| **`VectorStoreMemory`**                 | Integrates with a **vector store** (e.g., FAISS) to store and retrieve vectors (embeddings) of the conversation. | Tasks requiring semantic memory and retrieval, such as RAG (retrieval-augmented generation) applications. | Semantic search-based chatbot that fetches relevant documents or contexts for accurate response generation.      |

## 9. **Chain**:  A sequence of components that are linked together to process input data, often resulting in a final output.

| **Chain Type**               | **Description**                                                                 | **Use Case**                                                                                  | **Example**                                                                                                      |
|------------------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **`LLMChain`**                | A chain that links a language model (LLM) with a prompt template for text generation. | Basic text generation tasks like Q&A, summarization, or translation.                          | Generating a translation or summary using a simple LLM chain.                                                   |
| **`SequentialChain`**         | Links multiple chains in a sequence where the output of one chain is passed to the next. | Multi-step workflows, such as translation followed by summarization or answering complex questions. | Chaining translation and summarization tasks sequentially.                                                    |
| **`RouterChain`**             | Routes input data to different chains based on a specific condition or logic. | Conditional workflows, e.g., directing queries about translations to one chain and questions to another. | Routing queries for translations or explanations to different chains based on input conditions.               |
| **`SimpleChain`**             | A simplified version of the `SequentialChain`, linking multiple chains without customization. | Quick prototyping or simple task chains where multiple operations are needed in sequence.       | Simple chain linking tasks such as text summarization, translation, or summarization followed by a generation task. |
| **`MapReduceChain`**          | Breaks tasks into smaller chunks, processes them in parallel (map), and reduces them into a final output. | Large document processing, parallel computation for summarization or data extraction tasks.     | Breaking down a large document and summarizing sections before aggregating them into a final summary.           |
| **`RetrievalQAChain`**        | Combines a retriever with a language model for question answering, often used for RAG (retrieval-augmented generation). | Question answering over large datasets, using document retrieval for context.                  | Answering questions using information retrieved from a vector database or document corpus.                      |
| **`AgentExecutorChain`**      | Allows the creation of autonomous agents that decide on which actions to take next based on logic. | Autonomous decision-making, AI assistants, and dynamic workflows requiring multiple actions.    | An AI agent querying multiple sources or making decisions to execute tasks like data retrieval or interaction.   |

## 10. **Agent**: Dynamically select which tool model has to use to achieve a goal.
  - Decision-making: Make decisions based on the inputs they receive. These decisions are typically made by querying an LLM (Language Model) that helps decide the next action or step.
  - Actions: Perform actions based on the context provided. These actions can involve calling external APIs, interacting with databases, querying memory, or running other LangChain components (like chains).
  - Tools: Use various tools such as external APIs, databases, LLMs, or vector stores to gather information and perform actions that lead them closer to the goal.
  - Execution: Actions in a sequence or based on its logic and might re-evaluate its course of action if it detects that its current plan is not leading to a successful outcome.
  - Memory: Use memory to remember previous interactions or context, enabling it to adapt its actions based on historical data. For example, the agent may remember the user’s preferences, previous answers, or feedback.


| **Agent Type**                       | **Description**                                                                | **Use Case**                                                                                        | **Example**                                                                                                      |
|--------------------------------------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **`Zero-shot-react-description`**    | A **zero-shot** agent that uses a language model to react to tasks without training for specific actions. | General-purpose agent for tasks that do not require specific training or fine-tuning.               | Performing tasks based on a general prompt or instruction (e.g., summarizing text, querying knowledge).         |
| **`Zero-shot`**                      | An agent that doesn't have prior training but reacts based on inputs and a predefined prompt.   | Simple queries or when you don't need a pre-trained model for specific actions.                    | A chatbot responding to a wide range of user queries without being explicitly trained for those queries.         |
| **`Self-reflective`**                 | An agent that can **reflect** on its actions and adjust its behavior accordingly. | More advanced agents where self-correction or adjustment of actions based on feedback is necessary. | A task agent that reviews the outcome of its actions and makes corrections based on feedback or new conditions.    |
| **`Tool-using`**                     | An agent that utilizes **external tools** (APIs, databases) to retrieve or perform actions. | Complex workflows where external tools or APIs are needed to execute specific tasks.                | A research agent that queries databases or performs actions like retrieving data from APIs for further analysis.  |
| **`AgentExecutor`**                  | Executes a series of **actions** and may re-evaluate the next action based on feedback or results. | Multi-step tasks that involve executing various actions in a sequence, with decision-making involved. | An agent executing a sequence of actions (e.g., querying a database, processing the result, and generating a report).|
| **`Custom Agents`**                  | Custom agents designed for specific tasks, integrating multiple tools and chains. | When you need specialized behavior tailored to a particular domain or task.                          | A custom agent for real-time trading, querying financial data, analyzing trends, and making trading decisions.     |
