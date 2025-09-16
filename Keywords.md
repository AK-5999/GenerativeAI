# 🌐 NLP Glossary – Key Terms & Definitions

A curated list of important terms in Natural Language Processing (NLP), along with concise definitions. Perfect for quick reference!

---

## 📌 Basic Terms

- **[Tokenization](https://github.com/AK-5999/NOTES/blob/main/Transformers.md#7-tokenization-methods-comparison)**
  Splitting text into individual units like words, subwords, or characters.... POINT: 7

- **Stop Words**  
  Common words (e.g., *the*, *is*, *and*) often removed from text during preprocessing due to low semantic value.

- **Lemmatization**  
  Reducing words to their dictionary form (lemma). Example: "running" → "run".

- **Stemming**  
  Truncating words to their base form. Example: "running" → "runn".

---

## 🧠 [Embeddings & Representations](https://github.com/AK-5999/NOTES/blob/main/Transformers.md#10-feature-extraction-methods-comparison)
- point 10
- **Word Embedding**  
  Vector representation of words capturing semantic meaning (e.g., Word2Vec, GloVe).

- **Bag of Words (BoW)**  
  Text representation based on word counts, ignoring grammar and order.

- **TF-IDF (Term Frequency–Inverse Document Frequency)**  
  Weights terms based on how unique they are across documents.

- **Sentence Embedding**  
  Encodes full sentences as dense vectors for comparison or input to models.

---

## 🏷️ Annotation & Tagging

- **Part-of-Speech (POS) Tagging**  
  Labeling words with their grammatical role (noun, verb, adjective, etc.).

- **Named Entity Recognition (NER)**  
  Identifying names of people, places, dates, organizations in text.

- **Dependency Parsing**  
  Analyzing grammatical structure by linking words based on syntactic dependencies.

- **Coreference Resolution**  
  Detecting when different expressions refer to the same entity (e.g., "Alice... she... her").

---

## 🔍 Semantics & Context

- **Semantic Similarity**  
  Measures how close two texts are in meaning.

- **Word Sense Disambiguation (WSD)**  
  Identifying the correct meaning of a word based on context.

- **Semantic Parsing**  
  Converting natural language into structured meaning representations.

- **Textual Entailment**  
  Determining if one sentence logically follows from another.

---

## ⚙️ Model Types & Architectures

- **Language Model (LM)**  
  Predicts the next word/token in a sequence (e.g., GPT, BERT).

- **[Transformer](https://github.com/AK-5999/NOTES/blob/main/Transformers.md)**  
  Architecture using self-attention to process input in parallel (replacing RNNs/LSTMs).

- **[RNN (Recurrent Neural Network)](https://github.com/AK-5999/NOTES/blob/main/Transformers.md#8-lstm-vs-rnn)**  
  Processes sequences one element at a time; struggles with long-term dependencies.......POINT:8

- **[LSTM (Long Short-Term Memory)](https://github.com/AK-5999/NOTES/blob/main/Transformers.md#8-lstm-vs-rnn)**  
  A type of RNN that uses gates to better capture long-range dependencies..........POINT:8

---

## ✨ [Attention Mechanisms](https://github.com/AK-5999/NOTES/blob/main/Transformers.md#3-attention-mechanisms-in-transformers)

- POINT: 3
- **Attention**  
  Focuses on relevant parts of input when generating output.

- **Self-Attention**  
  A token attends to other tokens in the same sequence (used in Transformers).

- **Cross-Attention**  
  Decoder attends to encoder output in seq2seq models.

---

## 🧪 Evaluation Metrics

### ➤ [Classification](https://github.com/AK-5999/NOTES/blob/main/DeepLearning.md#7-evaluation-metrics-for-categorical-classification)

- **Accuracy**  
  Ratio of correct predictions to total predictions.

- **Precision**  
  Correct positive predictions / Total positive predictions.

- **Recall**  
  Correct positive predictions / Actual positives.

- **F1 Score**  
  Harmonic mean of precision and recall.

### ➤ [Text Generation](https://github.com/AK-5999/NOTES/blob/main/DeepLearning.md#8-evaluation-metrics-for-text-generation)

- **BLEU (Bilingual Evaluation Understudy)**  
  Measures n-gram overlap with reference translations.

- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**  
  Compares overlapping sequences (n-gram, LCS).

- **METEOR**  
  Considers precision, recall, synonyms, and alignment.

- **Perplexity**  
  Measures how well a language model predicts a sequence.

- **BERTScore**  
  Uses contextual embeddings from BERT to compare similarity.

---

## 🛠️ Pretraining & Fine-Tuning

- **Pretraining**  
  Training on large generic data to learn general language understanding.

- **[Fine-Tuning](https://github.com/AK-5999/NOTES/blob/main/Finetuning.md)**  
  Further training a pretrained model on a specific task.

- **Transfer Learning**  
  Applying knowledge from one task/domain to another.

---

## 🧩 Other Key Terms

- **Zero-shot Learning**  
  Model performs a task without having seen examples during training.

- **Few-shot Learning**  
  Model performs a task with very few labeled examples.

- **Prompt Engineering**  
  Designing input prompts to guide LLM behavior.

- **Knowledge Distillation**  
  Compressing a large model’s knowledge into a smaller model.

- **Hallucination (in LLMs)**  
  When a language model generates factually incorrect but plausible-sounding content.

---

## 🧠 Useful NLP Libraries

- **NLTK**  
  Natural Language Toolkit for traditional NLP tasks.

- **spaCy**  
  Fast, industrial-strength NLP library with pretrained models.

- **Transformers (Hugging Face)**  
  Library to use and fine-tune transformer-based models (BERT, GPT, etc.).

- **Gensim**  
  Specializes in topic modeling and word embeddings like Word2Vec.

---

## Cloud
- How would you deploy a generative AI model using AWS services? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#4-how-would-you-deploy-a-generative-ai-model-using-aws-services)
- How does Amazon Bedrock simplify the use of foundation models? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#5-how-does-amazon-bedrock-simplify-the-use-of-foundation-models)
- Compare SageMaker with Bedrock. When would you use one over the other? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#6-compare-sagemaker-with-bedrock-when-would-you-use-one-over-the-other)
- How do you optimize cost and performance when training LLMs on AWS? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#7-how-do-you-optimize-cost-and-performance-when-training-llms-on-aws)
- Which instance types are suitable for GenAI on AWS? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#8-which-instance-types-are-suitable-for-genai-on-aws)
- How would you set up distributed training for large models in SageMaker? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#9-how-would-you-set-up-distributed-training-for-large-models-in-sagemaker)
- What are the key security considerations for GenAI on AWS? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#10-what-are-the-key-security-considerations-for-genai-on-aws)
- How do you ensure data privacy and compliance in Bedrock or SageMaker? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#11-how-do-you-ensure-data-privacy-and-compliance-in-bedrock-or-sagemaker)
- Suppose your client wants a chatbot powered by GenAI that can access private enterprise documents. How would you design and deploy it on AWS? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#12-suppose-your-client-wants-a-chatbot-powered-by-genai-that-can-access-private-enterprise-documents-how-would-you-design-and-deploy-it-on-aws)
- How would you fine-tune an open-source LLM on AWS? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#13-how-would-you-fine-tune-an-open-source-llm-on-aws)
- How do you handle prompt engineering in Bedrock and check output quality? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#14-how-do-you-handle-prompt-engineering-in-bedrock-and-check-output-quality)
- How would you implement CI/CD for GenAI with SageMaker? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#15-how-would-you-implement-cicd-for-genai-with-sagemaker)
- What are the limitations of GenAI, and how does AWS help? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#16-what-are-the-limitations-of-genai-and-how-does-aws-help)
- What is RAG, and how do you implement it on AWS? [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#17-what-is-rag-and-how-do-you-implement-it-on-aws)
- Services [Answer](https://github.com/AK-5999/NOTES/blob/main/Cloud.md#18-service-of-aws)
