# 🌐 NLP Glossary – Key Terms & Definitions

A curated list of important terms in Natural Language Processing (NLP), along with concise definitions. Perfect for quick reference!

---

## 📌 Basic Terms

- **Tokenization**  
  Splitting text into individual units like words, subwords, or characters.

- **Stop Words**  
  Common words (e.g., *the*, *is*, *and*) often removed from text during preprocessing due to low semantic value.

- **Lemmatization**  
  Reducing words to their dictionary form (lemma). Example: "running" → "run".

- **Stemming**  
  Truncating words to their base form. Example: "running" → "runn".

---

## 🧠 Embeddings & Representations

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

- **Transformer**  
  Architecture using self-attention to process input in parallel (replacing RNNs/LSTMs).

- **RNN (Recurrent Neural Network)**  
  Processes sequences one element at a time; struggles with long-term dependencies.

- **LSTM (Long Short-Term Memory)**  
  A type of RNN that uses gates to better capture long-range dependencies.

---

## ✨ Attention Mechanisms

- **Attention**  
  Focuses on relevant parts of input when generating output.

- **Self-Attention**  
  A token attends to other tokens in the same sequence (used in Transformers).

- **Cross-Attention**  
  Decoder attends to encoder output in seq2seq models.

---

## 🧪 Evaluation Metrics

### ➤ Classification

- **Accuracy**  
  Ratio of correct predictions to total predictions.

- **Precision**  
  Correct positive predictions / Total positive predictions.

- **Recall**  
  Correct positive predictions / Actual positives.

- **F1 Score**  
  Harmonic mean of precision and recall.

### ➤ Text Generation

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

- **Fine-Tuning**  
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

📘 *This glossary is regularly updated. You can contribute or suggest edits to keep it fresh!*
