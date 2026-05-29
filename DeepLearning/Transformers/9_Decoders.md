# Decoders:

Decoders behaves like auto-regressive during inference time but not auto-regrssive during training time.

# Decoder-only architecture (like GPT-3, GPT-4, or Llama), it can feel like a piece of the puzzle is missing.

* ***When there is no encoder, cross-attention is completely removed.** Here is exactly how it works and why we don't need it.

---

## 1. The Short Circuit: Skipping Cross-Attention

In a standard Encoder-Decoder Transformer, cross-attention exists so the decoder can "look back" at the encoder's final hidden states. It uses the Decoder's hidden state as the **Query ($Q$)**, and the Encoder's outputs as the **Key ($K$)** and **Value ($V$)**.

In a Decoder-only architecture, there is no encoder to provide those Keys and Values. Therefore, the entire cross-attention sub-layer is simply deleted from the network block.

Instead, the architecture looks like this:

1. **Masked Self-Attention:** The tokens look backward at themselves and previous tokens.
2. **Feed-Forward Network (FFN):** Processes the features.
3. **Repeat** for $N$ layers.

---

## 2. How Does It Handle Inputs without an Encoder?

If there is no encoder, you might wonder: *“Where does the prompt/input go?”*

In a decoder-only model, **the input prompt and the generated output are treated as a single continuous sequence.** They go into the exact same embedding layer and the exact same attention blocks.

Here is the breakdown of how it processes information:

### The Masked Attention Trick

Because it uses **Masked Self-Attention**, a token can only look at tokens that came before it.

* **During the Prompt Phase:** If your prompt is `"The sky is"`, the word `"is"` can use self-attention to look back at `"The"` and `"sky"`. This acts exactly like an encoder would—it compresses the context of the prompt.
* **During the Generation Phase:** When the model generates the next word, `"blue"`, that new token is appended to the sequence. To generate the *next* word, the model performs self-attention across the entire sequence: `"The sky is blue"`.

> **In summary:** Self-attention does double duty. It acts like an encoder when processing your prompt, and acts like a decoder when generating new text.

---

## Quick Comparison

| Feature | Encoder-Decoder (e.g., T5, BART) | Decoder-Only (e.g., GPT, Llama) |
| --- | --- | --- |
| **Input & Output** | Kept in separate distinct streams. | Combined into one continuous stream. |
| **Self-Attention** | Unmasked in Encoder, Masked in Decoder. | **Masked** for the entire sequence. |
| **Cross-Attention** | **Yes** (Bridging Encoder & Decoder). | **No** (Completely removed). |
| **Best Used For** | Translation, Summarization. | Text Generation, Code Generation, Chat. |