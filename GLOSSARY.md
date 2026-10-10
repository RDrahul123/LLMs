# LLM Systems Glossary

Short definitions for terms used throughout the modules. Follow the links for detailed derivations and examples.

| Term | Meaning |
|---|---|
| **Agent** | A software loop where a model proposes steps and a controller manages state, tool execution, and stopping conditions. See [Module 8](08-ai-agents/README.md). |
| **Attention** | A learned weighted combination of information from token positions, computed from query/key compatibility and value vectors. See [Module 2](02-transformer-architecture/deep-dive.md). |
| **Autoregressive generation** | Repeatedly predicting and appending the next token conditioned on preceding tokens. See [Module 1](01-introduction/deep-dive.md). |
| **BM25** | A lexical document-ranking method based on query-term matches, term rarity, and document length. See [ContextIQ](projects/03-document-qa/README.md). |
| **Causal mask** | An attention constraint that prevents a decoder position from seeing future positions. |
| **Chunk** | A segment of source material indexed/retrieved as one unit in a RAG system. See [Module 5](04-rag/deep-dive.md). |
| **Context window** | The token capacity available to a particular model request, including input and usually generated output. |
| **Cross-encoder** | A ranking model that reads a query and candidate passage jointly; usually slower than vector retrieval but useful for reranking a smaller candidate set. |
| **DPO** | Direct Preference Optimization, a preference-training method that optimizes a model directly from chosen/rejected examples without a separately trained reward model. |
| **Embedding** | A numerical vector representation produced for text or another modality; distance/similarity is a ranking signal, not proof of relevance. |
| **Fine-tuning** | Updating model parameters using task-specific examples. See [Module 6](05-fine-tuning/deep-dive.md). |
| **GQA** | Grouped-query attention; multiple query heads share fewer key/value heads to reduce inference memory and bandwidth. |
| **Hallucination** | A plausible but unsupported or incorrect model-generated claim; detection requires comparison with evidence or trusted checks. |
| **Hybrid retrieval** | Combining lexical and semantic retrieval signals, often through rank fusion. See [Module 9](09-advanced-rag/deep-dive.md). |
| **Inference** | Running a trained model on a request to produce outputs; it does not ordinarily update the model weights. |
| **KV cache** | Stored key/value projections from prior tokens, reused during autoregressive decoding to avoid recomputing them. |
| **LoRA** | Low-Rank Adaptation, a parameter-efficient fine-tuning method that learns a low-rank weight update while base weights stay frozen. |
| **MCP** | Model Context Protocol, a standard protocol for connecting AI applications to external tools and resources; always apply authorization at the host/application boundary. |
| **Next-token prediction** | The language-model objective of estimating the next token conditioned on preceding tokens. |
| **Open weights** | Model parameters available for download/use under a license; the term alone does not imply unrestricted use or open-source licensing. |
| **Prompt injection** | Malicious or conflicting instructions in user or external content that attempt to influence a model; mitigate with layered controls, not prompt wording alone. |
| **RAG** | Retrieval-Augmented Generation: retrieve source material at request time and supply it to a model to ground a response. See [Module 5](04-rag/deep-dive.md). |
| **Reranking** | Re-scoring an initial candidate list with a more expressive ranking model to improve ordering. |
| **RLHF** | Reinforcement Learning from Human Feedback; preference data is used to train a reward signal and optimize model behavior. |
| **SFT** | Supervised Fine-Tuning; train on prompt/desired-response demonstrations. |
| **Temperature** | A decoding parameter that rescales logits and changes sampling diversity; it does not certify answer quality. |
| **Token** | A tokenizer unit—often a word piece, punctuation, whitespace segment, or byte-level fragment—represented by an integer ID. |
| **Tool calling** | A model emits a structured proposal for an application-defined function; the application validates and executes it. See [Module 8](08-ai-agents/deep-dive.md). |
| **Vector database** | A system for storing vector representations and retrieving near neighbors, often with metadata filters. |

## A useful distinction

```text
Weights  = patterns learned during training
Prompt  = temporary instructions/context for a request
RAG     = external evidence retrieved at request time
Memory  = application-managed state persisted between requests
Tool    = an application capability executed under code-enforced permissions
```

These mechanisms solve different problems; adding one does not automatically provide the guarantees of another.
