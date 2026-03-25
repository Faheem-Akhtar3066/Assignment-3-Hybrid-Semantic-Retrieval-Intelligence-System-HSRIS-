📌 Overview

This repository contains the implementation of HSRIS, a multi-stage Natural Language Processing (NLP) pipeline designed for processing and retrieving customer support tickets.

To demonstrate a deep understanding of underlying mathematical operations, the entire system was built from scratch using base PyTorch and NumPy. We strictly avoided high-level wrappers like Scikit-Learn's TfidfVectorizer or LabelEncoder.

The system leverages a hybrid approach, combining the exact keyword matching of statistical methods with the contextual understanding of neural semantic embeddings.

🚀 Key Features & Architecture
1. The Categorical Foundation (Custom Encoders)

    Label Encoding: Maps Ticket Priority (Low, Medium, High, Critical) to ordinal integers from first principles, including unseen category handling.

One-Hot Encoding: Converts Ticket Channels into binary vector representations using pure NumPy.

2. Engine A: Sparse Representation (TF-IDF)

    Custom Tokenizer & N-Grams: Uses regex for lowercasing and punctuation removal, combined with a sliding window to generate bigrams and trigrams for context capture.

From-Scratch Vectorizer: Builds a vocabulary of the top 5,000 tokens and manually computes IDF scores and TF-IDF transformations.

Memory Optimization: TF-IDF matrices are stored using torch.sparse to prevent RAM crashes during computation.

3. Engine B: Dense Semantic Layer (GloVe)

    Pre-Trained Embeddings: Loads 300-dimensional GloVe weights into a torch.nn.Embedding layer.

OOV Strategy: Implements a catch-all <UNK> zero vector for out-of-vocabulary words.

Preventing Semantic Dilution: Variable-length descriptions are converted into fixed 300-dimensional vectors using TF-IDF weighted averaging rather than a simple mean, ensuring rare technical keywords are not drowned out by common words.

4. Hybrid Search Logic & Hardware Optimization

    The Formula: Retrieves the top 5 most relevant tickets using a weighted hybrid score:
    FinalScore=α(TF−IDFScore)+(1−α)(GloVeScore) (Default α=0.4) 

Dual-GPU Parallelization: Cosine similarity calculations are distributed across a Kaggle Dual T4 x2 GPU environment to process large query batches rapidly.

📊 Evaluation

    Quantitative: Evaluated using Precision@5 for Ticket Type matching across random queries.

Qualitative: The system successfully demonstrates edge cases where semantic search (GloVe) outperforms keyword search (TF-IDF) by understanding synonyms and context.

💻 Interactive UI (Gradio)

The project includes a Gradio application featuring:

    A text area for entering new ticket descriptions.

An interactive slider to adjust the α value between 0.0 and 1.0, allowing users to visually observe the shift between keyword and semantic matching.

A display predicting the Ticket Type and rendering the top 3 similar past resolutions.

🤝 Team & Division of Labor

This project was a collaborative effort. Maximum team size: 2 members.

    Zaid Hassan (Data & Sparse Engine Lead): Responsible for the Kaggle environment setup, data cleaning, custom categorical encoders, custom regex tokenizer, N-gram generation, and building the complete TF-IDF mathematical pipeline from scratch. Authored the project's Medium article detailing the implementation logic.

    [Insert Partner Name] (Neural Semantic & Deployment Lead): Responsible for integrating the GloVe 300d embeddings, implementing TF-IDF weighted averaging, dual T4 GPU parallelization, hybrid search evaluation, and building/deploying the interactive Gradio UI. Managed the LinkedIn system showcase.

🔗 Project Links

Medium Article: https://medium.com/p/33fe12b2566d?postPublishedType=initial

LinkedIn Showcase: https://www.linkedin.com/posts/zaid-hassan-2a1990332_datascience-nlp-pytorch-ugcPost-7442439221136740354-tXkh?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFPLZpkB3_iNYm_sUgvI1OBYwQLIxD3hrkg
<img width="827" height="181" alt="Screenshot_20260325_094802" src="https://github.com/user-attachments/assets/6c268c8c-93d4-49f5-b0b1-09227041cfbf" />
<img width="827" height="521" alt="Screenshot_20260325_094200" src="https://github.com/user-attachments/assets/42811c29-752c-4aff-a606-ed0e17dcb046" />
<img width="1366" height="898" alt="Screenshot 2026-03-25 at 09-41-02 HSRIS — Hybrid Semantic Retrieval   Intelligence System" src="https://github.com/user-attachments/assets/ebecafa5-9d06-4adc-991a-06275ce8bfb1" />
<img width="2085" height="742" alt="gpu_performance_plot" src="https://github.com/user-attachments/assets/77006e7c-2bde-464b-9a60-a758d3498d1d" />
