---
title: 'GPT2 from Scratch Part 1'
date: 2024-7-22
permalink: /posts/2024/7/gpt2-from-scratch-part-1/
tags:
  - ML
  - AI
  - LLM
  - GPT
  - NLP
  - IIT Bombay
---

## Part 1 - GPT2 from Scratch: Preparation of Data Pipeline

>**Table of Contents:**
>
> 1. Embeddings
    1. What is Embedding?
    2. Why is Embedding Required?
    3. Mathematical Definition of Embedding
    4. Different Types of Embeddings
> 2. Tokenization
    1. What is Tokenization?
    2. Why is Tokenization Required?
    3. Implement a Simple Tokenizer
    4. Byte Pair Encoding (BPE)
    5. Huggingface 🤗 Tokenizers
> 3. Data Pipeline for GPT
    1. Data Sampling by Sliding Window
    2. Generate Word Embeddings
    3. Generate Positional Embeddings

## 1. Embeddings

### 1.1. What is Embedding?

In the context of machine learning, an _embedding_ is a representation of data (it could be any form of data such as audio, video, image, text etc.) in a finite-dimensional vector space. Embeddings are used to transform complex and often high-dimensional data into a more manageable, dense, and continuous vector space where similar items are closer together and dissimilar items are farther apart.

### 1.2. Why is Embedding Required?

The primary motivation for using embeddings is to translate high-dimensional, sparse, and non-numeric data into a lower-dimensional, dense, and numeric format that machine learning algorithms can efficiently process. This transformation facilitates better model performance.

- **Dimensionality Reduction:** Many machine learning models perform better when the input data is not high-dimensional due to the curse of dimensionality. Embeddings help in reducing the dimensions while retaining its essential features.

- **Feature Learning:** Embeddings learn to capture the underlying patterns in the data, acting as learned features that are useful for various tasks like classification, clustering, or summarization.

- **Handling Categorical Variables:** In many real-world applications, data comes in categorical form (like words, user IDs, or item IDs) that cannot be used directly by most machine learning algorithms, which expect numerical input. Embeddings convert these categorical variables into vectors in a continuous vector space.

### 1.3. Mathematical Definition of Embedding

Mathematically, an embedding can be defined as a mapping: $f: X \rightarrow \mathbb{R}^d$ where:
> $X$ is the original input space (possibly categorical data). $\mathbb{R}^d$ is the embedding vector space with dimension $d$. The mapping $f$ is typically a function parameterized by weights learned from the training data.

[ f: \mathbb{R}^m \rightarrow \mathbb{R}^n \quad \text{where} \quad m >> n ]

The above equation shows one special case of using the embedding map where we are converting the embedding from m-dimensional vector space to n-dimensional vector space where m is larger than n.

### 1.4. Different Types of Embeddings

Any form of data (it could be audio, image, video or text) can be converted to a point in a continuous real valued vector space known as embedding. Since the blog is more focused on NLP related applications, the embeddings related to text will be covered.

#### 1.4.1 Word Embedding

Word embeddings convert text into a set of feature vectors where each word (or more technically _token_) in the corpus is represented by a point in a vector space. The vectors are trained to ensure that words with similar meanings are located close to each other.

Word embeddings are extensively used in transformer based large language models (LLMs). There are several algorithms to generate word embedding but one of the most popular algorithm is _Word2Vec_. In principle, Word2Vec is trained using one of two architectures: _CBOW (Continuous Bag of Words)_ or _Skip-gram_. In the CBOW architecture, the model predicts the current word based on the context, and in the Skip-gram architecture, it predicts surrounding context words given the current word, both using a simple neural network which is trained to optimize similarity between the word's embedding vectors.

> In the embedding space, words that are closely related to a particular concept would form a cluster in the embedding space. Essentially, these embeddings create a high-dimensional map where distances between points (word embedding vectors) reflect how related the words are the source language.

For language modeling task, we could use the pre-trained word embeddings such as one obtained from Word2Vec to generate the word embeddings but it won't be efficient due to mismatch between Word2vec's training corpus and the GPT like LLM's training corpus. To capture complex patterns and context in the training corpus, LLMs would generate the word embeddings on the fly, directly from the training data at the time of training the model. This embedding vector will be highly optimized for the training corpus that have been used to train the LLM.

![Word Embedding](/images/word_embedding.jpeg "Word Embedding")

#### 1.4.2 Sentence/Document Embeddings

Sentence or document embeddings extend the concept of word embeddings to larger units of text, like sentences or entire documents. These embeddings capture the semantic meaning of the text in a single vector. BERT generates embeddings for sentences which can be used in natural language inference task which maps a sentence pair `<A, B>` to one of the label between `entailment`, `contradiction` and `neutral`. BERT need to know the sentence level information which is captured by sentence embedding to infer whether sentence A is an entailment (or contradiction or no relation) of sentence B.

#### 1.4.3 Positional Embeddings

Positional embeddings are used in models that need to understand sequence order (like sentences in text or time series data), adding information about the relative or absolute position of elements in the sequence.

In NLP, models like GPT, BERT etc. use positional embeddings to retain the order of words in a sentence which is crucial for understanding context and meaning accurately. This itself is a huge topic and it will be covered in the blog where I will cover GPT model.

### Reference Books

- Sebastian Raschka, _Build a Large Language Model (From Scratch)_
- Aston Zhang, _Dive into Deep Learning_
- Daniel Jurafsky, _Speech and Language Processing_

### Written by

> Soumen Mondal (Email: [23m2157@iitb.ac.in](mailto:23m2157@iitb.ac.in))

### Tables

| Plugin | README |
| ------ | ------ |
| Dropbox | [PlDb](https://plugins/dropbox/README.md) |
| GitHub | abc |

### Code

```sh
docker run -d -p 8000:8080 --restart=always --cap-add=SYS_ADMIN --name=dillinger <youruser>/dillinger:${package.json.version}
```

> Note: `--capt-add=SYS-ADMIN` is required for PDF rendering.

```python
import torch
a = torch.tensor([1,2,3])
print(a)
if a.dim() > 2:
    raise ValueError("Invalid!")
for i in a:
  print(f"Hello {a}")

```
