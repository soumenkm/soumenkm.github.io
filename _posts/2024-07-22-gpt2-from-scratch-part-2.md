---
title: 'GPT2 from Scratch Part 2'
date: 2024-7-25
permalink: /posts/2024/7/gpt2-from-scratch-part-2/
tags:
  - ML
  - AI
  - LLM
  - GPT
  - NLP
---

## Part 2 - GPT2 from Scratch: Attention Mechanism and Transformer

>**Table of Contents:**
>
> 1. Embedding Layer
    1. Word Embedding
    2. Positional Embedding
> 2. Attention Layer
    1. Motivation for Attention
    2. Scaled Dot Product Attention
    3. Causal Self Attention
    4. Multi-head Self Attention
    5. PyTorch's Attention Layer
> 3. Transformer Layer
    1. Layer Normalization
    2. GELU Activation
    3. MLP Layer
    4. Residual Connection
> 4. GPT Model
    1. Model Architecture
    2. Generate Text
    3. Greedy Decoding
    4. Top-K Decoding
    5. Effect of Temperature

## Prerequisites

- Firstly, a medium level knowledge of `Python` (including OOPs concepts) is required, as the implementation will be carried out in this language.

- A foundation level machine learning concepts is needed to understand the concepts. This includes neural networks, loss functions, training dynamics etc. The knowledge of Recurrent Neural Network (RNN) is a bonus but not strictly needed.

- Experience with deep learning frameworks `PyTorch` is necessary, as it will be used to build and train the model. Basic knowledge of `Huggingface` library is a plus.

- A basic mathematical background in linear algebra and probability will support the understanding of model operations.

## Previous Blog

In [_Part 1_](https://soumenkm.github.io/posts/2024/7/gpt2-from-scratch-part-1/) of this blog, I have covered the aspects of data preparation process for GPT model which includes basics of embedding, tokenization process and sliding window method of data sampling. 

## 1. Embedding Layer
### 1.1. Word Embedding
### 1.2. Positional Embedding

#### 1.2.1. Why is "Position" Important?

We have already seen the word embedding which converts the integer token IDs into a continuous finite dimensional real vector representation. In [_Section 3.1: Working Principle of GPT_](https://soumenkm.github.io/posts/2024/7/gpt2-from-scratch-part-1/), I have explained in detail that a GPT model processes an entire token sequence in one time step which is fundamentally different from an RNN model which processes a token sequence one at a time step which makes these models to compute the outputs "recurrently". In summary, RNN process one token at a time which means that the positional information of the tokens are implicitly considered in the computation due to the nature of "recurrence" computation in RNN models. Since it handles the tokens one at a time, it knows which token appears first and which token appears last in the sequence as the position of token and time step of processing that token are exactly identical.

```sh
Tokens: ["I", "love", "machine", "learning"]
Positions: [0, 1, 2, 3]
Time Step: [0, 1, 2, 3]
```

In contrary, the GPT models perform its computation in such a way that it doesn't need to know the tokens one by one in steps, rather it has the capability of computing all the tokens, all at one time step, therefore removing the "recurrence" component from the model. This improves the time complexity of the computation significantly as compared to RNN model. Although, in search of improving the computational cost, we "lost" one important aspects of RNN model - _implicit positional information_.

> Since, the GPT models process the entire sequence using a mechanism called _attention_ which doesn't have any notion of "position" or "order" of the tokens in the sequence (as it processes the tokens all at one time step using linear combination of embedding vectors), the model has no way to figure it out the positional information from the sequence. I will revisit this point again in next section where I will talk about the attention mechanism. 

For those who cannot wait to know the reason, I am going to provide a brief intuitive explanation for this. You can think the attention mechanism as a function which takes the embeddings of all the tokens as an input and returns the contextual embedding vector of all the tokens as an output. At its core, this function computes or "attends" all the input embedding vectors by taking the similarity (or dot product) with all the other vectors. It means the attention mechanism can be though of taking the linear combination of all the input embedding vectors at a token position where the combination coefficient will vary as we move from one token position to another token position during the computation of context vectors. Think the combination coefficients as a measure of the relevance or importance of each token in the sequence relative to the current token.

\[
    \mathbf{c}^k = \sum_{i = 1}^{T} \alpha _i^k \cdot \mathbf{x} _i \quad (\forall k \in {1, 2, \dots, T})
\]

Where $\mathbf{x}_i$ is the input embedding vector and $\mathbf{c}^k$ is the $k$-th output context vector. The combination coefficient (often called _attention weights_) are permutation invariant which means if we provide the sequence $(\mathbf{x} _1, \mathbf{x} _3, \mathbf{x} _2)$ instead of $(\mathbf{x} _1, \mathbf{x} _2, \mathbf{x} _3)$ then the attention weights will be readjusted accordingly by the attention function automatically (which is kind of a _black-box_ to us at this moment) to produce the same output context vector. For instance, whether a sentence starts with _"Variable x is pointer array"_ or _"Variable x is array pointer"_, the model recalibrates the attention weights to maintain contextually appropriate output vectors. That is why the order of tokens in a sequence does not matter to the attention function!

Consider a hypothetical vocabulary of only 5 tokens and each of the tokens are embedded by a 3 dimensional vector. Therefore, the token IDs can be either 0, 1, 2, 3, 4. Let's take two possible sequence of token IDs as follows:

```python
E = [[0.1, 0.2, 0.3, 0.4, 0.5],
     [0.6, 0.7, 0.8, 0.9, 1.0],
     [1.1, 1.2, 1.3, 1.4, 1.5]] # Embedding Matrix size is (d, V)

token_IDs_A = [1, 2] # (x1, x2)
word_embedding_A = [[0.2, 0.7, 1.2], 
                    [0.3, 0.8, 1.3]] # Extracts 1-st and 2-nd column from E
token_IDs_B = [2, 1] # (x2, x1)
word_embedding_B = [[0.3, 0.8. 1.3],
                    [0.2, 0.7, 1.2]] # Extracts 2-nd and 1-st column from E
```

> Note that when we are changing the order of tokens in a sequence, the embedding vectors $\mathbf{x}$ values are not changed! It is only the position of the vectors that are changed. From the previous section, it is obvious that the embedding vectors are nothing but the columns of the embedding matrix. The embedding vector for a token ID is found after extracting its corresponding column from the embedding matrix. Therefore, the order in which the columns are extracted is changed but the individual column elements ($d$ elements in each column) are not changed! Therefore, the attention computation is not affected as the attention weights will just be readjusted. For example, $\mathbf{c} = \alpha _1 \cdot \mathbf{x} _1 + \alpha _2 \cdot \mathbf{x} _2 + \alpha _3 \cdot \mathbf{x} _3$ when the sequence is $(\mathbf{x} _1, \mathbf{x} _2, \mathbf{x} _3)$ which is same as $\mathbf{c} = \alpha _1' \cdot \mathbf{x} _1 + \alpha _2' \cdot \mathbf{x} _3 + \alpha _3' \cdot \mathbf{x} _2$ when the sequence is $(\mathbf{x} _1, \mathbf{x} _3, \mathbf{x} _2)$. Note these two linear combinations are exactly same. It is just the readjustment of the attention weights ($\alpha _1' \leftarrow \alpha _1, \alpha_2' \leftarrow \alpha _3$ and $\alpha _3' \leftarrow \alpha _2$) that makes it permutation invariant. You can also think it as a consequence of the fact that the vector addition in a vector space $\mathbb{V}$ which is denoted as $+ _\mathbb{V}$ or simply $+$ (when the context of $\mathbb{V}$ is clear then the subscript can be removed) is a commutative operator.

If you have not understood this above section on attention, it's absolutely fine as it will be more clear when I will unwrap this attention "black-box" in the next section.

Let's first understand why the position or order of tokens in a sequence matter in natural language. Consider the following two sentences:

```sh
Sequence 1: "Although, I did not understand the Attention mechanism, I was engaged with the blog!"
Sequence 2: "Although, I did understand the Attention mechanism, I was not engaged with the blog!"
```

If you tokenize these two sequences, it will give you the exact same tokens, but in different order. If we do not consider the order of the tokens in a sequence, then the above two sequences will possess same meaning. Although, in reality, it is not the case! These two sentences are clearly contradictory of each other in literal meaning. Therefore, it's established that the position of the token in a sequence is the most important property of the sequence which needs to be captured by the GPT model. Since there is no direct way to capture it implicitly, we need to "inject" some positional information to the model explicitly so that it can understand the text better as _"position"_ is what makes a sequence to behave like _"text sentence"_.

The _positional embedding_ is also a vector in the embedding space which has the same dimension as the word embedding space. Therefore, at the token position $i$, to get the final embedding input vector ($\mathbf{t} _i$), we will add the word embedding vector ($\mathbf{x} _i$) and the positional embedding vector ($\mathbf{e} _i$) element-wise. 

\[
    \mathbf{t} _i = \mathbf{x} _i + \mathbf{e} _i \quad \in \mathbb{R}^d
\]

#### 1.2.2. Properties of Positional Embedding

Before I talk about the properties of a "good" positional embedding, let's first take a couple of examples of what constitutes a "bad" positional embedding.

Since we are interested to encode the positions of each tokens, as a naive approach, one could propose of using just the sequence numbers as their representation of positions. Mathematically, we can have a position embedding of a token at position $i$ such that:

\[
    \mathbf{e} _i = (i) _{j=0}^{j=d-1} \quad \text{in other words,}\quad e _i^j = i \quad \forall j = {0, 1, \dots, d-1}
\]

For example if we have an embedding dimension of $d=5$ and a sequence of $T=1024$ tokens then:

```python
x0 = [0.1, 0.05, 0.14, 0.42, 0.25] # Word embedding at token position 0
e0 = [0, 0, 0, 0, 0] # Position embedding at token position 0

x1 = [0.16, 0.72, 0.28, 0.39, 1.5] # Word embedding at token position 1
e1 = [1, 1, 1, 1, 1] # Position embedding at token position 1

...

x1023 = [0.19, 0.72, 0.93, 0.44, 0.51] # Word embedding at token position 1023
e1023 = [1023, 1023, 1023, 1023, 1023] # Position embedding at token position 1023
```


## 2. Attention Layer
### 2.1. Motivation for Attention
### 2.2. Scaled Dot Product Attention
### 2.3. Causal Self Attention
### 2.4. Multi-head Self Attention
### 2.5. PyTorch's Attention Layer
## 3. Transformer Layer
### 3.1. Layer Normalization
### 3.2. GELU and SoLU Activation
### 3.3. Feed Forward Layer
### 3.4. Residual Connection
## 4. GPT Model
### 4.1. Model Architecture
### 4.2. Generate Text
### 4.3. Greedy Decoding
### 4.4. Top-K Decoding
### 4.5. Effect of Temperature

## What's Next?

In next blog, I will cover ...

Happy learning! 😃

### Reference Books

- Sebastian Raschka, [_Build a Large Language Model (From Scratch)_](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- Aston Zhang, [_Dive into Deep Learning_](https://d2l.ai)
- Daniel Jurafsky, [_Speech and Language Processing_](https://web.stanford.edu/~jurafsky/slp3/)
- [_Huggingface tokenizers_](https://huggingface.co/docs/transformers/en/main_classes/tokenizer)

### Written by

> Soumen Mondal (Email: [23m2157@iitb.ac.in](mailto:23m2157@iitb.ac.in)), MS in AI and DS, CMInDS, IIT Bombay
