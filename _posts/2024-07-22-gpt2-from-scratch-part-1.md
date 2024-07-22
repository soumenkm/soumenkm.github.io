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

\[
f: \mathbb{R}^m \rightarrow \mathbb{R}^n \quad \text{where} \quad m \gt n
\]

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

In NLP, models like GPT, BERT etc. use positional embeddings to retain the order of words in a sentence which is crucial for understanding context and meaning accurately. This itself is a huge topic and it will be covered later in this blog.

## 2. Tokenization

### 2.1 What is Tokenization?

_Tokenization_ is a fundamental process in NLP where a large body of text is divided into smaller, manageable units known as _tokens_. These tokens can be individual words, phrases, or even characters depending on the granularity required for the specific application.

The primary purpose of tokenization is to simplify the complex structure of natural languages into structured data that LLMs can understand and process. Each token represents a discrete piece of information, and when analyzed together, these tokens can reveal patterns and insights about the text.

> In LLM, tokenization is the first step in preprocessing for many NLP tasks such as sentiment analysis, machine translation, and text generation. It allows the LLMs to efficiently perform operations on text by converting the unstructured data into a numerical format that machine learning algorithms can work with.

### 2.2 Why is Tokenization Required?

- **Structure and Manageability:** Natural language is inherently unstructured and variable in length. Tokenization breaks text into consistent, manageable pieces, making it easier for models to process. Tokenization allows models to handle vocabulary efficiently by representing and processing text as sequences of tokens rather than entire sentences or documents at once.

- **Model Training and Performance:** LLMs are trained on numeric data. Tokens are converted into numerical IDs called _token IDs_, which can be fed into embedding layer for generating the embedding vector. By breaking text into tokens, models can learn the context and meaning associated with different sequences of tokens, which is crucial for generating coherent and contextually appropriate responses.

- **Efficiency:** Advanced tokenization techniques like subword tokenization (e.g., Byte-Pair Encoding) help in managing vocabulary size by breaking down rare or unknown word into known subword, allowing the model to handle a wider variety of words without significantly increasing the vocabulary size. Efficient tokenization reduces the memory footprint of the LLMs by needing fewer unique tokens for the dataset.

### 2.3 Implement a Simple Tokenizer

#### 2.3.1 A Quick Refresher on `re` Library

In practical application, we are never going to build our own tokenizer but it will give an idea of how the tokenization is performed in pre-built tokenizers. To build our own tokenizer, we can use Python's regular expression `re` library.

```python
import re

pattern1 = r"\s" # treats (\) as a literal (raw) char
pattern2 = r"(\s)"
text = "Hello, World. This, is a test." # treat it as training data
result1 = re.split(pattern=pattern1, string=text)
result2 = re.split(pattern=pattern2, string=text)
print(result1)
print(result2)
```

```python
['Hello,', 'World.', 'This,', 'is', 'a', 'test.']
['Hello,', ' ', 'World.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']
```

Note the difference between two patterns. The first pattern is not capturing the group (if the match is found based on pattern `\s` which means any form of whitespace) as it is not retaining the group in the final result. Since we are using capturing group (`r"(...)"`) in the second pattern, It will capture the groups where the pattern is present while retaining the groups into the list.

The list of words in the result is called _tokens_. Note that there is a problem in this tokenization scheme as the list has individual words along with the punctuation. If we want to treat a word followed or preceded by punctuation, then the number of unique tokens will be large so the size of _vocabulary_ (list of tokens) will increase exponentially. Therefore, we want to treat the punctuations as separate tokens. 

> In _cased_ tokenization scheme, the case sensitivity of tokens matters for example `apple` is different from `Apple` so they will be treated as two different unique tokens. However, in _uncased_ tokenization scheme, those two tokens will be treated as one unique token `apple`.

Now consider a complex tokenization pattern where we want to split the text based on all the punctuation characters, whitespace and digits but at the same time we want to retain the capturing groups as well otherwise, the groups will be lost from the list.

```python
pattern1 = r"(--|[,.?_:;\"'()!]|\s)"
pattern2 = r"(--|[^0-9a-zA-Z])"
pattern3 = r"(--|[^0-9a-zA-Z]|\d+)"
text = "abc1-def2 -- ghi123. jkl 2 30 mno."
result1 = re.split(pattern1, text)
result2 = re.split(pattern2, text)
result3 = re.split(pattern3, text)
print(result1)
print(result2)
print(result3)
```

```python
['abc1-def2', ' ', '', '--', '', ' ', 'ghi123', '.', '', ' ', 'jkl', ' ', '2', ' ', '30', ' ', 'mno', '.', '']
['abc1', '-', 'def2', ' ', '', '--', '', ' ', 'ghi123', '.', '', ' ', 'jkl', ' ', '2', ' ', '30', ' ', 'mno', '.', '']
['abc', '1', '', '-', 'def', '2', '', ' ', '', '--', '', ' ', 'ghi', '123', '', '.', '', ' ', 'jkl', ' ', '', '2', '', ' ', '', '30', '', ' ', 'mno', '.', '']
```

`pattern1` is designed to split the text at punctuation characters (commas, periods, colons, semicolons, question marks, exclamation etc.) or whitespace characters (`\s` includes spaces, tabs, newlines, etc.). `pattern1` splits the text on three types of occurrences which are `--` (a double hyphen) or any characters from the character group or any form of whitespace. Note the special (`|`) symbol which means logical OR operation. The bracket (`[...]`) is defining a character class meaning that any characters withing the bracket will be matched for creating the groups. When it starts scanning the characters, the moment it finds `pattern1` then it will pause to split the expression, captures the group and resumes scanning thereafter. Empty strings (`''`) appear where there are no characters between two consecutive delimiters. For example, between two consecutive delimiter `ghi123. jkl` (delimiters are `.` and ` `), there is no character so it will capture the group as empty string (`''`)

`pattern2` splits the text on two types of occurrences. `--` (a double hyphen) or `[^0-9a-zA-Z]` which matches any character that is not a number `(0-9)`, not a lowercase letter `(a-z)`, and not an uppercase letter `(A-Z)`. Note the special (`^`) symbol which means logical NOT operation.

`pattern3` extends `pattern2` by also splitting at sequences of digits (`\d+`, `+` means one or more occurrence of a digit). The inclusion of `\d+` in the pattern results in numbers being split from alphabetic characters, as well as being captured as separate tokens. 

> I will use `pattern1` for the tokenization for simplicity. Moreover, I will not consider `\s` as part of vocabulary for simplicity, but they are included for real world application. To know more about `re` library, you can refer to [Datacamp](https://www.datacamp.com/tutorial/python-regular-expression-tutorial) or [Programiz](https://www.programiz.com/python-programming/regex).

#### 2.3.2. Convert `tokens` into `token IDs`

Once the tokens are created, we need to convert the tokens into an integer called _token IDs_. Given a corpus, the first task is to tokenize the entire corpus which gives us the list of tokens which is also called a _vocabulary_. Then the process in which we map each of the tokens of the vocabulary into an integer is called building a vocabulary. Since the vocabulary is represented as alphabetically sorted list so the list index can be considered as token IDs. Moreover, we can also represent a vocabulary using a dictionary where the dictionary keys will be represented by tokens and the values will be represented by the corresponding token IDs. It is also possible to invert the vocabulary dictionary since it is one to one mapping between tokens and token IDs. In the _inverted vocabulary_, the key will represent a token ID and the value will represent its corresponding token.

<p align="center">
<img src="/images/tokenizer.jpeg" alt="Tokenizer" width="400"/>
</p>

The `Tokenizer` class should take the corpus file path as an input where we have stored the training corpus. The job of the tokenizer should be to build the vocabulary and inverse vocabulary. Therefore, the `__init__` method should build the entire vocabulary and inverse vocabulary. It should also store the size of vocabulary (`vocab_size`), vocabulary dictionary (`vocab`) and inverse vocabulary dictionary (`inv_vocab`) as property of class. The `Tokenizer` class should also provide 3 basic methods which are `tokenize`, `encode`, and `decode`. In any prebuilt tokenizer e.g. provided by Huggingface or OpenAI, these 3 methods will be present along with other methods. The `tokenize` method takes a raw text as an input and outputs the list of tokens. The `encode` method also takes the raw text as an input and outputs the list of integer token IDs from the vocabulary. The `decode` method takes the list of token IDs as an input and outputs the reconstructed interpretable text from inverse vocabulary.

```python
import re
from pathlib import Path
from typing import List

class Tokenizer:
    def __init__(self, corpus_file_path: Path) -> None:
        self.vocab_size = None
        self.vocab = None
        self.inv_vocab = None
        pass
    
    def tokenize(self, text: str) -> List[str]:
        pass
    
    def encode(self, text: str) -> List[str]:
        pass
    
    def decode(self, token_ids: List[int]) -> str:
        pass
```
So far, I have just defined the prototype of the class `Tokenizer`. Before, I start implementing the interface of the class, I have to talk about the importance of special tokens.

#### 2.3.3. Special Tokens



### Reference Books

- Sebastian Raschka, _Build a Large Language Model (From Scratch)_
- Aston Zhang, _Dive into Deep Learning_
- Daniel Jurafsky, _Speech and Language Processing_

### Written by

> Soumen Mondal (Email: [23m2157@iitb.ac.in](mailto:23m2157@iitb.ac.in))
