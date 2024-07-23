import re

text = """Generative pre-trained transformers (GPT) are a type of large language model (LLM) and a prominent framework for generative artificial intelligence.
They are artificial neural networks that are used in natural language processing tasks.
GPTs are based on the transformer architecture, pre-trained on large data sets of unlabelled text, and able to generate novel human-like content.
As of 2023, most LLMs have these characteristics and are sometimes referred to broadly as GPTs."""

# Regular expression to match words and punctuations
tokens = re.split(r"\s", text)

print(tokens)