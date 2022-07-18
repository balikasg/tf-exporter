# tf-packager: Transformer model + tokenizer = a single graph

Create a single artifact to serve Transformers predictions without requiring a dinstict steps for tokenization and model predictions. 


```bash
pip install tf-packager
tf-packager --model-name-or-path sentence-transformers/nq-distilbert-base-v1 --out-dir /tmp/tf-nq-with-tokenizer
```
