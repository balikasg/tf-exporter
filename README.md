<a href="https://github.com/balikasg/tf-exporter/actions/workflows/testing.yml/badge.svg">
    <img src="https://github.com/balikasg/tf-exporter/actions/workflows/testing.yml/badge.svg" alt="build status"></a>

# tf-packager: Convert a sentence transformer model + tokenizer to a single graph

`tf-packager` let's you 
create a single artifact to serve Transformers prmredictions without requiring a dinstict steps for tokenization and model predictions. 

## Installation
```bash
python -m venv venv  # Recommended: Create a virtual environment
source venv/bin/activate # Activate it

# Install the code from git
python -m pip install git+https://github.com/balikasg/tf-exporter.git
```

## Usage
From python now you can:
```python
from tf_packager import ModelConverter
converter = ModelConverter(model_name_or_path='sentence-transformers/nq-distilbert-base-v1',
                           output_dir='/tmp/tf-model')
converter.convert_pytorch_to_tensorflow(input_test='This is a test')
# Persists tf model files at `/tmp/tf-model`
```
You can also use a shell script:
```bash
# Convert your model and save under models/ (default):
python tf_packager/convert_to_single_graph.py --model-name sentence-transformers/nq-distilbert-base-v1

ls models/tf_model  # Returns the persisted files
# assets            keras_metadata.pb saved_model.pb    variables
```
