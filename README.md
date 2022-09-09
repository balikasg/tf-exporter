# tf-packager: Transformer model + tokenizer = a single graph

`tf_packager` let's you 
create a single artifact to serve Transformers predictions without requiring a dinstict steps for tokenization and model predictions. 


```bash
git clone git@github.com:balikasg/tf-exporter.git  # Get the code
python -m venv venv  # Create a virtual environment
source venv/bin/activate # Activate it
python setup.py install  # install the code! 

# Convert your model and save under models/ (default):
python src/tf_packager/convert_to_single_graph.py --model-name sentence-transformers/nq-distilbert-base-v1

ls models/tf_model  # Returns the persisted files
# assets            keras_metadata.pb saved_model.pb    variables
```
