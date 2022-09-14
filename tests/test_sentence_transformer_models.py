import shutil
from tf_packager import ModelConverter


def test_model_conversion():
    """Tests the nq-distilbert-base-v1 model for conversion and
    equivalence of the sentence-transformer tokenizer and model predictions
    with the tf graph predictions"""
    out_dir = "./outdir"
    converter = ModelConverter(model_name_or_path='sentence-transformers/nq-distilbert-base-v1',
                               output_dir='/tmp/tf-model')
    converter.convert_pytorch_to_tensorflow(input_test='This is a test')
    shutil.rmtree(out_dir, ignore_errors=True)
