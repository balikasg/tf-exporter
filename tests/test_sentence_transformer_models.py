import pytest
from pathlib import Path
import shutil
import tensorflow as tf
from tf_exporter import ModelConverter


@pytest.mark.parametrize(
    "model_name",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v1",
        "sentence-transformers/nq-distilbert-base-v1",
    ],
)
def test_model_conversion(model_name):
    """Tests the nq-distilbert-base-v1 model for conversion and
    equivalence of the sentence-transformer tokenizer and model predictions
    with the tf graph predictions"""
    out_dir = Path("./outdir") / str(hash(model_name))

    converter = ModelConverter(model_name_or_path=model_name, output_dir=out_dir)
    converter.validate_converted_model(input_test="This is a test")
    shutil.rmtree(out_dir, ignore_errors=True)
    tf.keras.backend.clear_session()
