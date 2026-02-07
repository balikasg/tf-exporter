"""Convert a sentence transformer model to a TensorFlow graph with embedded tokenizer
Heavily inspired by: https://www.tensorflow.org/text/guide/bert_preprocessing_guide
"""
import argparse
import logging
from contextlib import contextmanager
from pathlib import Path
import tempfile
import shutil

from huggingface_hub import snapshot_download
from numpy.testing import assert_almost_equal
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import import_from_string

from tf_exporter.utils import load_json

LOG = logging.getLogger("tf-packager")
LOG.setLevel(logging.INFO)

MAX_SEQ_LENGTH_DEFAULT = 256
# Transformers model artifacts depending on the backend are named:
PYTORCH_MODEL = "pytorch_model.bin"
TF_MODEL = "tf_model.h5"
KERAS_INPUT_NAME = "input_sequence"
KERAS_OUTPUT_NAME = "sentence_transformer_packager"

MODEL_MAPPING = {
    "sentence_transformers.models.Transformer": "tf_exporter.models.Transformer",
    "sentence_transformers.models.Normalize": "tf_exporter.models.Normalize",
    "sentence_transformers.models.Pooling": "tf_exporter.models.Pooling",
    "sentence_transformers.models.Dense": "tf_exporter.models.TfDense",
}


@contextmanager
def get_model_path(model_name_or_path):
    """Get model path, always downloading to temporary directory.

    Parameters
    ----------
    model_name_or_path : str or Path
        Model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2') or local path

    Yields
    ------
    Path
        Path to the model directory
    """
    temp_dir = None

    try:
        if Path(model_name_or_path).exists():
            # Local path provided
            yield Path(model_name_or_path)
        else:
            # Download to temporary directory
            temp_dir = tempfile.mkdtemp(prefix="hf_model_")
            model_path = snapshot_download(
                repo_id=model_name_or_path,
                local_dir=temp_dir,
                local_dir_use_symlinks=False
            )
            yield Path(model_path)
    finally:
        # Cleanup temporary directory if created
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


class SentenceTransformerPackager(tf.keras.layers.Layer):
    """Class to convert a Sentence Transformer to a TensorFlow model
    with its tokenizer packed in the same graph"""

    def __init__(self, model_name_or_path):
        """Initialization.
        Parameters
        ----------
        model_name_or_path: str or pathlib.Path
            Path to the Sentence Transformer model or name of the model
        """
        super().__init__()

        # Store for later use
        self.model_name_or_path = model_name_or_path

        # Load modules immediately
        with get_model_path(model_name_or_path) as model_path:
            modules_path = model_path / "modules.json"
            modules = load_json(modules_path)

            self.modules = []
            for module in modules:
                LOG.info(module)
                current = MODEL_MAPPING[module["type"]]
                current_module = import_from_string(current)
                current_module_initialized = current_module.load(
                    model_path / module["path"]
                )
                self.modules.append(current_module_initialized)

    def call(self, batch):
        """Executes the forward pass on a input `batch`"""
        for module in self.modules:
            batch = module(batch)
        return batch


class ModelConverter:
    """Class to convert a Tranformer Model that consist of tokenizer, a transformers model and
    potentially other part to a single TensorFlow graph"""

    def __init__(self, model_name_or_path, output_dir):
        """Initialization.
        Parameters
        ----------
        model_name_or_path: str or pathlib.Path
            Path to the Sentence Transformer model or name of the model
        output_dir: str
            Path to store TensorFlow graph
        """
        self.model_name_or_path = model_name_or_path
        self.embedding_size = None
        self.tf_output_dir = Path(output_dir) / "tf_model"
        self.tf_output_dir.mkdir(exist_ok=True, parents=True)
        self.converted_model = self.convert_pytorch_to_tensorflow()

    def convert_pytorch_to_tensorflow(self):
        """Converts sentence transformer pytorch model to tensorflow."""
        # Convert model
        converted_model = self.get_complete_model()
        # Save the TensorFlow model
        converted_model.save(self.tf_output_dir)
        LOG.info("Saving `%s` TF model" % self.tf_output_dir)
        return converted_model

    def validate_converted_model(self, input_test):
        """The function will fail if it finds discrepancy between different intermediate models
        embeddings and currently only works for `sentence-transformers/nq-distilbert-base-v1`
        Parameters
        ---------
        input_test: str
            str to encode and to test the models
        """
        with get_model_path(self.model_name_or_path) as model_path:
            # Get the sentence-transformers predictions for comparison
            out_sentence_transformer_pt = self.get_prediction_with_sentence_tranformers(
                input_test
            )

            # Loading the PyTorch model from Hugging Face and convert it to TF
            embeddings_converted = self.converted_model(tf.constant([input_test]))
            assert_almost_equal(
                embeddings_converted.numpy()[0],
                out_sentence_transformer_pt,
                decimal=5,
                err_msg="embedding differs across different models for one input",
            )
            LOG.info(
                "\u2713 Sentence Transformer pytorch model embeddings are almost equal with "
                "CompleteSentenceTransformer model "
            )

            LOG.info("Loading `%s` TF saved_model" % self.tf_output_dir)
            embeddings_keras_loaded_model = self.get_predictions_from_tf_model(input_test)
            assert_almost_equal(
                embeddings_converted.numpy(),
                embeddings_keras_loaded_model.numpy(),
                decimal=5,
                err_msg="Embeddings differ across different models for one input",
            )

            LOG.info(
                "\u2713 Embeddings of the original and loaded TensorFlow models are almost equal!"
            )

    def get_predictions_from_tf_model(self, input_test):
        """Loads and gets predictions from a persisted tensorflow model
        Parameters
        ----------
        input_test: str
            test query used to validate predictions
        """
        keras_loaded_model = tf.saved_model.load(str(self.tf_output_dir))
        # https://www.tensorflow.org/api_docs/python/tf/saved_model/load
        serving_signature = keras_loaded_model.signatures["serving_default"]
        embeddings_keras_loaded_model = serving_signature(
            input_sequence=tf.constant([[input_test]])
        )
        return embeddings_keras_loaded_model[KERAS_OUTPUT_NAME]

    def get_complete_model(self):
        """Creates complete model (weights and tokenizer).
        Parameters
        ----------
        input_test: str
            test query used to validate predictions
        """
        # Define model
        complete_model = SentenceTransformerPackager(self.model_name_or_path)
        inputs = tf.keras.layers.Input(
            shape=(1,), dtype=tf.string, name=KERAS_INPUT_NAME
        )
        outputs = complete_model(inputs)
        model = tf.keras.Model(inputs, outputs)
        # Test with `input_test`
        return model

    def get_prediction_with_sentence_tranformers(self, input_test):
        """Loads a sentence tranformers model and gets predictions.

        Parameters
        ----------
        input_test: str
            test query used to validate predictions
        """
        sentence_transformer_pt_model = SentenceTransformer(self.model_name_or_path)
        return sentence_transformer_pt_model.encode(input_test)


def get_parser():
    """Argument parser"""
    parser = argparse.ArgumentParser(
        description="Convert sentence transformer PyTorch model "
        "to Keras Tensorflow model"
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="sentence-transformers/nq-distilbert-base-v1",
        help="Sentence Transformer model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save the tensorflow model as a single graph",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    converter = ModelConverter(
        model_name_or_path=args.model_name_or_path, output_dir=args.output_dir
    )
    converter.validate_converted_model(input_test="This IS a test")
