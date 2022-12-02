"""Convert a sentence transformer model to a TensorFlow graph with embedded tokenizer
Heavily inspired by: https://www.tensorflow.org/text/guide/bert_preprocessing_guide
"""
import argparse
import logging
from pathlib import Path

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
        modules_path = self.get_sentence_transformer_filepath(
            base_path=model_name_or_path, filename="modules.json"
        )
        modules = load_json(modules_path)
        self.modules = []
        for module in modules:
            LOG.info(module)
            current = MODEL_MAPPING[module["type"]]
            current_module = import_from_string(current)
            current_module_initialized = current_module.load(
                modules_path.parents[0] / module["path"]
            )
            self.modules.append(current_module_initialized)

    def get_sentence_transformer_filepath(
        self, filename, base_path, fail_if_not_found=True
    ):
        """Given a `filename` and a `base_path` looks to find the file under the path.
        If not found, returns None. Used because in different sentence-transformers versions
        some files are saved under different locations."""
        if not Path(
            base_path
        ).exists():  # The user provided a model name and not a path to model
            _ = SentenceTransformer(base_path)
            from torch.hub import _get_torch_home

            torch_cache_home = _get_torch_home()
            base_path = (
                Path(torch_cache_home)
                / "sentence_transformers"
                / "_".join([base_path.replace("/", "_")])
            )
        file_location = list(
            Path(base_path).rglob(filename)
        )  # rglob = recursively glob
        assert len(file_location) in (
            0,
            1,
        )  # Files either exists once (by ST design) or not
        if file_location:
            return file_location[
                0
            ]  # Sent. Transformer models dy design have a single such file
        if fail_if_not_found:  # We did not find the file and we should fail
            raise FileNotFoundError(
                f"Not found {filename} at {base_path} recursively."
                f" This is unexpected for sentence-transformers"
            )
        return None

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
