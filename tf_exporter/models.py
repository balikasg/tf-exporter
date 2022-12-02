import json
from pathlib import Path
import tensorflow as tf
import tensorflow_text as text
from transformers import TFAutoModel
from sentence_transformers import models


TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
MODULE_CONFIG = "config.json"
PYTORCH_MODEL = "pytorch_model.bin"
TF_MODEL = "tf_model.h5"
MAX_SEQ_LENGTH_DEFAULT = 256


class Transformer(tf.keras.layers.Layer):
    """This layer loads a transformer model with its tokenizer"""

    def __init__(
        self, input_path, backend, do_lower_case, max_seq_length=MAX_SEQ_LENGTH_DEFAULT
    ):
        """Initialization.
        Parameters
        ----------
        input_path: str or pathlib.Path
            Path to the Sentence Transformer model
        backend: str
            'tf' for tensorflow, 'pt' for pytorch
        do_lower_case: bool
            whether the tokenizer should lowercase
        max_seq_length:
            maximum sequense length for the tokenizer
        """
        super().__init__()
        self.model = TFAutoModel.from_pretrained(input_path, from_pt=(backend == "pt"))
        self._vocab = self._read_vocabulary(input_path / "vocab.txt")
        # TODO: extract the below in a dedicated method
        self._start_token = self._vocab.index(b"[CLS]")
        self._end_token = self._vocab.index(b"[SEP]")
        self._mask_token = self._vocab.index(b"[MASK]")
        self._unk_token = self._vocab.index(b"[UNK]")
        self._max_seq_length = max_seq_length
        self.tokenizer = self._get_tokenizer(lowercase=do_lower_case)
        self.trimmer = text.RoundRobinTrimmer(max_seq_length=max_seq_length)

    def call(self, batch):
        """Executes the forward pass on a input `batch`"""
        tokens = self.tokenizer.tokenize(batch).merge_dims(-2, -1)
        tokens = self.trimmer.trim([tokens])
        tokens = self.add_start_end(tokens[0])
        input_word_ids, input_mask = text.pad_model_inputs(
            tokens, max_seq_length=self._max_seq_length
        )
        model_output = self.model(
            {"input_ids": input_word_ids, "attention_mask": input_mask}
        )
        return model_output[0], input_mask

    def _get_tokenizer(self, lowercase):
        """Instantiates a tf_text.BertTokenizer based on a vocabulary lookup.

        Parameters
        ----------
        lowercase: bool
            Whether to lowercase the input sequences
        """
        lookup_table = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=self._vocab,
                key_dtype=tf.string,
                values=tf.range(
                    tf.size(self._vocab, out_type=tf.int64), dtype=tf.int64
                ),
                value_dtype=tf.int64,
            ),
            num_oov_buckets=1,
        )
        return text.BertTokenizer(
            lookup_table, token_out_type=tf.int64, lower_case=lowercase
        )

    def add_start_end(self, ragged):
        """Adds START and END special token ids in the ragged tensor `ragged`"""
        count = ragged.bounding_shape()[0]  # Gets shape (batch_size) of ragged
        starts = tf.fill([count, 1], tf.cast(self._start_token, tf.int64))
        ends = tf.fill([count, 1], tf.cast(self._end_token, tf.int64))
        return tf.concat(values=[starts, tf.squeeze(ragged, axis=1), ends], axis=1)

    @staticmethod
    def _read_vocabulary(vocabulary_path):
        """Reads a vocabulary file where each token is in a separate line.
        Returns a list of these tokens.
        """
        with open(str(vocabulary_path), "r", encoding="utf8") as infile:
            file_content = infile.read().split()
        return [str.encode(word) for word in file_content]

    @staticmethod
    def load(input_path):
        """Loads a transformer model from a path"""
        input_path = Path(input_path)
        with open(input_path / TOKENIZER_CONFIG_NAME) as infile:
            config = json.load(infile)
        if (input_path / TF_MODEL).exists():
            backend = "tf"
        elif (input_path / PYTORCH_MODEL).exists():
            backend = "pt"
        else:
            raise ImportError(
                f"Could not find {PYTORCH_MODEL} or {TF_MODEL} under {input_path}."
            )
        max_seq_length = config.get(
            "model_max_length", config.get("max_len", MAX_SEQ_LENGTH_DEFAULT)
        )
        return Transformer(
            input_path=input_path,
            backend=backend,
            do_lower_case=config["do_lower_case"],
            max_seq_length=max_seq_length,
        )


class Pooling(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """Mean pooling using the attention mask. inputs should be a list:
        [token_embeddings, attention_mask]"""
        token_embeddings, attention_mask = inputs[0], inputs[1]
        input_mask_expanded = tf.cast(
            tf.broadcast_to(
                tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)
            ),
            tf.float32,
        )
        numerator = tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1)
        denominator = tf.clip_by_value(
            tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max
        )
        output = numerator / denominator
        return output

    @staticmethod
    def load(path):
        return Pooling()


class Normalize(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """Mean pooling using the attention mask. inputs should be a list:
        [token_embeddings, attention_mask]"""
        output, _ = tf.linalg.normalize(inputs, 2, axis=1)
        return output

    @staticmethod
    def load(path):
        return Normalize()


class TfDense(tf.keras.layers.Layer):
    """Implements the transformation of a pytorch dense layer to a tensorflow dense layer."""

    def __init__(self, tf_dense_layer: tf.keras.layers.Dense):
        super().__init__()
        self.dense_layer = tf_dense_layer

    def call(self, inputs):
        """Executes the forward pass on a input `inputs`"""
        return self.dense_layer(inputs)

    @staticmethod
    def load(input_path):
        """Instantiates a tf dense keras models from a pytorch one"""
        input_path = Path(input_path)
        with open(input_path / MODULE_CONFIG) as infile:
            config = json.load(infile)
        if (input_path / PYTORCH_MODEL).exists():
            # Gets weights as an OrderedDict with keys 'linear.weight' and 'linear.bias'
            dense_weights = models.Dense.load(input_path).state_dict()
            # Get a tf linear model
            activation_fn = config["activation_function"].rsplit(".", 1)[-1].lower()
            tf_dense = tf.keras.layers.Dense(
                units=config["out_features"],
                input_shape=(config["in_features"],),
                activation=activation_fn,
                weights=[
                    dense_weights["linear.weight"].numpy().T,
                    dense_weights["linear.bias"].numpy().T,
                ],
            )
            return TfDense(tf_dense)
        raise ImportError(
            f"Could not find {PYTORCH_MODEL} under {input_path}. "
            f"{TF_MODEL} not currently supported for Dense Layer initialization."
        )
