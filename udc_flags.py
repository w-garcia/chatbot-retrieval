import tensorflow as tf
import os

# Originally from prepare_data.py
tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

tf.flags.DEFINE_string(
    "input_dir", os.path.abspath("chatbot-retrieval/data"),
    "Input directory containing original CSV data files (default = './data')")

tf.flags.DEFINE_string(
    "output_dir", os.path.abspath("chatbot-retrieval/data"),
    "Output directory for TFrEcord files (default = './data')")

tf.flags.DEFINE_string("model_dir", None, "Directory to load/store model checkpoints from")
tf.flags.DEFINE_string("trained_model_dir", "chatbot-retrieval/runs/1479850175", "Directory to load/store model checkpoints from")

tf.flags.DEFINE_string("vocab_processor_file", "chatbot-retrieval/data/vocab_processor.bin", "Saved vocabulary processor file")

tf.flags.DEFINE_string("test_file", "./data/test.tfrecords", "Path of test data in TFRecords format")
#tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("test_batch_size", 16, "Batch size for testing")

#tf.flags.DEFINE_string("input_dir", "./data", "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
#tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 10000, "Evaluate after this many train steps")

# Model Parameters
tf.flags.DEFINE_integer(
  "vocab_size",
  91620,
  "The size of the vocabulary. Only change this if you changed the preprocessing")

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_context_len", 160, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 80, "Truncate utterance to this length")
tf.flags.DEFINE_integer("min_word_frequency", 2, "Minimum word frequency")

# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", None, "Path to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", None, "Path to vocabulary.txt file")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")
