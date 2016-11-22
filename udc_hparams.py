import tensorflow as tf
from collections import namedtuple
import udc_flags

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [
    "batch_size",
    "embedding_dim",
    "eval_batch_size",
    "learning_rate",
    "max_context_len",
    "max_utterance_len",
    "optimizer",
    "rnn_dim",
    "vocab_size",
    "glove_path",
    "vocab_path"
  ])

def create_hparams():
  return HParams(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    vocab_size=FLAGS.vocab_size,
    optimizer=FLAGS.optimizer,
    learning_rate=FLAGS.learning_rate,
    embedding_dim=FLAGS.embedding_dim,
    max_context_len=FLAGS.max_context_len,
    max_utterance_len=FLAGS.max_utterance_len,
    glove_path=FLAGS.glove_path,
    vocab_path=FLAGS.vocab_path,
    rnn_dim=FLAGS.rnn_dim)