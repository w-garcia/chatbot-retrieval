import os
import time
import itertools
import sys
import tensorflow as tf
import udc_model as udc_model
import udc_hparams as udc_hparams
import udc_metrics as udc_metrics
import udc_inputs as udc_inputs
from models.dual_encoder import dual_encoder_model
import udc_flags

FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

tf.logging.set_verbosity(FLAGS.loglevel)

if __name__ == "__main__":
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir,
    config=tf.contrib.learn.RunConfig())

  input_fn_test = udc_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[FLAGS.test_file],
    batch_size=FLAGS.test_batch_size,
    num_epochs=1)

  eval_metrics = udc_metrics.create_evaluation_metrics()
  estimator.evaluate(input_fn=input_fn_test, steps=None, metrics=eval_metrics)
