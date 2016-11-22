import os
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import udc_model as udc_model
import udc_hparams as udc_hparams
import udc_inputs as udc_inputs
import udc_metrics
from models.dual_encoder import dual_encoder_model
from models.helpers import load_vocab
import csv
import random
import nltk
import operator
from collections import defaultdict
from gensim import corpora, models, similarities
from pathlib import Path
import udc_flags

FLAGS = tf.flags.FLAGS

TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.csv"))
DICT_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "deerwester.dict"))
CORPUS_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "deerwester.mm"))

if not FLAGS.trained_model_dir:
    print("You must specify a model directory")
    sys.exit(1)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_processor_file)

# Load your own data here
#INPUT_CONTEXT = "Example context"


def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)


def get_features(context, utterance):
    context_matrix = np.array(list(vp.transform([context])))
    utterance_matrix = np.array(list(vp.transform([utterance])))
    context_len = len(context.split(" "))
    utterance_len = len(utterance.split(" "))
    features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
    }
    return features, None

hparams = udc_hparams.create_hparams()


def populate_cache():
    cache = []
    with open(TRAIN_FILE, 'r') as csvFile:
        reader = csv.DictReader(csvFile)
        count = 0
        for row in reader:
            context = row["Context"]
            utterance = row["Utterance"]
            cache.append({"Context": context, "Utterance": utterance})
            count += 1
            if count == 10000:
                return cache

    return cache

CACHE = populate_cache()


def create_dict():
    texts = []
    for row in CACHE:
        texts.append(nltk.word_tokenize(row["Context"]))

    my_dict = Path(DICT_FILE)
    my_corpus = Path(CORPUS_FILE)

    dict = corpora.Dictionary(texts)

#if not my_dict.is_file():
    dict.save(DICT_FILE)

#if not my_corpus.is_file():
    corpus = [dict.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(CORPUS_FILE, corpus)


def get_responses(n, in_string):
    """
    :param n:
    :param in_string:
    :return: n most common utterances to input string
    """
    create_dict()

    dict = corpora.Dictionary.load(DICT_FILE)
    print(dict)
    corpus = corpora.MmCorpus(CORPUS_FILE)
    print(corpus)

    lsi = models.LsiModel(corpus, id2word=dict)
    vec_bow = dict.doc2bow(in_string.lower().split())
    vec_lsi = lsi[vec_bow] #convert query to lsi space

    similarity_list = []
    for i, score in vec_lsi:
        similarity_list.append({"score": score, "Utterance": CACHE[i]["Utterance"]})

    result = []
    count = 0

    for i in sorted(similarity_list, key=lambda k: k["score"], reverse=True):
        result.append(i["Utterance"])
        count += 1
        if count == n:
            return result

    return result


#if __name__ == "__main__":
def retrieve_response(input_c):
    #input_c = "hello"
    potential_responses = get_responses(10, input_c)
    model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
    estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.trained_model_dir)

    # Ugly hack, seems to be a bug in Tensorflow
    # estimator.predict doesn't work without this line
    estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))
    response = "..."
    max = 0
    for r in potential_responses:
        prob = estimator.predict(input_fn=lambda: get_features(input_c, r), as_iterable=True)
        val = next(prob)[0]
        if val > max:
            response = r
            max = val
        #print("{}: {:g}: {}".format(r, prob[0,0], prob))
    return response
