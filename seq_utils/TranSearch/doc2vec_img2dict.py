import os, sys
import random
import json
import argparse
import collections
import itertools
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from ast import literal_eval
from gensim.models import doc2vec

import config
import image_process


parser = argparse.ArgumentParser()

parser.add_argument("--embedding_size", 
    type=int,
    default=512,
    help="doc dimension.")
parser.add_argument("--window_size", 
    type=int,
    default=3,
    help="sentence window size.")
parser.add_argument("--img_feature_file", 
    type=str,
    default="data/image_features_Musical_Instruments.b",
    help="the raw image feature file")

parser.add_argument('--dataset', type=str, default='Musical_Instruments')
parser.add_argument('--main_path', type=str, default='/home/share/yinxiangkun/data/cold_start/')
parser.add_argument('--stop_file', type=str, default='../../seq_utils/TranSearch/stopwords.txt')
parser.add_argument('--processed_path', type=str,
                    default='/home/share/yinxiangkun/processed/cold_start/ordinary/Musical_Instruments/')

FLAGS = parser.parse_args()


######################### PREPARE DATA ############################
full_path = os.path.join(FLAGS.processed_path, '{}_full.csv'.format(FLAGS.dataset))
full_data = pd.read_csv(full_path)
full_data.query_ = full_data.query_.apply(literal_eval)
full_data.reviewText = full_data.reviewText.apply(literal_eval)
asin_set = set(full_data.asin.unique())
img_path = os.path.join(FLAGS.main_path, FLAGS.img_feature_file)

# gather reviews to same asins
raw_doc = collections.defaultdict(list)
for k, v in zip(full_data.asin, full_data.reviewText):
    raw_doc[k].append(v)

# concatenate the reviews together
for k in raw_doc.keys():
    m = [i for i in raw_doc[k]]
    m = list(itertools.chain.from_iterable(m))
    raw_doc[k] = m

# for query, it's hard to tag, so we just random tag them
query_idx, query_dict = 0, {} 
for q in full_data['query_']:
    if repr(q) not in query_dict:
            query_dict[repr(q)] = query_idx
            raw_doc[query_idx] = q
            query_idx += 1

########################## MODEL TRAINING #######################
analyzed_doc = collections.namedtuple(
                            'AnalyzedDocument', 'words tags')
docs = [analyzed_doc(raw_doc[d], [d]) for d in raw_doc.keys()]

alpha_val = 0.025
min_alpha_val = 1e-4
passes = 40

alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)
model = doc2vec.Doc2Vec(
    min_count=2,
    workers=4,
    epochs=10,
    vector_size=FLAGS.embedding_size,
    window=FLAGS.window_size)

model.build_vocab(docs) # Building vocabulary
for epoch in range(passes):
    random.shuffle(docs)

    model.alpha, model.min_alpha = alpha_val, alpha_val
    model.train(docs, total_examples=len(docs), epochs=model.epochs)
    alpha_val -= alpha_delta
    print('epochs:', epoch)

############################### SAVE TO DISK ################################

doc2model_path = FLAGS.processed_path + '{}_doc2model'.format(FLAGS.dataset)
query_path = FLAGS.processed_path + '{}_query.json'.format(FLAGS.dataset)
img_feature_path = FLAGS.processed_path + '{}_img_feature.npy'.format(FLAGS.dataset)

model.save(doc2model_path)

json.dump(query_dict, open(query_path, 'w'))
print("The query number is %d." %len(query_dict))

img_feature_dict = image_process._get_feature(asin_set, img_path)
np.save(img_feature_path, img_feature_dict)
