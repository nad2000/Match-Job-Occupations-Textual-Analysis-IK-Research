#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ai ts=4 sts=4 et sw=4 ft=python
# coding: utf-8

import multiprocessing
import csv

import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import utils
from utils import tokenizer, stopwords, stemmer

from scipy.spatial.distance import cosine
from importlib import reload
from sklearn.metrics.pairwise import cosine_similarity


tokenizer = RegexpTokenizer("[a-zA-Z]+[\w']+[+]*")

# largest distance between points for clustering:
eps = 0.1
min_sine_sim = 0.1
ngram_range = (1, 1)

include_title = False
include_description = True

# TODO:
title_weight = 0.5
description_weight = 1 - title_weight


def tokenize_and_stem(text):
    """
    Tokenize and stem English text
    """
    global stemmer, tokenizer
    return [stemmer.stem(token) for token in tokenizer.tokenize(text)]


def ilines(file1_name, file2_name, titles=False):
    """
    File line iterator
    Yields "raw" documents
    """
    global jo_ids, last_target_row, first_source_row

    # Line no -> code
    jo_ids = []
    
    # TARGET:
    with open(file2_name, mode='r') as fileobj:
        rdr = csv.reader(fileobj)

        for i, row in enumerate(rdr):
            if i == 0: ## skip header
                continue
            # industry, occ_code, occ_name, occ_descriptions
            # sample_job_titles, task1, task2, task3, task4, task5
            jo_ids.append(row[1])
            # include industry, occ_name, occ_descriptions
            # task1, task2, task3, task4, task5
            #res = row[0] + ' ' + row[2] + ' ' + row[3] + ' ' + ' '.join(row[5:])
            res = row[3]
            yield res
        last_target_row = i-1 # row #0 is the header row
        first_source_row = i # row #0 is the header row

    # SOURCE:
    with open(file1_name, mode='r') as fileobj:
        rdr = csv.reader(fileobj)

        for i, row in enumerate(rdr):
            jo_ids.append(i)
            res = row[1]
            yield res


#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(
    #input="filename",
    #max_df=1.1,
    #max_features=200000,
    #min_df=0.01,
    lowercase=True,
    analyzer='word',
    stop_words=stopwords,
    use_idf=True,
    tokenizer=tokenize_and_stem,
    ngram_range=ngram_range)


tfidf_matrix = tfidf_vectorizer.fit_transform(ilines("data/File 1.csv", "data/File 2.csv"))
print("*** Vector space shape:", tfidf_matrix.shape)


similarity = cosine_similarity(tfidf_matrix[first_source_row:,:], tfidf_matrix[:first_source_row,:])
print("*** Smilarity matrix shape:", similarity.shape)


def take_top10(row):
    return sorted([r for r in zip(range(row.size), row.tolist()) if r[1] > min_sine_sim],
           key=lambda r: -r[1])[:10]


top10 = []
for row in similarity:
    top10.append(take_top10(row))

for i, l in enumerate(top10, 1):
    if l != []:
        print("{}: ".format(i),  end="")
        print(", ".join(( "{} ({:.2f})".format(k+2, score) for (k, score) in l)))

