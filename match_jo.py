#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ai ts=4 sts=4 et sw=4 ft=python

##%matplotlib inline

import os
import sys
from collections import defaultdict
from itertools import groupby
from random import shuffle
from collections import namedtuple
import codecs
import multiprocessing
import csv


import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

import utils
from utils import tokenizer, stopwords, stemmer

# largest distance between points for clustering:
eps = 0.1
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
    """
    global jo_ids, last_occ_row
    # Line no -> code
    jo_ids = [None]
    
    with open(file1_name, mode='r') with fileobj:
        rdr = csv.reader(fileobj)

        for i, row in enumerate(rdr):
            if i == 0: ## skip header
                continue
            # industry, occ_code, occ_name, occ_descriptions
            # sample_job_titles, task1, task2, task3, task4, task5
            jo_ids.append(row[1])
            # include industry, occ_name, occ_descriptions
            # task1, task2, task3, task4, task5
            res = row[0] + ' ' + row[2] + ' ' + row[3] + ' ' + ' '.join(row[5:]
            yield res
        last_occ_row = i # row #0 is the header row

    with open(file2_name, mode='r') with fileobj:
        rdr = csv.reader(fileobj)

        for i, row in enumerate(rdr):
            jo_ids.append(i)
            # include industry, occ_name, occ_descriptions
            # task1, task2, task3, task4, task5
            res = row[1]
            yield res
                                                                        

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(
    #input="filename",
    #max_df=1.1,
    #max_features=200000,
    #min_df=0.01,
    stop_words=stopwords,
    use_idf=True,
    tokenizer=utils.tokenize_and_stem,
    ngram_range=(1,1))


tfidf_matrix = tfidf_vectorizer.fit_transform(ilines())
print "*** Vector space shape:", tfidf_matrix.shape

def get_clusters(category, eps=eps):
    """
    Runs clustering and returns labeled non-distinct product entires
    grouped by lables in a dictionary.
    """
    global tfidf_matrix, category_rows, row_nos

    row_nos = category_rows[category]

    # create and fit the model:
    db = DBSCAN(eps=eps).fit(tfidf_matrix[row_nos])
    return [
        (label, [p for _, p in prodi]) for label, prodi in groupby(
                ((l, prod_ids[row_nos[i]]) for i, l in sorted(enumerate(db.labels_), key=lambda e: -e[1]) if l != -1),
                key=lambda e: e[0]
        )
    ]


def category_clusters(categories):
    return [(c, get_clusters(c)) for c in categories]

shuffle(categories)

process_num = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=process_num)
clusters = pool.map(category_clusters, utils.chunks(categories, process_num))

clusters = filter(lambda c: c[1], sum(clusters, []))

for c, labels in clusters:
    for l, products in labels:
        print "\"{0}\", {1}: {2}".format(c, l, ','.join(map(str, products)))

