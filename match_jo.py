#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ai ts=4 sts=4 et sw=4 ft=python
# coding: utf-8

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
import numpy


tokenizer = RegexpTokenizer("[a-zA-Z]+[+]*")

# largest distance between points for clustering:
eps = 0.1
min_sine_sim = 0.1
ngram_range = (1, 2)

title_weight = 0.7
description_weight = 1 - title_weight


def tokenize_and_stem(text):
    """
    Tokenize and stem English text
    """
    global stemmer, tokenizer
    return [stemmer.stem(token) for token in tokenizer.tokenize(text)]


def ilines(filename1=None, filename2=None, titles=True, descriptions=True):
    """
    File line iterator
    Yields "raw" documents
    """
    global jo_ids, last_target_row, first_source_row

    # Line no -> code
    jo_ids = []
    
    # TARGET:
    if filename2 is not None:
        with open(filename2, mode='r') as fileobj:
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
                res = ''
                if descriptions:
                    res = row[3]
                if titles:
                    res += ' ' + row[2]
                yield res
            last_target_row = i-1 # row #0 is the header row
            first_source_row = i # row #0 is the header row

    # SOURCE:
    if filename1 is not None:
        with open(filename1, mode='r') as fileobj:
            rdr = csv.reader(fileobj)

            for i, row in enumerate(rdr):
                jo_ids.append(i)
                res = ''
                if descriptions:
                    res = row[1]
                if titles:
                    res += ' ' + row[0]
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

# ceate space basad on both descripitons and titles from both files:
tfidf_vectorizer.fit(ilines("data/File 1.csv", "data/File 2.csv"))

matrix1 = tfidf_vectorizer.transform(ilines(filename1="data/File 1.csv", titles=False))
matrix2 = tfidf_vectorizer.transform(ilines(filename2="data/File 2.csv", titles=False))

description_similarity = cosine_similarity(matrix1, matrix2)
print("*** Smilarity matrix (descriptions) shape:", description_similarity.shape)

title_matrix1 = tfidf_vectorizer.transform(ilines(filename1="data/File 1.csv", descriptions=False))
title_matrix2 = tfidf_vectorizer.transform(ilines(filename2="data/File 2.csv", descriptions=False))

title_similarity = cosine_similarity(title_matrix1, title_matrix2)
print("*** Smilarity matrix (titles) shape:", title_similarity.shape)


# if description is absent, take the similarity based on the title
# or else weighed on both title and description
similarity = numpy.where(
        description_similarity == 0,
        title_similarity,
        description_weight * description_similarity + title_weight * title_similarity)


def take_top10(row):
    return sorted([r for r in zip(range(row.size), row.tolist()) if r[1] > min_sine_sim],
           key=lambda r: -r[1])[:10]


if __name__ == "__main__":
    top10 = []
    for row in similarity:
        top10.append(take_top10(row))

    for i, l in enumerate(top10, 1):
        if l != []:
            print("{}: ".format(i),  end="")
            print(", ".join(( "{} ({:.2f})".format(k+2, score) for (k, score) in l)))

