# -*- coding: utf-8 -*-
# vim: ai ts=4 sts=4 et sw=4 ft=python

import dill
import nltk
from stem import IndonesianStemmer
import string
import os
import re
from collections import defaultdict

import multiprocessing
import dill
import csv
import string
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from id_stopwords import stopwords
import math
from itertools import groupby

# Location of category product similarity files:
data_dir = "/data/python-scrapy-data-mining/"

stemmer = IndonesianStemmer()
# Include digits since it is important for product distinciont (eg a model number or version number):
tokenizer = nltk.tokenize.RegexpTokenizer("[\w'0-9]+|\$[\d\.]+|\S+", flags=re.UNICODE)

categories = category_products = category_rows = prod_ids = None

def backup_object(obj, file_name=None):
    if file_name is None:
        file_name = "{0}.pkl".format(id(obj))
        print "*** Object stored in:", file_name
    with open(file_name, "wb") as bf:
        dill.dump(obj, bf)

        
def norm(vect):
    return math.sqrt(sum((v * v for v in vect.values())))


def distance(v1, v2):
    v1, n1 = v1[:2]
    v2, n2 = v2[:2] 
    dist = sum((v1[w] * v2[w] for w in set(v1).intersection(v2)), 0.0)
    return dist / (n1 * n2)

        
def restore_object(file_name):
    with open(file_name, "rb") as bf:
        return dill.load(bf)


def chunks(l, num):
    """
    splits list in num chunks
    for running in parallel
    """
    n = len(l) / num + 1
    return [l[i:i + n] for i in range(0, len(l), n)]


def tokenize_and_stem_ID(text):
    """
    Tokenize and stem Indonesian text
    """
    global stemmer, tokenizer
    return [stemmer.stem(token) for token in tokenizer.tokenize(text)]


def get_categories():

    global categories

    if categories is None:
      
        # Product categorie list:
        with open(os.path.join(data_dir, "urls.csv")) as raw:
            categories = list(set(('/'.join(parts[4:-1]) for parts in (url[1].split('/') for url in csv.reader(raw))
                    if parts[3] == 'p')))

    return categories


def get_prod_ids():

    global prod_ids

    if prod_ids is None:

        with open(os.path.join(data_dir, "corpus.csv")) as raw:
            prod_ids = [int(prod_id) for prod_id, _, url in csv.reader(raw)]

    return prod_ids


def get_category_products():
    """
    returns diction {'<category>': [<product ID list>], ... }
    """
    global category_products

    if category_products is None:
      
        with open(os.path.join(data_dir, "corpus.csv")) as raw:
            category_products = dict(
                (c, sorted([prod_id for _, prod_id in c_ids])) for c, c_ids in groupby(
                    sorted( (('/'.join(url.split('/')[4:-1]), int(prod_id)) for prod_id, _, url in csv.reader(raw)),
                            key=lambda t: t[0]),
                        key=lambda t: t[0]
                    )
            )

    return category_products


def get_category_rows():
    """
    returns diction {'<category>': [<row number in the corpus file>], ... }
    """
    global category_rows

    if category_rows is None:
      
        with open(os.path.join(data_dir, "corpus.csv")) as raw:
            category_rows = dict(
                (c, sorted([row_no for _, row_no in row_nos])) for c, row_nos in groupby(
                        sorted( (('/'.join(url.split('/')[4:-1]), row_no) for row_no, (_, _, url) in enumerate( csv.reader(raw))),
                            key=lambda t: t[0]),
                        key=lambda t: t[0]
                    )
            )

    return category_rows


def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):

    if type(unicode_csv_data) is str: ## file name
        unicode_csv_data = open(unicode_csv_data, "rb")

    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

        
def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')