#!/usr/bin/env python
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


import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN


# adjust this path to point to 'site-aggregation'
sys.path.append(os.path.join(os.environ["HOME"], "site-aggregation"))

import crawler.settings
# set MySQL DB root user password here or in site-aggregation private settings:
crawler.settings.MYSQL_PASSWORD = os.environ.get("MYSQL_PWD", crawler.settings.MYSQL_PASSWORD)

from crawler import db
from crawler.db import Product

import utils
from id_stopwords import stopwords


# largest distance between points for clustering:
eps = 0.1
include_title = True
include_description = True


#list(Product.select(Product.title, Product.url).limit(10))
db.mysql_db.connect()

CsvRow = namedtuple('CsvRow', 'id, title, url')
def ilines(
        file_name=None,
        include_description=include_description):
    """
    File line iterator
    """
    global categories, prod_ids, category_rows
    prod_ids = []
    category_rows = defaultdict(lambda: [])

    for i, row in enumerate(
            Product.select(
                Product.id,
                Product.title,
                Product.description,
                Product.url)
            .where(Product.url % "https://www.bukalapak.com/p/%") if file_name is None
            else (CsvRow(*l) for l in utils.unicode_csv_reader(file_name)) ):
        prod_ids.append(row.id)
        category = '/'.join(row.url.split('/')[4:-1])
        category_rows[category].append(i)
        res = row.title
        if file_name is None and include_description and row.description:
            res += (' ' + row.description)
        yield res

        categories = sorted(category_rows.keys())

    return

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(
    #input="filename",
    #max_df=1.1,
    #max_features=200000,
    #min_df=0.01,
    stop_words=stopwords,
    use_idf=True,
    tokenizer=utils.tokenize_and_stem_ID,
    ngram_range=(1,1))


db.mysql_db.connect()
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

