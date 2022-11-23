# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:07:32 2022

@author: WilliamKiger
"""

import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import Sentiment as s

stopwords = set(open('lemur-stopwords.txt', 'r').read().split())


def remove_stopwords(document):
    return [word for word in document if word not in stopwords]


def get_corpus(data, sentiment_scores):
    tokenized_corpus = []
    for document in data.itertuples():
        id = document[0]
        sentiment_score_is_from_content = sentiment_scores.values[id][1]
        title = 1
        content = 4

        doc = document[content] if sentiment_score_is_from_content else document[title]
        doc = doc.lower()
        doc = doc.split(" ")
        doc = remove_stopwords(doc)
        tokenized_corpus.append(doc)

    return tokenized_corpus


def ranker(query):
    data = pd.read_csv('database\data.csv', index_col=0, encoding='utf8')
    sentiment_scores = pd.read_csv('database\data_sentiment_scores.csv', index_col=0)

    tokenized_query = remove_stopwords(query.split(" "))
    tokenized_corpus = get_corpus(data, sentiment_scores)
    bm25 = BM25Okapi(tokenized_corpus)

    path_to_model = "model\sentiment_model.pt"
    model, device, tokenizer, init_token_id, eos_token_id = s.loadSentimentModel(path_to_model)
    query_sentiment_score = s.predictSentiment(model, device, tokenizer, init_token_id, eos_token_id, query)

    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    sentiment_scores = np.array(sentiment_scores.iloc[:, 0])
    relative_sentiment_scores = np.abs(sentiment_scores - query_sentiment_score)
    sentiment_adjusted_bm25_scores = np.log(bm25_scores / relative_sentiment_scores)
    data["sentiment_adjusted_bm25_scores"] = sentiment_adjusted_bm25_scores
    sorted_by_adjusted_bm25_scores = data.sort_values(by=['sentiment_adjusted_bm25_scores'], ascending=False)
    top_n = 5
    sorted_by_adjusted_bm25_scores = sorted_by_adjusted_bm25_scores[["title", "link"]][:top_n]
    sorted_by_adjusted_bm25_scores.rename(columns={'link': 'url'}, inplace=True)
    return sorted_by_adjusted_bm25_scores


def main():
    query = "this is a test query"
    print(ranker(query))


if __name__ == '__main__':
    main()
