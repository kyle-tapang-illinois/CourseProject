# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:07:32 2022

@author: WilliamKiger
"""
import pandas as pd


def main():
    data = pd.read_csv('database\data.csv', index_col=0)
    sentiment_scores = pd.read_csv('database\data_sentiment_scores.csv', index_col=0)


if __name__ == '__main__':
    main()
