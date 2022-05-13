#!/usr/bin/env python3.7
# Leonardo José Zanotti

import pickle
import sys

import pandas as pd

model = pickle.load(open('financial_text_classifier.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('financial_text_vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('financial_text_encoder.pkl', 'rb'))


def process(inPath, outPath):
    # read input file
    input_df = pd.read_csv(inPath)
    # vectorize the data
    features = tfidf_vectorizer.transform(input_df['body'])
    # predict the classes
    predictions = model.predict(features)
    # convert output labels to categories
    input_df['category'] = label_encoder.inverse_transform(predictions)
    # save results to csv
    output_df = input_df[['id', 'category']]
    output_df.to_csv(outPath, index=False)


if __name__ == '__main__':
    if (len(sys.argv) != 3):
        exit('Use python3.7 classify_financial_articles.py inputPath outputPath')
    process(sys.argv[1], sys.argv[2])
    exit('Output saved to ' + sys.argv[2])
