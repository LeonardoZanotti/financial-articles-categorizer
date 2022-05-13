import pickle
import sys

import pandas as pd

model = pickle.load(open('', 'rb'))
tfidf_vectorizer = pickle.load(open('', 'rb'))
label_encoder = pickle.load(open('', 'rb'))


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
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
