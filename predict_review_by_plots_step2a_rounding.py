import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

def main(plots_json_gz):
    # plots_json_gz is an output file generated from step1: join tables
    data = pd.read_json(plots_json_gz, orient='records', lines=True)
    print('Data size before dropping empty values:', len(data))

    # Drop empty values/plots
    data = data.dropna()
    # Round review averages and convert them to string: to be used as class labels
    data['audience_average'] = data['audience_average'].round()
    # Rounding cont'd: convert to int to trim decimals, then convert to str
    data['audience_average'] = data['audience_average'].astype('int').astype('str')
    print('Data size after dropping empty values:', len(data))

    X = data['omdb_plot'].values
    y = data['audience_average'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    bayes_model = make_pipeline(
        CountVectorizer(),
        TfidfTransformer(use_idf=True),
        MultinomialNB(),
    )
    bayes_model.fit(X_train, y_train)

    # Score with training data: I got 0.876...
    nb_train = bayes_model.score(X_train, y_train)
    # Score with validation data: I got 0.598... which is too low
    nb_valid = bayes_model.score(X_valid, y_valid)

    knn_model = make_pipeline(
        CountVectorizer(),
        TfidfTransformer(use_idf=True),
        KNeighborsClassifier(n_neighbors=100)
    )
    knn_model.fit(X_train, y_train)

    knn_train = knn_model.score(X_train, y_train)
    knn_valid = knn_model.score(X_valid, y_valid)

    svm_model = make_pipeline(
        CountVectorizer(),
        TfidfTransformer(use_idf=True),
        SVC(C=1e4, kernel='rbf', gamma='scale')
    )
    svm_model.fit(X_train, y_train)

    # Score with training data: I got 0.999...
    svm_train = svm_model.score(X_train, y_train)
    # Score with validation data: I got 0.596... which is way too low
    svm_valid = svm_model.score(X_valid, y_valid)

    neural_model = make_pipeline(
        CountVectorizer(),
        TfidfTransformer(use_idf=True),
        MLPClassifier()
    )
    neural_model.fit(X_train, y_train)

    mlp_train = neural_model.score(X_train, y_train)
    mlp_valid = neural_model.score(X_valid, y_valid)

    # This program used rounding method to categorize review scores:
    # Maybe use percentile?
    print(OUTPUT_TEMPLATE.format(
        nb_train=nb_train,
        nb_valid=nb_valid,
        knn_train=knn_train,
        knn_valid=knn_valid,
        svm_train=svm_train,
        svm_valid=svm_valid,
        mlp_train=mlp_train,
        mlp_valid=mlp_valid
    ))

OUTPUT_TEMPLATE = (
    '\n                          Train Score | Validation Score\n'
    'MultinomialNB classifier: {nb_train:.3g}           {nb_valid:.3g}\n'
    'kNN classifier:           {knn_train:.3g}           {knn_valid:.3g}\n'
    'SVM classifier:           {svm_train:.3g}           {svm_valid:.3g}\n'
    'MLP classifier:           {mlp_train:.3g}           {mlp_valid:.3g}\n'
)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 program.py <input_json_gz>')
        print('  e.g. python3 program.py plots.json.gz')
    else:
        plots_json_gz = sys.argv[1]
        main(plots_json_gz)