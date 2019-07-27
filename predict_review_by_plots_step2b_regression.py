import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main(plots_json_gz):
    # plots_json_gz is an output file generated from step1: join tables
    data = pd.read_json(plots_json_gz, orient='records', lines=True)
    print('Data size before dropping empty values:', len(data))

    # Drop empty values/plots
    data = data.dropna()
    print('Data size after dropping empty values:', len(data))

    X = data['omdb_plot'].values
    y = data['audience_average'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    regression_model = make_pipeline(
        CountVectorizer(),
        TfidfTransformer(use_idf=True),
        LinearRegression(fit_intercept=True)
    )
    regression_model.fit(X_train, y_train)

    # Score with training data:
    regression_train = regression_model.score(X_train, y_train)
    # Score with validation data:
    regression_valid = regression_model.score(X_valid, y_valid)

    print(OUTPUT_TEMPLATE.format(
        regression_train=regression_train,
        regression_valid=regression_valid,
    ))

    #r2_score
    y_predicted = regression_model.predict(X_valid)
    print('r2_score:', r2_score(y_valid, y_predicted))

OUTPUT_TEMPLATE = (
    '\n                       Train Score | Validation Score\n'
    'Linear Regression Model: {regression_train:.3g}         {regression_valid:.3g}\n'
)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 program.py <input_json_gz>')
        print('  e.g. python3 program.py plots.json.gz')
    else:
        plots_json_gz = sys.argv[1]
        main(plots_json_gz)