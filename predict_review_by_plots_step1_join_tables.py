import numpy as np
import pandas as pd
import sys

def main(in_directory, output_json):
    odata = pd.read_json(in_directory + '/omdb-data.json.gz', orient='records', lines=True)
    rdata = pd.read_json(in_directory + '/rotten-tomatoes.json.gz', orient='records', lines=True)
    wdata = pd.read_json(in_directory + '/wikidata-movies.json.gz', orient='records', lines=True)

    # Join wdata and odata by: first set 'imdb_id' as index then join on 'imdb_id'
    # We need column 'omdb_plot' from odata
    data = wdata.join(odata.set_index('imdb_id'), on='imdb_id')

    # Drop 'rotten_tomatoes_id' column to avoid conflict on joining
    # (The above join already contains column named 'rotten_tomatoes_id')
    rdata = rdata.drop(['rotten_tomatoes_id'], axis=1)

    # Join data and rdata similar to above:
    # We need 'audience_average' and 'critic_average' from rdata
    data = data.join(rdata.set_index('imdb_id'), on='imdb_id')

    # Select only the columns needed (plot => audience_average)
    data = data[['imdb_id', 'enwiki_title', 'omdb_plot', 'audience_average', 'critic_average']]

    data.to_json(output_json, orient='records', lines=True)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 program.py <input_directory> <output_json>')
        print('  e.g. python3 program.py data plots.json')
    else:
        in_directory = sys.argv[1]
        output_json = sys.argv[2]
        main(in_directory, output_json)
