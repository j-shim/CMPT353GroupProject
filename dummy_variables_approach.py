#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import difflib

wikidata = pd.read_json("data/wikidata-movies.json.gz", lines=True)
rotten = pd.read_json("data/rotten-tomatoes.json.gz", lines=True)
imdb = pd.read_json("data/omdb-data.json.gz", lines=True)

#select neccessary columns and remove all rows containing any NaN values
wikidata = wikidata[['cast_member', 'director', 'imdb_id', 'rotten_tomatoes_id', 'wikidata_id']];
wikidata = wikidata.dropna(axis=0, how='any');

#select neccessary columns and remove all rows containing any NaN values
rotten = rotten[['audience_ratings', 'critic_average', 'critic_percent', 'rotten_tomatoes_id']];
rotten = rotten.dropna(axis=0, how='any');
rotten

#drop all unneccessary columns and remove all NaN rows
imdb = imdb[['imdb_id', 'omdb_genres', 'omdb_plot']];
imdb = imdb.dropna(axis=0, how='any');
imdb

#join all the data tables and remove NaN rows
wr = wikidata.set_index('rotten_tomatoes_id').join(rotten.set_index('rotten_tomatoes_id'))
wri = wr.set_index('imdb_id').join(imdb.set_index('imdb_id'));
wri = wri.dropna(axis=0, how='any');

# remove index columns and retain only the useful columns.
wri = wri.reset_index()
wri = wri.drop(['imdb_id', 'wikidata_id'], axis=1)

# ## What to do with categorical variables?
# 
# The general idea is to convert cast_member, director, and omdb_genres columns to meaningful variables before running them on machine learning algorithms.
# 
# Cast_member column will be converted to the following columns:
# count casts who appeared in n>=50 movies, ..in 49>n>=45 movies, ..in 45>n>=40 movies, etc.
# 
# Same goes for director column.
# 
# Omdb-genres will be converted to dummy variables (basically, add new column for each genre and the value for each column will be boolean. e.g. 'horror' column has 0 or 1 for each movie). Since adding too many features will result in overfitting, we should bad the low-occurence genres together into one column)
# (resource: https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512)

# define count function. count each item in a column and put the count information in a dictionary.
def countEach(listdata, counts):
    for item in listdata:
        counts[item] = counts.get(item, 0) + 1

countCol = np.vectorize(countEach)        

# print the dict keys and values, sorted by its values.
def printSortedDict(dic):
    for w in sorted(dic, key=dic.get, reverse=True):
        print (w + ": " + str(dic[w]))
        
# process the cast_member column
castCounts = {}
casts = np.array(wri['cast_member'])
countCol(casts, castCounts)

#convert this into bags
numRows = wri.shape[0]

cast_over50 = np.zeros([1,numRows])

wri['cast_over50'] = np.zeros([numRows,1])
wri['cast_45to49'] = np.zeros([numRows,1])
wri['cast_40to44'] = np.zeros([numRows,1])
wri['cast_35to39'] = np.zeros([numRows,1])
wri['cast_30to34'] = np.zeros([numRows,1])
wri['cast_25to29'] = np.zeros([numRows,1])
wri['cast_20to24'] = np.zeros([numRows,1])
wri['cast_15to19'] = np.zeros([numRows,1])
wri['cast_10to14'] = np.zeros([numRows,1])
wri['cast_5to9'] = np.zeros([numRows,1])
wri['cast_1to5'] = np.zeros([numRows,1])

firstRow = np.zeros((1,3))

for i in range(casts.size):
	newRow = np.zeros((1,3))
	for c in casts[i]:
		# if castCounts[c] >= 50:
		# 	newRow[0][0] += 1
		# elif castCounts[c] < 49 and castCounts[c] >= 40:
		#     newRow[0][1] += 1
		# elif castCounts[c] < 39 and castCounts[c] >= 30:
		#     newRow[0][2] += 1
		# elif castCounts[c] < 29 and castCounts[c] >= 20:
		#     newRow[0][3] += 1
		# elif castCounts[c] < 19 and castCounts[c] >= 10:
		#     newRow[0][4] += 1
		# elif castCounts[c] < 9 and castCounts[c] >= 1:
		#     newRow[0][5] += 1

		if castCounts[c] >= 40:
		    newRow[0][0] += 1
		elif castCounts[c] < 39 and castCounts[c] >= 20:
		    newRow[0][1] += 1
		elif castCounts[c] < 19 and castCounts[c] >= 1:
		    newRow[0][2] += 1
	firstRow = np.concatenate((firstRow, newRow), axis=0)

baggedCasts = np.delete(firstRow, (0), axis=0)
print (baggedCasts)


# process the director column
dirCounts = {}
directors = np.array(wri['director'])
countCol(directors, dirCounts)


