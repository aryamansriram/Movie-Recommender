# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 15:17:49 2018

@author: Aryaman Sriram
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names=u_cols,
                    encoding='latin-1')
print("Users: ")
print(users.head())
print("Unique occupations: ", users.occupation.nunique())

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=r_cols,
                      encoding='latin-1')
print("Ratings: ")
print(ratings.head())

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')
print("Movies: ")
print(movies.head())

# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)
print("Lens: ")
print(lens.head())

print("X:")
print(lens.iloc[0, [6, 9, 10, 11]])
print("Y:")
print(lens.iloc[0, 1])

DS=lens.iloc[:,[6,9,10,11]]
ID=lens.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(DS,ID , test_size = 1/3, random_state = 0)

le = preprocessing.LabelEncoder()
le1 = preprocessing.LabelEncoder()

print("X_train:")
print(X_train.iloc[0, :])

print("y_train:")
print(y_train.head())

MF=le1.fit_transform(X_train.iloc[:,1])
le.fit(X_train.iloc[:,2])
print(MF)

OCC=le.transform(X_train.iloc[:,2])
print(OCC)

ratings=X_train.iloc[:,0]
print("Ratings: ")
print(ratings.head())

df=pd.DataFrame({'Ratings':ratings,'Sex':MF,'Occupation':OCC})
print(df.head())

nbr=KNeighborsClassifier(n_neighbors=1)
nbr.fit(df,y_train)
pred=nbr.predict(df)

print(pred[:5])
