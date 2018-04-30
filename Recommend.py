# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 15:17:49 2018

@author: Aryaman Sriram
"""

import pandas as pd
import numpy as np
# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)



DS=lens.iloc[:,[6,9,10,11]]
ID=lens.iloc[:,1]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DS,ID , test_size = 1/3, random_state = 0)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le1 = preprocessing.LabelEncoder()
MF=le1.fit_transform(X_train.iloc[:,1])
le.fit(X_train.iloc[:,2])
OCC=le.transform(X_train.iloc[:,2])
print(MF)
ratings=X_train.iloc[:,0]
df=pd.DataFrame({'Ratings':ratings,'Sex':MF,'Occupation':OCC})
from sklearn.neighbors import KNeighborsClassifier
nbr=KNeighborsClassifier(n_neighbors=1)
nbr.fit(df,y_train)
pred=nbr.predict(df)
