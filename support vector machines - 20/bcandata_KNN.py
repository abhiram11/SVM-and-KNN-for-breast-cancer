#from KNN

import numpy as np 
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd 

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
#txt file was changed by adding a header column manually

df.replace('?',-99999, inplace=True) #replace null values by -99999 which makes
#it an outlier
df.drop(['id'], 1, inplace=True) #removing ID significantly increases accuracy

X = np.array(df.drop(['class'],1)) #creates features
y = np.array(df['class']) #creates labels

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)