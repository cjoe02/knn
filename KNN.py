# author: Caleb Joseph
# loading libraries
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# loading training data
df = pd.read_csv('breast-cancer-wisconsin.data') #reading data
df.replace('?', -99999, inplace=True)            #replace missing data with negative value
df.drop(['id'], 1, inplace=True)                 #drop id column

X = np.array(df.drop(['class'],1))   #features
y = np.array(df['class'])            #labels

#cross validation
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#classifier
k = 1;
clf = neighbors.KNeighborsClassifier(k)
clf.fit(X_train, y_train)

#accuracy score
accuracy = clf.score(X_test, y_test)
print("Accuracy: ",accuracy)

#sample data
s1 = [4,2,1,3,3,2,3,5,1] #should be 2
s2 = [4,5,4,3,4,5,3,2,1] #should be 2
s3 = [5,10,10,10,4,10,5,6,3] #should be 4
new_samples = np.array([s1, s2, s3])
new_samples = new_samples.reshape(len(new_samples),-1)

prediction = clf.predict(new_samples)
print("Prediction of inputs: ", prediction)

pred = clf.predict(X_test)
print(classification_report(y_test, pred))



