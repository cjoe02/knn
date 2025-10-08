# author: Caleb Joseph
# loading libraries
import pandas as pd
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap

# loading training data
df = pd.read_csv('breast-cancer-wisconsin2.data') #reading data
df.replace('?', -99999, inplace=True)             #replace missing data with negative value
df.drop(['id'], 1, inplace=True)                  #drop id column

X = np.array(df.drop(['class'],1))   #features
y = np.array(df['class'])            #labels

X = X[:, :2]   #picking first two features


#INSERT K HERE
k = 3;
clf = neighbors.KNeighborsClassifier(k)
clf.fit(X, y)

cmap_bg = ListedColormap(['#52eeff', '#ff8060'])
cmap_points = ListedColormap(['#22b2ff', '#9c3315'])

# Boundaries of graph
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
pl.figure()
pl.pcolormesh(xx, yy, Z, cmap=cmap_bg)

pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points)
pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.title("Benign or malignant breast cancer cells (k = %i)" % (k))
pl.xlabel('Clump Thickness')
pl.ylabel('Uniform Cell Size')
pl.show()