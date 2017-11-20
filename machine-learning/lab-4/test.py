from sklearn import cluster, preprocessing
import pandas as pd


X = pd.read_csv('X_test.csv', engine='python', sep=';', index_col=False).as_matrix()
#X = preprocessing.scale(X)

k = 5
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

from matplotlib import pyplot
import numpy as np

for i in range(k):
    # select only data observations with cluster label == i
    ds = X[np.where(labels==i)]
    # plot the data observations
    pyplot.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    pyplot.setp(lines,ms=15.0)
    pyplot.setp(lines,mew=2.0)
pyplot.show()