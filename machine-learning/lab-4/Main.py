import pandas as pd
from sklearn.cluster import DBSCAN

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from sklearn import metrics


def compare(pred_y, y_test):
    count = 0
    n = len(pred_y)
    for i in range(n):
        if int(pred_y[i]) == int(y_test[i]):
            count += 1

    print("{}/{} {}%".format(count, n, (count * 100 / n)))

#Считываем данные
X = pd.read_csv('X_test.csv', engine='python', sep=';', index_col=False).as_matrix()
y = pd.read_csv('y_test.csv', engine='python', sep=';', index_col=False)

#K-Means
y_pred = KMeans(n_clusters=6, init='k-means++').fit_predict(X)
y = y.iloc[:, 0].tolist()
y_pred = y_pred.tolist()
print("KMeans")
compare(y_pred, y)
print("adjusted_rand_score:", metrics.adjusted_rand_score(y, y_pred), "\n")


#Mini K-Means
batch_size = 45
mbk = MiniBatchKMeans(init='k-means++', n_clusters=6, batch_size=batch_size, n_init=10, max_no_improvement=10, verbose=0)
y_pred = mbk.fit_predict(X)
y_pred = y_pred.tolist()
print("MiniBatchKMeans")
compare(y_pred, y)
print("adjusted_rand_score:", metrics.adjusted_rand_score(y, y_pred), "\n")


#DBSCAN
'''В таких плотных скоплениях не будет работать, нужно использовать что-то другое'''
y_pred = DBSCAN().fit_predict(X)
y_pred = y_pred.tolist()
print("DBSCAN")
compare(y_pred, y)
print("adjusted_rand_score:", metrics.adjusted_rand_score(y, y_pred), "\n")


#AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering

print("AgglomerativeClustering")
for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=6)
    y_pred = clustering.fit_predict(X)
    y_pred = y_pred.tolist()
    print(linkage)
    print("adjusted_rand_score:", metrics.adjusted_rand_score(y, y_pred), "\n")