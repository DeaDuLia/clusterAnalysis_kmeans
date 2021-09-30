import pandas as pd
cust_df = pd.read_csv('zzz.csv')
df = cust_df.drop('region', axis = 1)
from sklearn.preprocessing import StandardScaler
import numpy as np
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
from sklearn.cluster import KMeans
clusterNum = 6
k_means = KMeans(init= "k-means++", n_clusters= clusterNum, n_init= 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)
df['Clus_km'] = labels
df.groupby('Clus_km').mean()
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 5], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Рождаемость', fontsize=18)
plt.ylabel('Доход', fontsize=16)
plt.show()

# 3D фигура, нет необходимости
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1, figsize=(8, 6))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
# ax.set_xlabel('')
# ax.set_ylabel('')
# ax.set_zlabel('')
#
# ax.scatter(X[:, 5], X[:, 0], X[:, 1], c= labels.astype(np.float))
# plt.show()
