import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot


def findClusterSize(data): #also returns the value of the Silhouette Coefficient (I trust it more than elbow method)
	K = range(2,10)
	meandistortions = []
	silCoeffs = []

	for k in K:
		kmeans = KMeans(n_clusters=k)
		kmeans.fit(data)
		meandistortions.append(sum(np.min(cdist(data,kmeans.cluster_centers_,'euclidean'),axis=1))/data.shape[0])
		silCoeffs.append(metrics.silhouette_score(data,labels=kmeans.labels_,metric='euclidean'))

	print "\n\nMean Distortions"
	print "------------------"
	for i in K:
		print 'K:',i,'\t',meandistortions[i-2],'\t Silhouette Coeff: ',silCoeffs[i-2]

	kDistTuple = []
	print "\nDistortion Decline"
	print "-----------------------"
	for i in range(1,len(meandistortions)):
		difference = meandistortions[i-1]-meandistortions[i]
		kDistTuple.append([difference,i+1])
	for i in range(len(kDistTuple)-1):
		print 'K: %2d -> %2d %10.5f'%(kDistTuple[i][1],kDistTuple[i+1][1],kDistTuple[i][0])

	kDistTuple.sort()
	print "\n\nElbow Method Suggestion for Clusters: ",kDistTuple[0][1]

	kSilCoeff = zip(silCoeffs,K)
	kSilCoeff.sort()
	print "Best Silhouette Coeff and cluster size: ",kSilCoeff[-1]
	return kSilCoeff[-1]

data = pd.read_csv('../math-redacted.csv')
X = data[data.columns[6:]]

bestK = findClusterSize(X)

'''
#useful for encoding later
for i in range(len(X)):
	for j in range(len(X.iloc[i])):
		val = X.iloc[i][j]
		if val==-1:
			X.iloc[i][j]=2

encoded = OneHotEncoder(n_values=3).fit(X)
encoded = encoded.transform(X).toarray()
#SilCoeff = findClusterSize(encoded)
'''

kmeans = KMeans(n_clusters=2)#I put 2 because bestK somehow messed everything up?
kmeans.fit(X)

grads = data[data['Graduated']=='Y'].index
nograds = data[data['Graduated']=='N'].index

#gradsCluster = kmeans.labels_[grads]
gradsCluster = []
nogradsCluster = []
for i in range(len(kmeans.labels_)):
	if kmeans.labels_[i]==1:
		gradsCluster.append(i)
	else:
		nogradsCluster.append(i)

print "Grads Cluster:", len(gradsCluster)
print "True Grads in Grads Cluster:", len(grads.intersection(gradsCluster))
print "Not Grads Cluster:", len(nogradsCluster)
print "True Not Grads in Not Grads Cluster:", len(nograds.intersection(nogradsCluster))
#basically what Chris got with R


#DECISION TREE FOR MATH GRADUATING CLUSTER
dTree = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes = 50)
dTree.fit(X.iloc[gradsCluster],data[data.columns[2]].iloc[gradsCluster])
dot_data = StringIO()
tree.export_graphviz(dTree,out_file = dot_data,feature_names=data.columns[6:].values)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
if len(grads.intersection(gradsCluster))>30:
	graph.write_pdf('math-gradcluster-50-node-entropy-decision-tree.pdf')
else:
	graph.write_pdf('math-nogradcluster-50-node-entropy-decision-tree.pdf')


#DECISION TREE FOR MATH NONGRADUATING CLUSTER
dTree = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes = 50)
dTree.fit(X.iloc[nogradsCluster],data[data.columns[2]].iloc[nogradsCluster])
dot_data = StringIO()
tree.export_graphviz(dTree,out_file = dot_data,feature_names=data.columns[6:].values)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
if len(grads.intersection(gradsCluster))<30:
	graph.write_pdf('math-gradcluster-50-node-entropy-decision-tree.pdf')
else:
	graph.write_pdf('math-nogradcluster-50-node-entropy-decision-tree.pdf')

#=========================================
#=========================================
#=========================================
#=========================================
#=========================================
#============  PSYCH DATA   ==============
#=========================================
#=========================================
#=========================================
#=========================================
#=========================================

#note that this psych section of the code is more general-purpose in that
#I did not assign any one cluster as graduated vs not graduated
#Since there are only 2 clusters, this would still be a good idea but
#for other departments that are better split up in >2 clusters, this would
#be the general method to go about creating the trees.

data = pd.read_csv('../psych-redacted.csv')
X = data[data.columns[5:]]

print "Finding best cluster size for Psych data..."
#bestK = findClusterSize(X)
#not sure why the above line freaks out, but it said best silhouette was 2 and elbow was 7. I trust 2. Moving on...
bestK = 2
'''
#useful for encoding later
for i in range(len(X)):
	for j in range(len(X.iloc[i])):
		val = X.iloc[i][j]
		if val==-1:
			X.iloc[i][j]=2

encoded = OneHotEncoder(n_values=3).fit(X)
encoded = encoded.transform(X).toarray()
#SilCoeff = findClusterSize(encoded)
'''

print "Fitting KMeans with Psych data..."
kmeans = KMeans(n_clusters=bestK)
kmeans.fit(X)

totalClusters = []#list containing lists. Each list inside is the list of indices for a particular cluster. 
for i in range(bestK):#partition indices based on cluster labels
	cluster = []#particular cluster
	for j in range(len(kmeans.labels_)):
		if kmeans.labels_[j]==i:
			cluster.append(j)
	totalClusters.append(cluster)

clusterID = 0
for i in totalClusters:
	dTree = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes = 50)
	dTree.fit(X.iloc[i],data[data.columns[0]].iloc[i])
	dot_data = StringIO()
	tree.export_graphviz(dTree,out_file = dot_data,feature_names=data.columns[5:].values)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf('psych-'+str(clusterID)+'-cluster-50-node-entropy-decision-tree.pdf')
	clusterID +=1
