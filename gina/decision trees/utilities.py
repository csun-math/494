
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def printClusters( df, cluster_list,k=2):

    categories=['non-graduates','graduates']
    gradClustersDF=pd.DataFrame({'Graduated':df['Graduated'], 'ClusterLbl':cluster_list})
    for i in range(2):
        print 'total',categories[i], len(gradClustersDF[gradClustersDF['Graduated']==i])
        for j in range(k):
            print j,'cluster',\
                gradClustersDF[(gradClustersDF['Graduated']==i)&(gradClustersDF['ClusterLbl']==j)]['ClusterLbl'].count()
    return gradClustersDF
    

#kmod.cluster(X_raw,pre_runs=5)

def plotPCA(X_raw,cluster_list,k=2,norm=False):
    pca = PCA(n_components=2)
    #X_raw=df[df.columns[6:]]
    if norm==True:
        normalizer=sklearn.preprocessing.Normalizer()
        X_raw_norm=normalizer.fit_transform(X_raw.transpose().values.astype('float64')).transpose()
    else:    
        X_raw_norm= (X_raw - X_raw.mean()) # / (X_raw.max() - X_raw.min())
        
    reduced_X = pca.fit_transform(X_raw_norm)
    
    points=[[[],[]] for j in range(k)]
   
   
    for i in range(len(reduced_X)):
        for j in range(k):
            if cluster_list[i]==j: #red
                points[j][0].append(reduced_X[i][0])
                points[j][1].append(reduced_X[i][1])

    colors='rbgmyck'   
    markers='xD.+ov*s'
    for i in range(min(k, len(colors))):
        plt.scatter(points[i][0], points[i][1], c=colors[i],marker=markers[i], label=str(i)+' Cluster')
        
    plt.legend()
    plt.show()
    
    
def getDistancesDF(columns, kmeans_model,k=2):
    distancesDF=pd.DataFrame({'class':columns,
                          'Cluster0':kmeans_model.cluster_centers_[0],
                          'Cluster1':kmeans_model.cluster_centers_[1]})
    distancesDF=distancesDF[['class','Cluster0','Cluster1']]
    for i in range(2,k):
        distancesDF['Cluster'+str(i)]=kmeans_model.cluster_centers_[i]
    return distancesDF
                          

#no need to use it
def clusterCenters(X, k=2,n_tests=10,init='random'):
    
    cluster_centers=[]
    #X = data[data.columns[6:]]
    for i in range(n_tests):
        cluster_centers.append(KMeans(n_clusters=2,init=init).fit(X).cluster_centers_)
    avg_center=sum(cluster_centers)/float(k)
    
        
    return avg_center
    