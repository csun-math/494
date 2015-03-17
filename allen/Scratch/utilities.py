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
    

def getPCADataFrame(columns, pca, n_components=2):
    
    pcaDF=pd.DataFrame({'class':columns,
                          'PC1':pca.components_[0],
                          'PC2':pca.components_[1]})  
    pcaDF=pcaDF[['class','PC1','PC2']]
    for i in range(2,n_components):
        pcaDF['PC'+str(i)]=pca.components_[i]
    return pcaDF

def plotPCA(X_raw,cluster_list,pca=None, k=2,norm=False,colors='rbgmyck',markers='xDo+v*s.'):
    
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

    #colors='rbgmyck'   
    #markers='xD.+ov*s'
    for i in range(min(k, len(colors))):
        plt.scatter(points[i][0], points[i][1], c=colors[i],\
                    marker=markers[i], label=str(i)+' Cluster')
        
    plt.legend()
    plt.show()
    
    
def getDistancesDF(columns, cluster_centers,k=2):
    distancesDF=pd.DataFrame({'class':columns,
                          'Cluster0':cluster_centers[0],
                          'Cluster1':cluster_centers[1]})  
    distancesDF=distancesDF[['class','Cluster0','Cluster1']]
    for i in range(2,k):
        distancesDF['Cluster'+str(i)]=cluster_centers[i]
    return distancesDF
                          

