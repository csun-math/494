import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

'''
def printClusters( df, cluster_list,k=2):

    categories=['non-graduates','graduates']
    gradClustersDF=pd.DataFrame({'Graduated':df['Graduated'], 'ClusterLbl':cluster_list})
    for i in range(2):
        print 'total',categories[i], len(gradClustersDF[gradClustersDF['Graduated']==i])
        for j in range(k):
            print j,'cluster',\
                gradClustersDF[(gradClustersDF['Graduated']==i)&(gradClustersDF['ClusterLbl']==j)]['ClusterLbl'].count()
    return gradClustersDF
'''
    
def getGradsClusterDF(df, cluster_list):
    return pd.DataFrame({'Graduated':df['Graduated'],'ClusterLbl':cluster_list})
#x =getGradsClusterDF(data, kmeans_model.labels_)
#x.head()

def printContingencyTable(y,ypred,labels):
    confusion_matrix = metrics.confusion_matrix(y, ypred)
    plt.matshow(confusion_matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print pd.crosstab(y,ypred,rownames=[labels[0]],colnames=[labels[1]])

gradsCluster = getGradsClusterDF(data,kmeans_model.labels_)
printContingencyTable(gradsCluster['Graduated'],gradsCluster['ClusterLbl'],['Graduated','ClusterLbl'])


    
'''
def getPCADataFrame(columns, pca, n_components=2):
    
    pcaDF=pd.DataFrame({'class':columns,
                          'PC1':pca.components_[0],
                          'PC2':pca.components_[1]})  
    pcaDF=pcaDF[['class','PC1','PC2']]
    for i in range(2,n_components):
        pcaDF['PC'+str(i)]=pca.components_[i]
    return pcaDF
'''
def getPCADataFrame(columns, pca, n_components=2):
    
    pcaDF=pd.DataFrame({'class':columns})
    
    for i in range(1,n_components+1):#add principal component columns
        pcaDF['PC'+str(i)]=pca.components_[i-1]
    
    return pcaDF

#########################################################
############     refactored up to here       ############
#########################################################

#def determineK #where k is the best number of clusters based on our choice of metric


def plotPCA(X_raw,cluster_list,pca=None, k=2,norm=False, centered=False,colors='rbgmyck',markers='xDo+v*s.'):
    
    #X_raw=df[df.columns[6:]]
    if norm:
        normalizer=sklearn.preprocessing.Normalizer()
        X_raw_norm=normalizer.fit_transform(X_raw.transpose().values.astype('float64')).transpose()
    elif centered:    
        X_raw_norm= (X_raw - X_raw.mean()) # / (X_raw.max() - X_raw.min())
    else:
       X_raw_norm= X_raw 
        
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
                          
def CH_index(X, labels, centroids):
    
    '''
    slightly changed the code in:
    https://github.com/scampion/scikit-learn/blob/master/scikits/learn/cluster/__init__.py
    change is in the original line:
    B = np.sum([ (c - mean)**2 for i,c in enumerate(centroids)])
    
    
    The pseudo F statistic :
    pseudo F = [( [(T - PG)/(G - 1)])/( [(PG)/(n - G)])] 
    The pseudo F statistic was suggested by Calinski and Harabasz (1974)
    Calinski, T. and J. Harabasz. 1974. 
    A dendrite method for cluster analysis. Commun. Stat. 3: 1-27.
    http://dx.doi.org/10.1080/03610927408827101
    '''
    mean = np.mean(X,axis=0) 
    
    B = np.sum([ np.sum(labels==i)*(c - mean)**2 for i,c in enumerate(centroids)])
    W = np.sum([ (x-centroids[labels[i]])**2 
                        for i, x in enumerate(X)])
    c = len(centroids)
    n = len(X)
    return ((n-c)*B )/((c-1)*W)
