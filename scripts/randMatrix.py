#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rand score and hierarchical clustering processing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as sch
from sklearn.metrics import adjusted_rand_score
import graphs as plot

def calc_cluster_characteristics(k, data, clusteringMethod, **kwargs): 
    """
    define a function that will give us the relevant output for the input k
    
    Args:
        k: epsilon value (for DBSCAN), resolution parameter (for phenograph), or cut height (for hierarchical clustering)
        data: UMAP embedding of data
        clusteringMethod: 'DBSCAN' or 'HC'
    """
    
    print('Parameter value:', k)    
    results = pd.Series(dtype = 'float64')
    results['k'] = k    
    sampleMeta = kwargs.get('sampleMeta', None)
    
    if clusteringMethod == 'DBSCAN':
        results['communities'] = DBSCAN(eps=k, min_samples=2).fit(data).labels_
    
    elif clusteringMethod == 'HC':
        Y_cluster = sch.linkage(data.values.T, metric='correlation') 
        results['communities'], _ = labelHCLandrace(data, sampleMeta, Y_cluster, cutHeight = k, newVariety = False)
        
    else:
        print('invalid clustering method')
        
    return results

def randScoreMatrix(data, ks, clusteringMethod, **kwargs):
    """
    Matrix of adjusted rand scores for a range of parameter values

    Args:
        data: UMAP embedding of data
        ks: list of parameter values to scan over
        clusteringMethod: 'DBSCAN', 'phenograph', or 'HC'        
    """
    sampleMeta = kwargs.get('sampleMeta', None)
    admixedCutoff = kwargs.get('admixedCutoff', None)

    if sampleMeta is None:
        cluster_chars_list = [calc_cluster_characteristics(k, data, clusteringMethod, admixedCutoff = admixedCutoff) for k in ks]

    else:
        cluster_chars_list = [calc_cluster_characteristics(k, data, clusteringMethod, sampleMeta = sampleMeta) for k in ks]

    cluster_characteristics = pd.concat(cluster_chars_list, axis = 1).transpose()
    
    #calculate rand score
    n = len(ks)
    Rand_indices = pd.DataFrame(np.zeros((n,n)), index = ks, columns = ks)
    
    Rand_indices.index.name = 'k1'
    Rand_indices.columns.name = 'k2'
    
    for i in range(n):
        for j in range(n):
            Rand_indices.iloc[i, j] = adjusted_rand_score(
                cluster_characteristics['communities'][i], 
                cluster_characteristics['communities'][j]
            )
                   
    #plot
    plt.figure(figsize = (8,8))
    plt.imshow(Rand_indices, aspect='auto', interpolation='none', cmap='coolwarm',vmin=0,vmax = 1)
    plt.gca().invert_yaxis()
    plt.xticks(np.arange(len(ks)),ks)
    plt.yticks(np.arange(len(ks)),ks)
    plt.title('Adjusted Rand Score')
    plt.colorbar()
    if clusteringMethod == 'DBSCAN':
        plt.xlabel('epsilon 1')
        plt.ylabel('epsilon 2')
    if clusteringMethod == 'HC':
        plt.xlabel('cut height 1')
        plt.ylabel('cut height 2')   
    plt.tight_layout()
    
def labelHCLandrace(clusterSubset, sampleMeta, Y_cluster, cutHeight, clusterNumber = 0, admixedCutoff = None, newVariety = True):
    """
    Take a dendrogram and label samples that match a reference variety, otherwise label as admixed or a new variety 

    Args:
        clusterSubset: processed SNP proportion data subset to a single DBSCAN cluster
        Y_cluster: dendrogram for clusterSubset
        clusterNumber: DBSCAN cluster number
        cutHeight: cutoff value for cutting a dendrogram into clusters 
        sampleMeta: metadata paired with genotyping data
        admixedCutoff: divergence cutoff to differentiate between admixtures and non reference varieties
    """
    references = sampleMeta[(sampleMeta['reference'].notna())]
    subClusterNumber = np.zeros(len(Y_cluster)+1) #zero is admixed
    subClusterNames = [['Admixed']]
    counter = 0
    
    subClusters = sch.cut_tree(Y_cluster, height = cutHeight)
    for cluster in np.unique(subClusters):
        sampleIndex = np.where(subClusters == cluster)[0]
        shortName = clusterSubset.columns[sampleIndex][np.isin(clusterSubset.columns[sampleIndex], references['short_name'].values.astype('str'))]
        refInSubCluster = np.unique(sampleMeta[sampleMeta['short_name'].isin(shortName.astype('int'))]['reference'])
        
        if (len(refInSubCluster) > 0): #if there's a reference in the cluster 
            if (len(refInSubCluster) > 1): #if multiple references, flatten       
                refInSubCluster = np.asarray(['+'.join(refInSubCluster)])
            if not np.isin(refInSubCluster, subClusterNames): #add if it's a new reference
                subClusterNames.append(list(refInSubCluster))            
                
            subClusterNumber[sampleIndex.tolist()] = np.where(subClusterNames == refInSubCluster)[0][0]
          
        elif newVariety: #add new non reference varieties            
            if not admixedCutoff: #no admixedCutoff
                subClusterNames.append(['Genetic entity-'+str(clusterNumber)+'-'+str(counter)]) 
                subClusterNumber[sampleIndex.tolist()] = (len(subClusterNames)-1)
                counter += 1  
            
            elif min(plot.homozygousDivergence(clusterSubset.values[:,sampleIndex])) < admixedCutoff: #at least one sample must be less than admixedCutoff         
                subClusterNames.append(['Genetic entity-'+str(clusterNumber)+'-'+str(counter)]) 
                subClusterNumber[sampleIndex.tolist()] = (len(subClusterNames)-1)
                counter += 1
                
    return subClusterNumber, subClusterNames        


def cutoffQuality(clusterSubset, sampleMeta, Y_cluster):
    """
    Evaluate cut height values based on the number of varieties/cluster and technical replicates split up

    Args:
        clusterSubset: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        Y_cluster: dendrogram for clusterSubset
    """
    references = sampleMeta[(sampleMeta['reference'].notna())]        
    refRows = np.arange(len(clusterSubset.columns))[np.isin(clusterSubset.columns, references['short_name'].values.astype('str'))]
    
    varNames = []
    for i in clusterSubset.columns[refRows]:
        varNames.append(references[references['short_name'] == int(i)]['reference'].values[0])
    varNames = np.asarray(varNames)
            
    cutList = np.linspace(min(Y_cluster[:,2]),max(Y_cluster[:,2]),50)
    clusters = sch.cut_tree(Y_cluster, height=cutList)
    df = pd.DataFrame(clusters)
        
    repTogether = np.zeros(len(cutList)) #number of technical replicates in same cluster    
    test = pd.DataFrame(columns=df.columns, index=np.arange(len(np.unique(varNames)))) #number of clusters that contain more than one variety
    
    for i, var in enumerate(np.unique(varNames)):
        rows = refRows[np.where(varNames == var)]
        repTogether += pd.Series({c: len(df.iloc[rows][c].unique()) for c in df}).values == 1
        test.iloc[i] = pd.Series(df.iloc[rows][c].unique() for c in df.columns) #don't count the same variety multiple times

    avgVarInCluster = []
    for c in test.columns:
        _, counts = np.unique(np.concatenate(test[c].values), return_counts=True)
        avgVarInCluster.append(np.average(counts))

    plt.figure()
    plt.plot(cutList, repTogether, '-o', markersize = 2, label = 'Technical replicates together')  
    plt.plot(cutList, avgVarInCluster, '-^', markersize = 2, label = 'Avg varieties/cluster')
    plt.legend(loc = 'upper left')
    plt.xlabel('Cut height')
    plt.ylabel('Number of varieties')
    plt.tight_layout()
    
    return repTogether, np.asarray(avgVarInCluster), len(np.unique(varNames)), cutList