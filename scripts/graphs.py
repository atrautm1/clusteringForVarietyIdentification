#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a variety of figures
"""

import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import pandas as pd

def clusterReorder(subset, counts):
    """
    Sort genes and samples using heirarchical clustering
    
    Args:
        subset: processed SNP proportion data (values only)
        counts: list of the number of samples per DBSCAN cluster
    """
    #sort genes using heirarchical clustering
    Y_gene = sch.linkage(subset, metric='euclidean')
    Z_gene = sch.leaves_list(Y_gene)
    subsetReorder = subset[Z_gene,:]

    #sort samples within DBSCAN clusters using heirarchical clustering
    breakPoints = [0]+list(np.cumsum(counts))
    clusterOrder = []
    for i, j in enumerate(counts):
        if j > 1:
            clusterSubset = subset[:,breakPoints[i]:breakPoints[i+1]]
            Y_cluster = sch.linkage(clusterSubset.T, metric='correlation')
            Z_cluster = sch.leaves_list(Y_cluster)
            if i != 0: 
                Z_cluster += breakPoints[i]
            clusterOrder += list(Z_cluster)
        else:
            clusterOrder += [breakPoints[i]]         

    return subsetReorder[:,clusterOrder], clusterOrder, breakPoints

def homozygousDivergence(x):
    """
    Calculate the divergence for a numpy array of processed SNP proportion data
    """
    _, total = np.unique(np.where((x < 0.2) | (x > 0.8))[1], return_counts=True)
    highDivergence = np.nansum(1 - np.where(x > 0.8, x, np.nan), axis = 0)
    lowDivergence = np.nansum(np.where(x < 0.2, x, np.nan), axis = 0)
    return (lowDivergence + highDivergence)/ total

def plotTemplate():
    """
    Basic layout for a figure with one plot and a colorbar
    """
    fig = plt.figure(figsize=(7.2,6))
    gs=GridSpec(1,2, width_ratios=[6,0.2], figure = fig)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    return ax1, ax2

def plotDouble():
    """
    Basic layout for a figure with two plots
    """
    fig = plt.figure(figsize=(10,4))
    gs = GridSpec(1, 2, width_ratios=[1,1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    return ax1, ax2

def umapCluster(embedding, communities):
    """
    UMAP with samples colored by cluster
    
    Args:
        embedding: UMAP embedding of data 
        communities: DBSCAN cluster number for each sample
    """
    ax1, ax2 = plotTemplate()
    tab20 = mpl.colormaps['tab20'].resampled(max(communities)+1)
    SC=ax1.scatter(embedding[:, 0], embedding[:, 1], s = 1, c=communities, cmap = tab20)
    
    #add cluster labels
    for i in np.unique(communities):
        samples = np.where(communities == i)
        ax1.text(np.mean(embedding[samples,0]), np.mean(embedding[samples,1]), i, fontsize=12)
    
    #add colorbar
    cbar = plt.colorbar(SC, cax=ax2)
    boxWidth = max(communities)/(max(communities)+1)
    cbar.set_ticks((np.linspace(0,max(communities),max(communities)+2)[:-1]+boxWidth/2), labels = np.arange(max(communities)+1).astype('str'))
    plt.tight_layout()

def umapReference(snpProportion, embedding, sampleMeta, communities):
    """
    UMAP with references colored by cluster 
    
    Args:
        snpProportion: processed SNP proportion data
        embedding: UMAP embedding of data 
        sampleMeta: metadata paired with genotyping data
        communities: DBSCAN cluster number for each sample
    """
    references = sampleMeta[(sampleMeta['reference'].notna())]
    referencesIndex = np.where(np.isin(snpProportion.columns, references['short_name'].astype('str')))[0]
    refClusters, refCounts = np.unique(communities[referencesIndex], return_counts=True)
    
    ax1, ax2 = plotTemplate()
    tab20 = mpl.colormaps['tab20'].resampled(len(refClusters))
    ax1.scatter(embedding[:, 0], embedding[:, 1], s = 1, c='grey', alpha = 0.1)
    labels = []
    bot = 0
    
    for i, j in enumerate(refClusters): #all the ref in a cluster should be the same color
        index = np.intersect1d(np.where(communities == j), referencesIndex)
        ax1.scatter(embedding[index, 0], embedding[index, 1], s = 3, color=tab20(i))
        
        #add cluster labels
        samples = np.where(communities == j)
        ax1.text(np.mean(embedding[samples,0]), np.mean(embedding[samples,1]), j, fontsize=12)
    
        #add legend
        shortName = snpProportion.columns[index].astype('int')
        variety = sampleMeta[sampleMeta['short_name'].isin(shortName)]['reference']    
        labels += np.unique(variety).tolist()
        ax2.bar(0,len(np.unique(variety)),bottom = bot, color = tab20(i))
        bot += len(np.unique(variety))  
            
    ax2.yaxis.tick_right()    
    ax2.set_yticks(np.arange(len(labels))+0.5,labels, fontsize=5) #need to offset labels by box width
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylim((0,bot))
    plt.tight_layout()
    
def umapVariety(embedding, varList, communities):
    """
    UMAP with samples colored by variety
    
    Args:
        embedding: UMAP embedding of data 
        varList: list of varieties for each sample in embedding 
        communities: DBSCAN cluster number for each sample
    """
    varieties = np.unique(varList)

    ax1, ax2 = plotTemplate()
    tab20 = mpl.colormaps['tab20'].resampled(len(varieties))
    SC=ax1.scatter(embedding[:, 0], embedding[:, 1], s = 1, c=varList, cmap = tab20)
    
    #add cluster labels
    for i in np.unique(communities):
        samples = np.where(communities == i)
        ax1.text(np.mean(embedding[samples,0]), np.mean(embedding[samples,1]), i, fontsize=12)
    
    #add colorbar
    cbar = plt.colorbar(SC, cax=ax2)
    boxWidth = max(communities)/(max(communities)+1)
    cbar.set_ticks((np.linspace(0,max(communities),max(communities)+2)[:-1]+boxWidth/2), labels = np.arange(max(communities)+1).astype('str'))
    plt.tight_layout()

def heatmapSingleCluster(snpProportion, sampleMeta, communities, COI):
    """
    Heatmap of SNP proportions for samples in a single cluster

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        communities: DBSCAN cluster number for each sample
        COI: DBSCAN cluster to plot
    """

    subset = snpProportion[snpProportion.columns[np.where(communities == COI)[0]]].values
    subsetReorder, clusterOrder, breakPoints = clusterReorder(subset, [subset.shape[1]])
    
    references = sampleMeta[(sampleMeta['reference'].notna())]
    referencesIndex = np.where(np.isin(snpProportion.columns, references['short_name'].astype('str')))[0]
    
    ax1, ax2 = plotTemplate()
    SC=ax1.imshow(subsetReorder, aspect='auto', interpolation='none', cmap='coolwarm',vmin=0,vmax = 1)
    ax1.set_yticks([])
    plt.colorbar(SC, cax=ax2)
    ax1.set_title('Cluster '+str(COI))
    
    #add reference labels to x-axis
    numInRef, numInCluster, numInZ = np.intersect1d(referencesIndex, np.where(communities == COI)[0], return_indices=True) #number of the reference
    if len(numInRef) > 0:
        ticks = []
        labels = []
        for i in range(len(numInRef)):
            ticks.append(np.where(clusterOrder == numInZ[i])[0][0])
            labels.append(sampleMeta[sampleMeta['short_name'] == int(snpProportion.columns[numInRef[i]])]['reference'].values[0])      
        ax1.set_xticks(ticks, labels, rotation = 90)
        
    else:
        ax1.set_xticks([])
    
    plt.tight_layout()
 
def heatmapManyClusters(snpProportion, sampleMeta, communities, allCOI, tickType ='blank'):
    """
    Heatmap of SNP proportions for samples with a subplot for each cluster 

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        communities: DBSCAN cluster number for each sample
        allCOI: list of DBSCAN cluster to plot
        tickType: 'blank' (no labels), 'references' (label only one sample for each reference), 'referencesAll' (label every reference sample), 'sampleRef' (label sample names and references), 'sampleNames' (label sample names), 'divergence' (label divergence score), 'sampleDivergence' (label sample names and divergence)
    """

    references = sampleMeta[(sampleMeta['reference'].notna())]
    referencesIndex = np.where(np.isin(snpProportion.columns, references['short_name'].astype('str')))[0]
    
    #use the same gene order for all subplots
    subsetAllCOI = snpProportion[snpProportion.columns[np.where(np.isin(communities, allCOI))]] 
    Y_gene = sch.linkage(subsetAllCOI, metric='euclidean')
    Z_gene = sch.leaves_list(Y_gene)
    
    fig = plt.figure(figsize=(15,6))
    gs=GridSpec(1,len(allCOI)+1, width_ratios=[1]*len(allCOI)+[0.05], figure = fig)
    
    for index in range(len(allCOI)):
        subplot = plt.subplot(gs[index])
        
        clusterIndex = np.where(communities == allCOI[index])[0]
        clusterSubset = snpProportion[snpProportion.columns[clusterIndex]].values
        clusterSubset = clusterSubset[Z_gene,:]
    
        #cluster samples
        Y_cluster = sch.linkage(clusterSubset.T, metric='correlation')
        Z_cluster = sch.leaves_list(Y_cluster)
    
        SC = subplot.imshow(clusterSubset[:,Z_cluster], aspect='auto', interpolation='none', cmap='coolwarm',vmin=0,vmax = 1)
        subplot.set_yticks([])
        subplot.set_title('Cluster '+str(allCOI[index]))
        
        if tickType == 'references': #add reference labels to x-ticks (one reference per variety)
            numInRef, numInCluster, numInZ = np.intersect1d(referencesIndex, clusterIndex, return_indices=True) #number of the reference
            if len(numInRef) > 0:
                ticks = []
                labels = []
                for i in range(len(numInRef)):
                    ticks.append(np.where(Z_cluster == numInZ[i])[0][0])
                    labels.append(sampleMeta[sampleMeta['short_name'] == int(snpProportion.columns[numInRef[i]])]['reference'].values[0])      
                                
                uniqueLabel, uniqueIndex = np.unique(labels, return_index=True)
                subplot.set_xticks(np.asarray(ticks)[uniqueIndex], np.asarray(labels)[uniqueIndex], rotation = 90)
            else:
                subplot.set_xticks([])
        
        if tickType == 'referencesAll': #add reference labels to x-ticks (all references)
            numInRef, numInCluster, numInZ = np.intersect1d(referencesIndex, clusterIndex, return_indices=True) #number of the reference
            if len(numInRef) > 0:
                ticks = []
                labels = []
                for i in range(len(numInRef)):
                    ticks.append(np.where(Z_cluster == numInZ[i])[0][0])
                    labels.append(sampleMeta[sampleMeta['short_name'] == int(snpProportion.columns[numInRef[i]])]['reference'].values[0])      
                
                subplot.set_xticks(ticks, labels, rotation = 90) 
                
            else:
                subplot.set_xticks([])

                
        if tickType == 'sampleRef': #add sample names and reference labels to x-ticks
            labels = np.copy(snpProportion.columns[clusterIndex[Z_cluster]].values)
            numInRef, numInCluster, numInZ = np.intersect1d(referencesIndex, clusterIndex, return_indices=True) #number of the reference
            if len(numInRef) > 0:
                for i in range(len(numInRef)):
                    labels[np.where(Z_cluster == numInZ[i])[0][0]] = (sampleMeta[sampleMeta['short_name'] == int(snpProportion.columns[numInRef[i]])]['reference'].values[0])      

            subplot.set_xticks(np.arange(len(clusterIndex)),labels, rotation = 90)
                
        if tickType == 'sampleNames': #add sample names to x-ticks
            subplot.set_xticks(np.arange(len(clusterIndex)),snpProportion.columns[clusterIndex[Z_cluster]], rotation = 90)

        if tickType == 'divergence': #add sample divergence to x-ticks
            divergence = homozygousDivergence(clusterSubset[:,Z_cluster])
            subplot.set_xticks(np.arange(len(divergence)), np.around(divergence,2), rotation = 90)
            
        if tickType == 'sampleDivergence': #add sample names and divergence to x-ticks
            sampleName = snpProportion.columns[clusterIndex[Z_cluster]]
            divergence = np.around(homozygousDivergence(clusterSubset[:,Z_cluster]),2)
            labels = []
            for i in range(len(sampleName)): labels.append(sampleName[i]+'-'+str(divergence[i]))
            
            subplot.set_xticks(np.arange(len(divergence)), labels, rotation = 90)

        if tickType == 'blank': #no labels for x-ticks
            subplot.set_xticks([])

    plt.colorbar(SC, cax=plt.subplot(gs[len(allCOI)]))
    plt.tight_layout()

def heatmapReferences(snpProportion, sampleMeta, allVarieties, tick_type):
    """
    Heatmap of reference samples from a list of varieties

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        allVarieties: list of variety names
        tickType: 'inventory' (inventory number), 'short_name' (sample number), 'divergence' (label divergence score), 'source' (sample source), 'references' (reference name)
    """
    
    refShort = sampleMeta[
        (np.isin(sampleMeta['reference_original'],allVarieties) if 'reference_original' in sampleMeta else np.array([False])) | 
        np.isin(sampleMeta['reference'],allVarieties)
    ]['short_name'].values.astype('str')

    refShort = refShort[np.isin(refShort.astype(str), snpProportion.columns)]
    subset = snpProportion[refShort].values
    subsetReorder, clusterOrder, breakPoints = clusterReorder(subset, [subset.shape[1]])
        
    ax1, ax2 = plotTemplate()
    SC=ax1.imshow(subsetReorder, aspect='auto', interpolation='none', cmap='coolwarm',vmin=0,vmax = 1)
    ax1.set_yticks([])
    plt.colorbar(SC, cax=ax2)
    ax1.set_title(allVarieties)
    
    sampleOrder = refShort[clusterOrder]
    labels = []

    if tick_type == 'inventory': #add inventory number to x-ticks
        for sample in sampleOrder:
            labels.append(sampleMeta[sampleMeta['short_name'] == int(sample)]['inventory'].values[0])

    if tick_type == 'short_name': #add short_name to x-ticks
        labels = sampleOrder
    
    if tick_type == 'source': #add sample source to x-ticks
        for sample in sampleOrder:
            row = sampleMeta[sampleMeta['short_name'] == int(sample)]
            labels.append(row['reference'].values[0]+', '+row['seedSource'].values[0])

    if tick_type == 'references': #add reference labels to x-ticks (all references)
        for sample in sampleOrder:
            labels.append(sampleMeta[sampleMeta['short_name'] == int(sample)]['reference'].values[0])
            
    if tick_type == 'divergence': #add sample divergence to x-ticks
        labels = np.around(homozygousDivergence(subsetReorder),2)


    ax1.set_xticks(np.arange(len(labels)), labels, rotation = 90)

    plt.tight_layout()

def histogramMissingness(snpProportionNoInterpolation): 
    """
    Histograms of missingness by sample and marker

    Args:
        snpProportionNoInterpolation: SNP proportion data before interpolation
    """
    ax1, ax2 = plotDouble()
    ax1.hist(snpProportionNoInterpolation.isna().sum(axis = 0), bins = np.linspace(0,snpProportionNoInterpolation.shape[0],21)) #samples missing values from X genes
    ax1.set_xticks(np.linspace(0,snpProportionNoInterpolation.shape[0],21), np.arange(0,105,5))
    ax1.set_title('Missingness per sample')
    ax1.set_xlabel('Percent Missingness')
    ax2.hist(snpProportionNoInterpolation.isna().sum(axis = 1), bins = np.linspace(0,snpProportionNoInterpolation.shape[1],21)) #gene missing from X samples
    ax2.set_xticks(np.linspace(0,snpProportionNoInterpolation.shape[1],21), np.arange(0,105,5))
    ax2.set_title('Missingness per marker')
    ax2.set_xlabel('Percent Missingness')
    plt.tight_layout()

def histogramAverageCounts(countsFile,snpProportion, embedding):
    """
    Plot average counts per gene, UMAP with average counts

    Args:
        counts: input file with number of reads for each marker
        snpProportion: processed SNP proportion data
        embedding: UMAP embedding of data 
    """
    counts = pd.read_csv(countsFile, index_col='MarkerName')
    counts = counts.groupby(['MarkerName']).sum() #combine counts for both alleles
    counts = counts[snpProportion.columns].loc[snpProportion.index] #subset to the samples, loci in snpProportion
    
    avgCounts = np.mean(counts, axis = 0)
    belowcut = np.where(avgCounts < 200)[0]
    
    ax1, ax2 = plotDouble()
    ax1.hist(avgCounts, bins = 20)
    ax1.set_xlabel('Average counts per gene')
    ax2.scatter(embedding[:, 0], embedding[:, 1], s = 1, c='grey', alpha = 0.1)
    im = ax2.scatter(embedding[belowcut, 0], embedding[belowcut, 1], s = 1, c=avgCounts[belowcut])
    plt.colorbar(im)
    plt.tight_layout()

def barchartRef(snpProportion, output, sampleMeta):
    """
    Barchart with the prevalence of each observed reference variety

    Args:
        snpProportion: processed SNP proportion data
        output: output dataframe frome base.py
        sampleMeta: metadata paired with genotyping data
    """
    w = sampleMeta[sampleMeta['short_name'].isin(snpProportion.columns.astype('int'))]
    var, counts = np.unique(output['variety'], return_counts=True)
    
    refshort = w['short_name'][(sampleMeta['reference'].notna())].values.astype('str') #references
    admixedshort = output['short_name'][output['variety'] == 'Admixed'].values.astype('str') #admixed
    
    landraceName = var[np.flatnonzero(np.core.defchararray.find(var.astype('str'),'Genetic entity')!=-1)]
    landraceShort = output['short_name'][np.isin(output['variety'],landraceName)].values.astype('str')
    
    callshort = np.setdiff1d(output['short_name'],np.concatenate((refshort, admixedshort,landraceShort)))
    callVarieties, callVarietiesCount = np.unique(output[output['short_name'].isin(callshort)]['variety'], return_counts=True)

    fig, ax = plt.subplots(figsize=(14.4,4.8))
    ax.barh(np.arange(len(callVarieties)), callVarietiesCount[np.argsort(callVarietiesCount)])
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_yticks(np.arange(len(callVarieties)),callVarieties[np.argsort(callVarietiesCount)])
    plt.tight_layout()
    
def umapRefLandrace(snpProportion, output, sampleMeta, cutoff, noRef=False):
    """
    Two UMAPs, L colored by reference varieties and R colored by common non reference varieties 

    Args:
        snpProportion: processed SNP proportion data
        output: output dataframe frome base.py
        sampleMeta: metadata paired with genotyping data
        cutoff: show landraces that occur more times than this cutoff
        noRef: if True, don't plot reference samples in UMAP    
    """
    w = sampleMeta[sampleMeta['short_name'].isin(snpProportion.columns.astype('int'))]
    var, counts = np.unique(output['variety'], return_counts=True)
    
    refshort = w['short_name'][(sampleMeta['reference'].notna())].values.astype('str') #references
    admixedshort = output['short_name'][output['variety'] == 'Admixed'].values.astype('str') #admixed
    
    landraceLowName = var[counts < cutoff][np.flatnonzero(np.core.defchararray.find(var[counts < cutoff].astype('str'),'Genetic entity')!=-1)]
    landraceLowshort = output['short_name'][np.isin(output['variety'],landraceLowName)].values.astype('str')#landrace < cutoff
    landraceHighName = var[counts >= cutoff][np.flatnonzero(np.core.defchararray.find(var[counts >= cutoff].astype('str'),'Genetic entity')!=-1)]
    landraceHighshort = output['short_name'][np.isin(output['variety'],landraceHighName)].values.astype('str')#landrace >= cutoff
    landraceHighVarieties = np.unique(output[output['short_name'].isin(landraceHighshort)]['variety'])
    
    callshort = np.setdiff1d(output['short_name'],np.concatenate((refshort, admixedshort,landraceLowshort,landraceHighshort)))
    callVarieties = np.unique(output[output['short_name'].isin(callshort)]['variety'])
        
    ax1, ax2 = plotDouble()
    
    #varieties    
    tab20 = plt.colormaps['tab20b'].resampled(len(callVarieties))
    if noRef ==  True:
        noRefCol = np.where(~np.isin(snpProportion.columns, refshort))[0]
        ax1.scatter(output['embedding_X'][noRefCol].values, output['embedding_Y'][noRefCol].values, s = 1, c='grey', alpha = 0.1)
    else:
        ax1.scatter(output['embedding_X'].values, output['embedding_Y'].values, s = 1, c='grey', alpha = 0.1)
    for i, varietyName in enumerate(callVarieties):
        sampleNames = np.setdiff1d(output['short_name'][output['variety'] == varietyName].values.astype('str'), refshort)
        if len(sampleNames) > 0:
            sampleIndex = np.where(np.isin(snpProportion.columns, sampleNames))[0]
            ax1.scatter(output.iloc[sampleIndex]['embedding_X'].values, output.iloc[sampleIndex]['embedding_Y'].values, s = 2, color=tab20(i), label = varietyName)
    ax1.legend(markerscale=2, fontsize=5)
    
    #ax2, plot landraces with prevalence above cutoff
    tab20 = plt.colormaps['tab20b'].resampled(len(landraceHighVarieties))
    if noRef ==  True:
        ax2.scatter(output['embedding_X'][noRefCol].values, output['embedding_Y'][noRefCol].values, s = 1, c='grey', alpha = 0.1)
    else:
        ax2.scatter(output['embedding_X'].values, output['embedding_Y'].values, s = 1, c='grey', alpha = 0.1)
    for i,varietyName in enumerate(landraceHighVarieties):
        sampleNames = np.setdiff1d(output['short_name'][output['variety'] == varietyName].values.astype('str'), refshort)
        if len(sampleNames) > 0:
            sampleIndex = np.where(np.isin(snpProportion.columns, sampleNames))[0]
            ax2.scatter(output.iloc[sampleIndex]['embedding_X'].values, output.iloc[sampleIndex]['embedding_Y'].values, s = 2, color=tab20(i), label = varietyName)
    ax2.legend(markerscale=2, ncol=3, fontsize=5)
    plt.tight_layout()

def umapReferenceSeparate(snpProportion, embedding, sampleMeta):
    """
    Two UMAPs where the references from each variety are a different color

    Args:
        snpProportion: processed SNP proportion data
        embedding: UMAP embedding of data
        sampleMeta: metadata paired with genotyping data
    """
    references = sampleMeta[(sampleMeta['reference'].notna())]
    refNames = np.unique(references['reference'])
    midPoint = int(np.ceil(len(refNames)/2))
    
    ax1, ax2 = plotDouble()
    tab20 = mpl.colormaps['tab20b'].resampled(min(20,midPoint))
    
    ax1.scatter(embedding[:, 0], embedding[:, 1], s = 1, c='grey', alpha = 0.1)
    for i in range(0,midPoint):
        rows = references[references['reference'] == refNames[i]]
        referencesIndex = np.where(np.isin(snpProportion.columns, rows['short_name'].astype('str')))[0]
        ax1.scatter(embedding[referencesIndex, 0], embedding[referencesIndex, 1], s = 3, color = tab20(i), label = refNames[i])
    ax1.legend(markerscale=1, ncol = 3, fontsize=5)
     
    ax2.scatter(embedding[:, 0], embedding[:, 1], s = 1, c='grey', alpha = 0.1)
    for i in range(midPoint,len(refNames)):
        rows = references[references['reference'] == refNames[i]]
        referencesIndex = np.where(np.isin(snpProportion.columns, rows['short_name'].astype('str')))[0]
        ax2.scatter(embedding[referencesIndex, 0], embedding[referencesIndex, 1], s = 3, color = tab20(i-midPoint), label = refNames[i])
    ax2.legend(markerscale=1, ncol = 3, fontsize=5)
        
    plt.tight_layout()

def umapDivergence(snpProportion, embedding):
    """
    UMAP where samples are colored by divergence value

    Args:
        snpProportion: processed SNP proportion data
        embedding: UMAP embedding of data
    """
    divergence = homozygousDivergence(snpProportion.values)
    abovecut = np.where(divergence > 0.02)[0]
    
    ax1, ax2 = plotTemplate()
    ax1.scatter(embedding[:, 0], embedding[:, 1], s = 1, c='grey', alpha = 0.1)
    SC = ax1.scatter(embedding[abovecut, 0], embedding[abovecut, 1], s = 1, c=divergence[abovecut])
    plt.colorbar(SC, cax=ax2)
    plt.tight_layout()

def histogramDivergence(snpProportion,sampleMeta):
    """
    Histogram of divergence values for field sample (L) and reference samples (R)

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
    """
    ax1, ax2 = plotDouble()
    references = sampleMeta[(sampleMeta['reference'].notna())]
    sampleIndex = np.where(~np.isin(snpProportion.columns, references['short_name'].astype('str')))[0]
    divergenceBySample = homozygousDivergence(snpProportion[snpProportion.columns[sampleIndex]])
    ax1.hist(divergenceBySample, bins = 20)
    ax1.set_title('Field samples')
    ax1.set_xlabel('Mean divergence')
    
    referencesIndex = np.where(np.isin(snpProportion.columns, references['short_name'].astype('str')))[0]
    divergenceByRef = homozygousDivergence(snpProportion[snpProportion.columns[referencesIndex]])
    ax2.hist(divergenceByRef, bins = 20)
    ax2.set_title('References')
    ax2.set_xlabel('Mean divergence')
    plt.tight_layout()
    
def dendrogram(snpProportion, sampleMeta, communities, COI, cutHeight, tick_type='sampleRef', cutLine = False):
    """
    Dendrogram

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        communities: DBSCAN cluster number for each sample
        COI: DBSCAN cluster to plot
        cutHeight: cutoff value for cutting a dendrogram into clusters 
        tick_type: 'sampleRef' (label as sample number or reference), 'references' (only label references)
        cutLine: if True, plot a horizontal line at this value
    """

    clusterSubset = snpProportion[snpProportion.columns[np.where(communities == COI)]]
    Y_cluster = sch.linkage(clusterSubset.values.T, metric='correlation') #sort samples

    if tick_type == 'sampleRef': #add sample number and reference to x-ticks
        references = sampleMeta[(sampleMeta['reference'].notna())]
        refInCluster = clusterSubset.columns[np.isin(clusterSubset.columns, references['short_name'].values.astype('str'))]
        labels = np.copy(clusterSubset.columns.values)
        for i in refInCluster:
            index = (np.where(clusterSubset.columns == i)[0][0])    
            ref = (sampleMeta[sampleMeta['short_name'] == int(i)]['reference']).values[0]
            labels[index] = ref
            
    if tick_type == 'references': #add references to x-ticks
        references = sampleMeta[(sampleMeta['reference'].notna())]
        refInCluster = clusterSubset.columns[np.isin(clusterSubset.columns, references['short_name'].values.astype('str'))]
        labels = ['']*len(clusterSubset.columns)
        for i in refInCluster:
            index = (np.where(clusterSubset.columns == i)[0][0])    
            ref = (sampleMeta[sampleMeta['short_name'] == int(i)]['reference']).values[0]
            labels[index] = ref
     
    #plot dendrogram
    plt.figure(figsize=(14.4,4.8))
    sch.dendrogram(Y_cluster, labels = labels, color_threshold = cutHeight) #sample names
    if cutLine:
        plt.plot([0,10*clusterSubset.shape[1]],[cutHeight, cutHeight],'k')
    plt.tight_layout()

def umapReleaseYear(snpProportion, sampleMeta, embedding):
    """
    UMAP where samples are colored by release year

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        embedding: UMAP embedding of data
    """

    released = sampleMeta[(sampleMeta['release_year'].notna()) & (sampleMeta['release_year'] > 0)]
    releasedIndex = np.where(np.isin(snpProportion.columns, released['short_name'].astype('str')))[0]

    year = []
    for sample in snpProportion.columns[releasedIndex]: #these need to be in the same order
        year.append(released[released['short_name'] == int(sample)]['release_year'].values[0])
        
    ax1, ax2 = plotTemplate()
    ax1.scatter(embedding[:, 0], embedding[:, 1], s = 1, c='grey', alpha = 0.1)
    SC = ax1.scatter(embedding[releasedIndex, 0], embedding[releasedIndex, 1], s = 1, c=year)
    plt.colorbar(SC, cax=ax2)
    plt.tight_layout()
    
def umapRefCalls(snpProportion, output, sampleMeta, noRef=False, minYear=None):
    """
    UMAPs where samples that match reference varieties are colored by varietal release year

    Args:
        snpProportion: processed SNP proportion data
        output: output dataframe frome base.py
        sampleMeta: metadata paired with genotyping data
        noRef: if True, don't plot reference samples in UMAP    
        minYear: only color varieties released on or after this year
    """
    w = sampleMeta[sampleMeta['short_name'].isin(snpProportion.columns.astype('int'))]
    var, counts = np.unique(output['variety'], return_counts=True)
     
    refshort = w['short_name'][(sampleMeta['reference'].notna())].values.astype('str') #references        
    admixedshort = output['short_name'][output['variety'] == 'Admixed'].values.astype('str') #admixed
    landraceName = var[np.flatnonzero(np.core.defchararray.find(var.astype('str'),'Genetic entity')!=-1)]
    landraceshort = output['short_name'][np.isin(output['variety'],landraceName)].values.astype('str')
    
    callshort = np.setdiff1d(output['short_name'],np.concatenate((admixedshort,landraceshort, refshort)))          
    callVarieties = np.unique(output[output['short_name'].isin(callshort)]['variety'])

    if minYear != None: #fillter out varieties released on or after minYear
        filteredRef = np.unique(w[w['release_year'] >= minYear]['reference'])
        callVarieties = np.intersect1d(callVarieties, filteredRef)
             
    fig = plt.figure(figsize=(7.2,6))
    gs=GridSpec(1,1, figure = fig)
    ax1 = plt.subplot(gs[0])
     
    #varieties    
    tab20 = plt.colormaps['tab20b'].resampled(len(callVarieties))
    if noRef ==  True:
        noRefCol = np.where(~np.isin(snpProportion.columns, refshort))[0]
        ax1.scatter(output['embedding_X'][noRefCol].values, output['embedding_Y'][noRefCol].values, s = 1, c='grey', alpha = 0.1)
    else:
        ax1.scatter(output['embedding_X'].values, output['embedding_Y'].values, s = 1, c='grey', alpha = 0.1)
    for i, varietyName in enumerate(callVarieties):
        sampleNames = np.setdiff1d(output['short_name'][output['variety'] == varietyName].values.astype('str'), refshort)
        if len(sampleNames) > 0:
            sampleIndex = np.where(np.isin(snpProportion.columns, sampleNames))[0]
            ax1.scatter(output.iloc[sampleIndex]['embedding_X'].values, output.iloc[sampleIndex]['embedding_Y'].values, s = 2, color=tab20(i), label = varietyName)
    ax1.legend(markerscale=5, fontsize=5)   
    plt.tight_layout()

def heatmapDendrogramAll(snpProportion, sampleMeta, communities, filePrefix, cutHeight, heatmapTick = 'referencesAll', dendrogramTick = 'references', cutLine = False): 
    """
    Generate a heatmap and dendrogram for each cluster

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        communities: DBSCAN cluster number for each sample
        filePrefix: prefix for output filenames
        cutHeight: cutoff value for cutting a dendrogram into clusters
        heatmapTick: see options in heatmapManyClusters()
        dendrogramTick: see options in dendrogram()
        cutLine: if True, plot a horizontal line at this value
    """
    for clusterNum in np.unique(communities):
        heatmapManyClusters(snpProportion, sampleMeta, communities, [clusterNum], tickType=heatmapTick)
        plt.savefig(filePrefix+' heatmap cluster '+str(clusterNum)+'.png', dpi = 300)

        dendrogram(snpProportion, sampleMeta, communities, clusterNum, cutHeight, tick_type=dendrogramTick, cutLine = False)
        plt.savefig(filePrefix+' dendrogram cluster '+str(clusterNum)+' (cut height'+str(cutHeight)+').png', dpi = 300)

def barchartLandrace(snpProportion, output, sampleMeta, filePrefix = None, cutoff = 2):
    """
    Barchart with the prevalence of each observed genetic entity

    Args:
        snpProportion: processed SNP proportion data
        output: output dataframe frome base.py
        sampleMeta: metadata paired with genotyping data
        filePrefix: optional prefix for output filenames to save
    """
    var, counts = np.unique(output['variety'], return_counts=True)
        
    landraceName = var[np.flatnonzero(np.core.defchararray.find(var.astype('str'),'Genetic entity')!=-1)]
    landraceShort = output['short_name'][np.isin(output['variety'],landraceName)].values.astype('str')
    landraceVarieties, landraceVarietiesCount = np.unique(output[output['short_name'].isin(landraceShort)]['variety'], return_counts=True)
    
    landraceVarietiesFilter =landraceVarieties[landraceVarietiesCount > cutoff]
    landraceVarietiesCountFilter = landraceVarietiesCount[landraceVarietiesCount > cutoff]

    fig, ax = plt.subplots(figsize=(14.4,4.8))
    ax.barh(np.arange(len(landraceVarietiesFilter)), landraceVarietiesCountFilter[np.argsort(landraceVarietiesCountFilter)])
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_yticks(np.arange(len(landraceVarietiesFilter)),landraceVarietiesFilter[np.argsort(landraceVarietiesCountFilter)])
    plt.tight_layout()
    if filePrefix:
        plt.savefig(filePrefix + ' genetic entity barchart.png', dpi = 300)
        
    #histogram of occurences of genetic entities
    plt.figure()
    if (landraceVarietiesCount.size > 0):
        plt.hist(landraceVarietiesCount, bins = np.arange(max(landraceVarietiesCount)+2))
        plt.xticks(np.arange(1,max(landraceVarietiesCount)+2))
    plt.tight_layout()
    if filePrefix:
        plt.savefig(filePrefix + ' genetic entity histogram.png', dpi = 300)

def heatmapDendrogram(snpProportion, sampleMeta, communities, COI, cutHeight, cutLine = False):
    """
    Paired dendrogram and heatmap for the same cluster

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        communities: DBSCAN cluster number for each sample
        COI: DBSCAN cluster to plot
        cutHeight: cutoff value for cutting a dendrogram into clusters 
        cutLine: if True, plot a horizontal line at this value
    """
    fig = plt.figure(figsize=(7.2,12))
    gs=GridSpec(2,2, width_ratios=[0.2,6], height_ratios=[1,1], figure = fig)    

    #dendrogram half
    clusterSubset = snpProportion[snpProportion.columns[np.where(communities == COI)]]
    Y_cluster = sch.linkage(clusterSubset.values.T, metric='correlation') #sort samples
                 
    #plot dendrogram
    ax1 = plt.subplot(gs[0,1])
    sch.dendrogram(Y_cluster, color_threshold = cutHeight, no_labels = True)
    if cutLine:
        plt.plot([0,10*clusterSubset.shape[1]],[cutHeight, cutHeight],'k')
    ax1.set_title('Cluster '+str(COI))

    #heatmap half
    subset = snpProportion[snpProportion.columns[np.where(communities == COI)[0]]].values
    subsetReorder, clusterOrder, breakPoints = clusterReorder(subset, [subset.shape[1]])
    references = sampleMeta[(sampleMeta['reference'].notna())]
    referencesIndex = np.where(np.isin(snpProportion.columns, references['short_name'].astype('str')))[0]
    
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[1,1])
    SC=ax3.imshow(subsetReorder, aspect='auto', interpolation='none', cmap='coolwarm',vmin=0,vmax = 1)
    ax3.set_yticks([])
    plt.colorbar(SC, cax=ax2)
    
    #add reference labels to x-axis
    numInRef, numInCluster, numInZ = np.intersect1d(referencesIndex, np.where(communities == COI)[0], return_indices=True) #number of the reference
    if len(numInRef) > 0:
        ticks = []
        labels = []
        for i in range(len(numInRef)):
            ticks.append(np.where(clusterOrder == numInZ[i])[0][0])
            labels.append(sampleMeta[sampleMeta['short_name'] == int(snpProportion.columns[numInRef[i]])]['reference'].values[0])      
        ax3.set_xticks(ticks, labels, rotation = 90)
        
    else:
        ax3.set_xticks([])
        
    ax2.yaxis.tick_left()    
    plt.tight_layout()

