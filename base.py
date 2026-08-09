#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variety calls from counts data using clustering
"""

import pandas as pd
import numpy as np
import json
import umap
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scripts.graphs as plot
import scripts.randMatrix as rand
import scripts.referenceProcessing


def filterData(countsFile, metaFile, minloci, minSample, refFilter=None):
    """
    Input the reformatted counts file and paired metadata file, filter out low quality samples/genes and then interpolate missing data

    Args:
        countsFile: path to reformatted counts file
        metaFile: path to the metadata file paired with the countsFile
        minSample: samples must be missing counts data from less than X proportion of markers
        minloci: markers must be have counts data from more than than Y proportion of samples
        refFilter: (optional) remove references with a divergence score above this value
    """
    # import counts data
    counts = pd.read_csv(countsFile, index_col="MarkerName")
    snpProportion = (
        counts.groupby(["MarkerName"]).first() / counts.groupby(["MarkerName"]).sum()
    )  # if the count for both SNPs are zero --> NaN

    # check that all marker names are unique
    marker, markerCount = np.unique(counts.index, return_counts=True)
    if len(np.where(markerCount > 2)[0]) > 0:
        raise ValueError(
            "More than two rows with the same marker name, please differentiate marker names in the count file"
        )

    # import sample metadata
    sampleMeta = pd.read_csv(metaFile)
    refRemove = sampleMeta[sampleMeta["reference"] == "REMOVE"][
        "short_name"
    ].values.astype("str")

    # optionally remove references above a divergence cutoff
    if refFilter:
        divergent = snpProportion.columns[
            plot.homozygousDivergence(snpProportion) > refFilter
        ].astype(
            "int"
        )  # all samples above cutoff
        references = sampleMeta[(sampleMeta["reference"].notna())]["short_name"].values
        refRemove = np.append(
            refRemove,
            sampleMeta[
                sampleMeta["short_name"].isin(np.intersect1d(divergent, references))
            ]["short_name"].values.astype("str"),
        )

    snpProportion = snpProportion.drop(refRemove, axis=1)
    sampleMeta = sampleMeta.drop(
        sampleMeta[sampleMeta["short_name"].isin(refRemove.astype("int"))].index
    )

    snpProportionNoInterpolation = snpProportion.copy()

    # filter snpProportion for samples and genes with too many NaN
    snpProportion = snpProportion[
        snpProportion.isna().sum(axis=1) < (minloci * snpProportion.shape[1])
    ]  # remove genes
    snpProportion = snpProportion.drop(
        columns=snpProportion.columns[
            snpProportion.isna().sum(axis=0) > (minSample * snpProportion.shape[0])
        ]
    )  # remove samples

    # interpolate NaN using gene average
    snpProportion = snpProportion.where(
        pd.notna(snpProportion), snpProportion.mean(axis=1), axis="rows"
    )

    return snpProportion, snpProportionNoInterpolation, sampleMeta


def embedData(snpProportion, umapSeed):
    """
    Input the processed snpProportion data, embed with UMAP, and then cluster using DBSCAN

    Args:
        snpProportion: processed SNP proportion data
        umapSeed: RNG seed for umap embedding
    """
    # UMAP embedding
    reducer = umap.UMAP(random_state=umapSeed, init="random")
    embedding = reducer.fit_transform(
        snpProportion.T
    )  # the order is the same after embedding

    return embedding


def clusteringDBSCAN(
    snpProportion, sampleMeta, embedding, epsilon, filePrefix, admixedCutoff
):
    """
    Input the processed snpProprtion data, embed with UMAP, and then cluster using DBSCAN

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        embedding: UMAP embedding of snpProportion
        epsilon: epsilon parameter for DBSCAN clustering
        filePrefix: prefix for output filenames
    """
    # cluster using DBSCAN
    db_communities = DBSCAN(eps=epsilon, min_samples=1).fit(embedding).labels_

    # save output figures
    plot.umapCluster(embedding, db_communities)
    plt.savefig(filePrefix + " UMAP DBSCAN (epsilon " + str(epsilon) + ").png", dpi=300)
    plt.close()

    plot.umapReference(snpProportion, embedding, sampleMeta, db_communities)
    plt.savefig(
        filePrefix
        + " UMAP references (DBSCAN clusters, epsilon "
        + str(epsilon)
        + ").png",
        dpi=300,
    )
    plt.close()

    if admixedCutoff:
        plot.histogramDivergence(snpProportion, sampleMeta)
        plt.savefig(filePrefix + " histogram divergence.png", dpi=300)
        plt.close()

    return db_communities


def evaluateEpsilon(
    embedding,
    filePrefix,
    epsilonRangeStart=0.1,
    epsilonRangeStop=3.15,
    epsilonRangeStep=0.15,
):
    """
    Evaluate different epsilon values for DBSCAN

    Args:
        embedding: UMAP embedding of snpProportion
        filePrefix: prefix for output filenames
        epsilonRangeStart: (optional) epsilon range start
        epsilonRangeStop: (optional) epsilon range stop
        epsilonRangeStep: (optional) epsilon range step
    """
    ks = np.around(
        np.arange(epsilonRangeStart, epsilonRangeStop, epsilonRangeStep), 2
    )  # Range of epsilon values for DBSCAN
    rand.randScoreMatrix(embedding, ks, "DBSCAN")
    plt.savefig(filePrefix + " DBSCAN rand matrix.png", dpi=300)


def labelSamples(
    snpProportion,
    sampleMeta,
    db_communities,
    embedding,
    cutHeight,
    admixedCutoff,
    filePrefix,
    snpProportionNoInterpolation,
    parameterFile,
):
    """
    Evaluate different cut height values for processing the dendrogram

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        db_communities: DBSCAN cluster number for each sample
        embedding: UMAP embedding of snpProportion
        cutHeight: cutoff value for cutting a dendrogram into clusters
        admixedCutoff: clades without a reference and a minimum divergence value above this will be labeled as admixed
        filePrefix: prefix for output filenames
    """
    # consolidate outputs
    output = pd.DataFrame(embedding, columns=["embedding_X", "embedding_Y"])
    output["cluster"] = db_communities
    output["short_name"] = snpProportion.columns
    if admixedCutoff:
        output["divergence"] = plot.homozygousDivergence(snpProportion)
    output["variety"] = pd.NA

    for cluster in np.unique(db_communities):
        # subset for a single DBSCAN cluster
        subsetIndex = np.where(db_communities == cluster)[0]

        # cluster subset of samples using heirarchical clustering
        Y_cluster = sch.linkage(
            snpProportion[snpProportion.columns[subsetIndex]].values.T,
            metric="correlation",
        )

        # label samples
        communities, names = rand.labelHCLandrace(
            snpProportion[snpProportion.columns[subsetIndex]],
            sampleMeta,
            Y_cluster,
            cutHeight,
            clusterNumber=cluster,
            admixedCutoff=admixedCutoff,
        )
        varietiesList = []
        for i in communities.astype("int"):
            varietiesList.append(names[i][0])
        output.loc[subsetIndex, "variety"] = varietiesList

    # save outputs
    plot.umapRefLandrace(snpProportion, output, sampleMeta, 5, noRef=True)
    plt.savefig(
        filePrefix
        + " UMAP clustering predictions (cut height"
        + str(cutHeight)
        + ").png",
        dpi=300,
    )

    plot.barchartRef(snpProportion, output, sampleMeta)
    plt.savefig(
        filePrefix
        + " bar chart clustering predictions (cut height"
        + str(cutHeight)
        + ").png",
        dpi=300,
    )

    # add missingness
    output["missingness"] = (
        (snpProportionNoInterpolation.isna().sum(axis=0)[snpProportion.columns])
        / snpProportionNoInterpolation.shape[1]
    ).values

    # add heterozygosity
    output["heterozygosity"] = (
        ((snpProportion > 0.05) & (snpProportion < 0.95)).sum(axis=0)
        / snpProportion.shape[0]
    ).values

    output.to_csv(
        filePrefix + "_clusteringOutputData_cutHeight" + str(cutHeight) + ".csv",
        index=False,
    )

    # append sampleMeta
    metaOrder = []
    for i in output["short_name"]:
        metaOrder.append(np.where(sampleMeta["short_name"] == int(i))[0][0])

    sampleMetaCrop = sampleMeta.iloc[metaOrder]
    sampleMetaCrop.index = output.index
    output2 = pd.concat([output, sampleMetaCrop], axis=1)

    # add paramters
    with open(parameterFile) as f:
        data = json.load(f)

    # output2["parameters"] = np.nan
    # output2["parameters"].iloc[0 : len(data)] = list(dict.items(data))
    # output2.to_csv(
    #     filePrefix + "_clusteringOutputAllData_cutHeight" + str(cutHeight) + ".csv",
    #     index=False,
    # )

    return output, output2


def loadParameters(parameterFile):
    """
    load and validate json file with all of the parameters

    Args:
        minSample: samples must be missing from less than X loci
        minloci: loci must be absent from less than Y samples
        umapSeed: RNG seed for UMAP
        epsilon: epsilon value for DBSCAN
        cutHeight: cutoff value for cutting a dendrogram into clusters
        admixedCutoff: clades without a reference and a minimum divergence value above this will be labeled as admixed, null will be interpreted by JSON as None
        filePrefix: prefix for output filenames
        inputCountsFile: name and path to DArT counts file
        inputMetaFile: name and path to the metadata file paired with the countsFile
    """
    with open(parameterFile) as f:
        data = json.load(f)

    minSample = data.get("minSample", None)
    if not isinstance(minSample, (int, float)):
        raise ValueError("minSample must be a float. You entered: " + str(minSample))
    minSample = float(minSample)

    minloci = data.get("minloci", None)
    if not isinstance(minloci, (int, float)):
        raise ValueError("minloci must be a float")
    minloci = float(minloci)

    umapSeed = data.get("umapSeed", None)
    if not isinstance(umapSeed, int):
        raise ValueError("umapSeed must be an int")

    epsilon = data.get("epsilon", None)
    if not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be a float")
    epsilon = float(epsilon)

    cutHeight = data.get("cutHeight", None)
    if not isinstance(cutHeight, (int, float)):
        raise ValueError("cutHeight must be a float")
    cutHeight = float(cutHeight)

    admixedCutoff = data.get("admixedCutoff", None)
    if admixedCutoff is not None:
        if not isinstance(admixedCutoff, (int, float)):
            raise ValueError("admixedCutoff must be a float")
        admixedCutoff = float(admixedCutoff)

    filePrefix = data.get("filePrefix", None)
    if type(filePrefix) is not str:
        raise ValueError("filePrefix must be a string")

    inputCountsFile = data.get("inputCountsFile", None)
    inputMetaFile = data.get("inputMetaFile", None)

    return (
        minSample,
        minloci,
        umapSeed,
        epsilon,
        cutHeight,
        admixedCutoff,
        filePrefix,
        inputCountsFile,
        inputMetaFile,
    )


def runPipeline(parameterFile):
    # Load the parameters from the JSON file
    (
        minSample,
        minloci,
        umapSeed,
        epsilon,
        cutHeight,
        admixedCutoff,
        filePrefix,
        inputCountsFile,
        inputMetaFile,
    ) = loadParameters(parameterFile)

    # Filter the data based on the provided parameters
    snpProportion, snpProportionNoInterpolation, sampleMeta = filterData(
        inputCountsFile, inputMetaFile, minloci, minSample
    )

    # Embed the filtered data using UMAP
    # i.e., reduce the dimensionality of the data for clustering
    embedding = embedData(snpProportion, umapSeed)

    print(embedding)
    # Cluster the embedded data using DBSCAN
    db_communities = clusteringDBSCAN(
        snpProportion, sampleMeta, embedding, epsilon, filePrefix, admixedCutoff
    )

    # Label the samples based on the clustering results
    output, output2 = labelSamples(
        snpProportion,
        sampleMeta,
        db_communities,
        embedding,
        cutHeight,
        admixedCutoff,
        filePrefix,
        snpProportionNoInterpolation,
        parameterFile,
    )

    return (
        snpProportion,
        snpProportionNoInterpolation,
        sampleMeta,
        embedding,
        db_communities,
        output,
    )


if __name__ == "__main__":
    parameterFile = "./tutorial/parametersRiceTutorial.json"
    (
        minSample,
        minloci,
        umapSeed,
        epsilon,
        cutHeight,
        admixedCutoff,
        filePrefix,
        inputCountsFile,
        inputMetaFile,
    ) = loadParameters(parameterFile)

    snpProportion, snpProportionNoInterpolation, sampleMeta = filterData(
        inputCountsFile, inputMetaFile, minloci, minSample
    )
    embedding = embedData(snpProportion, umapSeed)
    db_communities = clusteringDBSCAN(
        snpProportion, sampleMeta, embedding, epsilon, filePrefix, admixedCutoff
    )
    output, output2 = labelSamples(
        snpProportion,
        sampleMeta,
        db_communities,
        embedding,
        cutHeight,
        admixedCutoff,
        filePrefix,
        snpProportionNoInterpolation,
        parameterFile,
    )
