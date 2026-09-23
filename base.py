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
from scripts.reduceDimensions import run_pca_and_select_markers
from pathlib import Path
import logging
import time
import joblib
from hdbscan import HDBSCAN
from sklearn.pipeline import Pipeline
from scripts.consensusPurity import (
    calculate_purity_by_name,
    get_cluster_metrics,
    calculate_purity_matrix,
    plot_cluster_metrics,
)

# from pca import run_pca_and_select_markers


# def filterData(countsFile, metaFile, minloci, minSample, refFilter=None):
#     """
#     Input the reformatted counts file and paired metadata file, filter out low quality samples/genes and then interpolate missing data

#     Args:
#         countsFile: path to reformatted counts file
#         metaFile: path to the metadata file paired with the countsFile
#         minSample: samples must be missing counts data from less than X proportion of markers
#         minloci: markers must be have counts data from more than than Y proportion of samples
#         refFilter: (optional) remove references with a divergence score above this value
#     """
#     # import counts data
#     counts = pd.read_csv(countsFile, index_col="MarkerName")
#     snpProportion = (
#         counts.groupby(["MarkerName"]).first() / counts.groupby(["MarkerName"]).sum()
#     )  # if the count for both SNPs are zero --> NaN

#     # check that all marker names are unique
#     marker, markerCount = np.unique(counts.index, return_counts=True)
#     if len(np.where(markerCount > 2)[0]) > 0:
#         raise ValueError(
#             "More than two rows with the same marker name, please differentiate marker names in the count file"
#         )

#     # import sample metadata
#     sampleMeta = pd.read_csv(metaFile)
#     refRemove = sampleMeta[sampleMeta["reference"] == "REMOVE"][
#         "short_name"
#     ].values.astype("str")

#     # optionally remove references above a divergence cutoff
#     if refFilter:
#         divergent = snpProportion.columns[
#             plot.homozygousDivergence(snpProportion) > refFilter
#         ].astype(
#             "int"
#         )  # all samples above cutoff
#         references = sampleMeta[(sampleMeta["reference"].notna())]["short_name"].values
#         refRemove = np.append(
#             refRemove,
#             sampleMeta[
#                 sampleMeta["short_name"].isin(np.intersect1d(divergent, references))
#             ]["short_name"].values.astype("str"),
#         )

#     snpProportion = snpProportion.drop(refRemove, axis=1)
#     sampleMeta = sampleMeta.drop(
#         sampleMeta[sampleMeta["short_name"].isin(refRemove.astype("int"))].index
#     )

#     snpProportionNoInterpolation = snpProportion.copy()

#     # filter snpProportion for samples and genes with too many NaN
#     snpProportion = snpProportion[
#         snpProportion.isna().sum(axis=1) < (minloci * snpProportion.shape[1])
#     ]  # remove genes
#     snpProportion = snpProportion.drop(
#         columns=snpProportion.columns[
#             snpProportion.isna().sum(axis=0) > (minSample * snpProportion.shape[0])
#         ]
#     )  # remove samples

#     # interpolate NaN using gene average
#     snpProportion = snpProportion.where(
#         pd.notna(snpProportion), snpProportion.mean(axis=1), axis="rows"
#     )

#     return snpProportion, snpProportionNoInterpolation, sampleMeta


def filterData(
    countsFile, metaFile, minloci, minSample, refFilter=None, metaFilter=None
):
    """
    Input the reformatted counts file and paired metadata file, filter out low quality samples/genes and then interpolate missing data

    Args:
        countsFile: path to reformatted counts file
        metaFile: path to the metadata file paired with the countsFile
        minSample: samples must be missing counts data from less than X proportion of markers
        minloci: markers must be have counts data from more than than Y proportion of samples
        refFilter: (optional) remove references with a divergence score above this value
        metaFilter: (optional) dictionary of metadata columns and values to filter by.
                    e.g. {'Region': 'North', 'Batch': [1, 2, 3]}
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

    # --- NEW: Filter based on dictionary provided in metaFilter ---
    # if metaFilter:
    #     logging.info(f"Filtering on {metaFilter}")
    #     for col, values in metaFilter.items():
    #         if col in sampleMeta.columns:
    #             # Ensure the values are in a list so we can use .isin()
    #             if not isinstance(values, (list, tuple, set, np.ndarray)):
    #                 values = [values]

    #             # Filter the metadata dataframe
    #             sampleMeta = sampleMeta[sampleMeta[col].isin(values)]
    #         else:
    #             print(
    #                 f"Warning: Column '{col}' not found in metadata. Skipping this filter."
    #             )

    #     # Keep only the columns in snpProportion that correspond to the filtered short_names
    #     # We convert to strings for safe matching between dataframes
    #     kept_short_names = set(sampleMeta["short_name"].astype(str))
    #     cols_to_keep = [
    #         col for col in snpProportion.columns if str(col) in kept_short_names
    #     ]
    #     snpProportion = snpProportion[cols_to_keep]
    if metaFilter:
        logging.info(f"Filtering on {metaFilter}")
        for col, condition in metaFilter.items():
            if col in sampleMeta.columns:
                # Handle advanced filtering conditions (dict values)
                if isinstance(condition, dict):
                    # 1. Handle "exclude" or "not_in"
                    if "exclude" in condition or "not_in" in condition:
                        exclude_vals = condition.get("exclude", condition.get("not_in"))
                        if not isinstance(exclude_vals, (list, tuple, set, np.ndarray)):
                            exclude_vals = [exclude_vals]

                        # Separate null sentinels from clean scalar values
                        has_null_exclusion = any(
                            v is None or pd.isna(v) for v in exclude_vals
                        )
                        clean_vals = [
                            v for v in exclude_vals if v is not None and not pd.isna(v)
                        ]

                        if clean_vals:
                            sampleMeta = sampleMeta[~sampleMeta[col].isin(clean_vals)]
                        if has_null_exclusion:
                            sampleMeta = sampleMeta[sampleMeta[col].notna()]

                    # 2. Handle explicit "not_null" / "not_na" flag
                    elif condition.get("not_null") or condition.get("not_na"):
                        sampleMeta = sampleMeta[sampleMeta[col].notna()]

                # Handle standard inclusion matching (existing behavior)
                else:
                    values = condition
                    if not isinstance(values, (list, tuple, set, np.ndarray)):
                        values = [values]

                    sampleMeta = sampleMeta[sampleMeta[col].isin(values)]
            else:
                logging.warning(
                    f"Warning: Column '{col}' not found in metadata. Skipping this filter."
                )

        # Keep only the columns in snpProportion that correspond to the filtered short_names
        kept_short_names = set(sampleMeta["short_name"].astype(str))
        cols_to_keep = [
            col for col in snpProportion.columns if str(col) in kept_short_names
        ]
        snpProportion = snpProportion[cols_to_keep]
    # -------------------------------------------------------------

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

    # Added errors="ignore" to prevent KeyErrors if reference columns were already dropped
    snpProportion = snpProportion.drop(refRemove, axis=1, errors="ignore")
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


def updatedFilterData(countsFile, metaFile, minloci, minSample, refFilter=None):
    """
    Input the reformatted counts file and paired metadata file, filter out low quality samples/genes,
    convert to a discrete genotype matrix (0, 1, 2), and then interpolate missing data.
    IN BETA

    Args:
        countsFile: path to reformatted counts file
        metaFile: path to the metadata file paired with the countsFile
        minSample: samples must be missing counts data from less than X proportion of markers
        minloci: markers must be have counts data from more than than Y proportion of samples
        refFilter: (optional) remove references with a divergence score above this value
    """
    # import counts data
    counts = pd.read_csv(countsFile, index_col="MarkerName")

    # Isolate reference (first row) and alternate (last row) counts per marker
    grouped_counts = counts.groupby(["MarkerName"])
    ref_counts = grouped_counts.first()
    alt_counts = grouped_counts.last()

    # Build the genotype matrix
    genotype = pd.DataFrame(np.nan, index=ref_counts.index, columns=ref_counts.columns)
    genotype[(ref_counts > 0) & (alt_counts == 0)] = 0  # Homozygous Reference
    genotype[(ref_counts > 0) & (alt_counts > 0)] = 1  # Heterozygous
    genotype[(ref_counts == 0) & (alt_counts > 0)] = 2  # Homozygous Alternate
    # Note: where ref == 0 and alt == 0, it naturally remains NaN

    # check that all marker names are unique
    _, markerCount = np.unique(counts.index, return_counts=True)
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
        # Calculate proportion locally JUST for the divergence function so we don't break it
        tempProportion = ref_counts / (ref_counts + alt_counts)
        divergent = tempProportion.columns[
            plot.homozygousDivergence(tempProportion) > refFilter
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

    genotype = genotype.drop(refRemove, axis=1)
    sampleMeta = sampleMeta.drop(
        sampleMeta[sampleMeta["short_name"].isin(refRemove.astype("int"))].index
    )

    genotypeNoInterpolation = genotype.copy()

    # filter genotype for samples and genes with too many NaN
    genotype = genotype[
        genotype.isna().sum(axis=1) < (minloci * genotype.shape[1])
    ]  # remove genes
    genotype = genotype.drop(
        columns=genotype.columns[
            genotype.isna().sum(axis=0) > (minSample * genotype.shape[0])
        ]
    )  # remove samples

    # interpolate NaN using gene average
    # Note: This will introduce floats (e.g., 0.8 or 1.2) into the 0,1,2 matrix
    genotype = genotype.where(pd.notna(genotype), genotype.mean(axis=1), axis="rows")

    return genotype, genotypeNoInterpolation, sampleMeta


def generateGenotypeMatrix(countsFile, metaFile, minloci, minSample, refFilter=None):
    """
    Input the reformatted counts file and paired metadata file, filter out low quality samples/genes,
    convert to a discrete genotype matrix (0, 1, 2), and then interpolate missing data.

    Args:
        countsFile: path to reformatted counts file
        metaFile: path to the metadata file paired with the countsFile
        minSample: samples must be missing counts data from less than X proportion of markers
        minloci: markers must be have counts data from more than than Y proportion of samples
        refFilter: (optional) remove references with a divergence score above this value
    """
    # import counts data
    counts = pd.read_csv(countsFile, index_col="MarkerName")

    # Isolate reference (first row) and alternate (last row) counts per marker
    grouped_counts = counts.groupby(["MarkerName"])
    ref_counts = grouped_counts.first()
    alt_counts = grouped_counts.last()

    # Build the genotype matrix
    genotype = pd.DataFrame(np.nan, index=ref_counts.index, columns=ref_counts.columns)
    genotype[(ref_counts > 0) & (alt_counts == 0)] = 0  # Homozygous Reference
    genotype[(ref_counts > 0) & (alt_counts > 0)] = 1  # Heterozygous
    genotype[(ref_counts == 0) & (alt_counts > 0)] = 2  # Homozygous Alternate
    # Note: where ref == 0 and alt == 0, it naturally remains NaN

    # check that all marker names are unique
    _, markerCount = np.unique(counts.index, return_counts=True)
    if len(np.where(markerCount > 2)[0]) > 0:
        raise ValueError(
            "More than two rows with the same marker name, please differentiate marker names in the count file"
        )

    # import sample metadata
    sampleMeta = pd.read_csv(metaFile)
    refRemove = sampleMeta[sampleMeta["reference"] == "REMOVE"][
        "short_name"
    ].values.astype("str")

    genotype = genotype.drop(refRemove, axis=1)
    sampleMeta = sampleMeta.drop(
        sampleMeta[sampleMeta["short_name"].isin(refRemove.astype("int"))].index
    )

    genotypeNoInterpolation = genotype.copy()

    # filter genotype for samples and genes with too many NaN
    genotype = genotype[
        genotype.isna().sum(axis=1) < (minloci * genotype.shape[1])
    ]  # remove genes
    genotype = genotype.drop(
        columns=genotype.columns[
            genotype.isna().sum(axis=0) > (minSample * genotype.shape[0])
        ]
    )

    return genotype, sampleMeta


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

    return embedding, reducer


def clusteringDBSCAN(
    snpProportion, sampleMeta, embedding, epsilon, filePrefix, admixedCutoff, outputDir
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
    plt.savefig(
        Path(outputDir).joinpath(f"{filePrefix} UMAP DBSCAN (epsilon {epsilon}).png"),
        dpi=300,
    )
    plt.close()

    plot.umapReference(snpProportion, embedding, sampleMeta, db_communities)
    plt.savefig(
        Path(outputDir).joinpath(
            f"{filePrefix} UMAP references (DBSCAN clusters, epsilon {epsilon}).png",
        ),
        dpi=300,
    )
    plt.close()

    if admixedCutoff:
        plot.histogramDivergence(snpProportion, sampleMeta)
        plt.savefig(
            Path(outputDir).joinpath(f"{filePrefix} histogram divergence.png"), dpi=300
        )
        plt.close()

    return db_communities


def clusteringHDBSCAN(
    snpProportion,
    sampleMeta,
    embedding,
    min_cluster_size,
    filePrefix,
    admixedCutoff,
    outputDir,
):
    """
    Input the processed snpProportion data, embed with UMAP, and then cluster using HDBSCAN.

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        embedding: UMAP embedding of snpProportion
        min_cluster_size: minimum number of samples in a group for it to be considered a cluster
        filePrefix: prefix for output filenames
        admixedCutoff: threshold/boolean for plotting histogram divergence
    """
    # Cluster using HDBSCAN
    hdbs = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True).fit(
        embedding
    )
    # hdbs = HDBSCAN(min_cluster_size=min_cluster_size).fit(embedding)
    hdb_communities = hdbs.labels_

    plot.umapCluster(embedding, hdb_communities)
    plt.savefig(
        Path(outputDir).joinpath(
            f"{filePrefix} UMAP HDBSCAN (min_cluster_size {min_cluster_size}).png",
        ),
        dpi=300,
    )
    plt.close()

    plot.umapReference(snpProportion, embedding, sampleMeta, hdb_communities)
    plt.savefig(
        Path(outputDir).joinpath(
            f"{filePrefix} UMAP references HDBSCAN (min_cluster_size {min_cluster_size}).png",
        ),
        dpi=300,
    )
    plt.close()

    if admixedCutoff:
        plot.histogramDivergence(snpProportion, sampleMeta)
        plt.savefig(
            Path(outputDir).joinpath(f"{filePrefix} histogram divergence.png"), dpi=300
        )
        plt.close()

    return hdbs, hdb_communities


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
    plt.close()


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
    outputDir,
    probabilities=None,
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
    output["probabilities"] = probabilities
    if admixedCutoff:
        output["divergence"] = plot.homozygousDivergence(snpProportion)
    output["variety"] = pd.NA

    for cluster in np.unique(db_communities):
        # subset for a single DBSCAN cluster
        subsetIndex = np.where(db_communities == cluster)[0]

        # cluster subset of samples using hierarchical clustering
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
    logging.info(f"DEBUG snpProportion.shape: {snpProportion.shape}")
    logging.info(f"DEBUG output.shape: {output.shape}")
    logging.info(f"DEBUG sampleMeta.shape: {sampleMeta.shape}")
    logging.info(f"DEBUG output.variety unique: {output['variety'].unique()[:10]}")

    fig_title = f"UMAP clustering predictions (cut height {cutHeight})"
    plot.umapRefLandrace(
        snpProportion, output, sampleMeta, 5, noRef=True, fig_title=fig_title
    )
    logging.info(f"DEBUG umapRefLandrace artists: {plt.gca().get_children()[:10]}")

    plot.barchartRef(snpProportion, output, sampleMeta)
    logging.info(f"DEBUG barchartRef artists: {plt.gca().get_children()[:10]}")

    plt.savefig(
        Path(outputDir).joinpath(
            f"{filePrefix}_barChart_clusteringPredictions_cutHeight{cutHeight}.png",
        ),
        dpi=300,
    )
    plt.close()

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
        Path(outputDir).joinpath(
            f"{filePrefix}_clusteringOutputData_cutHeight{cutHeight}.csv"
        ),
        index=False,
    )

    # append sampleMeta
    metaOrder = []
    for i in output["short_name"]:
        metaOrder.append(np.where(sampleMeta["short_name"] == int(i))[0][0])

    sampleMetaCrop = sampleMeta.iloc[metaOrder]
    sampleMetaCrop.index = output.index
    output2 = pd.concat([output, sampleMetaCrop], axis=1)

    # add parameters
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
    if not Path(inputCountsFile).exists():
        raise ValueError(f"inputCountsFile must exist. I got {inputCountsFile}")

    inputMetaFile = data.get("inputMetaFile", None)
    if not Path(inputMetaFile).exists():
        raise ValueError(f"inputMetaFile must exist. I got {inputMetaFile}")

    outputDir = data.get("outputDir", None)
    if not Path(outputDir).exists():
        Path(outputDir).mkdir(parents=True, exist_ok=True)

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
        outputDir,
    )


def initLogging(outputdir, verbose=False):
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        filename=outputdir + "/app.log",
        filemode="a",  # 'a' to append logs, 'w' to overwrite every run
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=level,
    )


def runPipeline(
    parameterFile,
    pipelineName="clusteringForVarietyIdentification",
    verbose=False,
    method=2,
):
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
        outputDir,
    ) = loadParameters(parameterFile)

    initLogging(outputDir, verbose=verbose)

    time_stamp = time.localtime(time.time())
    logging.info(
        f"----------------- {time.strftime('%Y-%m-%d %H:%M:%S', time_stamp)} -----------------"
    )
    logging.info(f"Running pipeline: {pipelineName}")

    match pipelineName:
        case "clusteringForVarietyIdentification":
            # Load and validate he parameters from the JSON file
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
                outputDir,
            ) = loadParameters(parameterFile)
            # Filter the data based on the provided parameters
            snpProportion, snpProportionNoInterpolation, sampleMeta = filterData(
                inputCountsFile, inputMetaFile, minloci, minSample
            )

            # Embed the filtered data using UMAP
            # i.e., reduce the dimensionality of the data for clustering
            embedding = embedData(snpProportion, umapSeed)

            # Cluster the embedded data using DBSCAN
            db_communities = clusteringDBSCAN(
                snpProportion,
                sampleMeta,
                embedding,
                epsilon,
                filePrefix,
                admixedCutoff,
                outputDir=outputDir,
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
                outputDir=outputDir,
            )
        case "revisedClusteringForVarietyIdentification":
            # Load and validate he parameters from the JSON file
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
                outputDir,
            ) = loadParameters(parameterFile)

            # Filter the data based on the provided parameters
            snpProportion, snpProportionNoInterpolation, sampleMeta = filterData(
                inputCountsFile, inputMetaFile, minloci, minSample
            )

            snpProportion = run_pca_and_select_markers(
                snpProportion, n_keep=0.9, filePrefix=filePrefix, outputDir=outputDir
            )

            # Embed the filtered data using UMAP
            # i.e., reduce the dimensionality of the data for clustering
            embedding = embedData(snpProportion, umapSeed)

            # Cluster the embedded data using HDBSCAN
            db_communities = clusteringHDBSCAN(
                snpProportion,
                sampleMeta,
                embedding,
                3,
                filePrefix,
                admixedCutoff,
                outputDir=outputDir,
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
                outputDir=outputDir,
            )
        case "genotypeClusteringForPurityCalculation":
            # Eventually give these as a pass through option
            pca_marker_output = Path(outputDir).joinpath(
                f".ckpt1_{filePrefix}_pca_markers.pkl"
            )
            hdbscan_output_dir = Path(outputDir).joinpath(
                f".ckpt2_{filePrefix}_hdbscan_model.joblib"
            )
            genotype_matrix_dir = Path(outputDir).joinpath(
                f".ckpt3_{filePrefix}_genotype_matrix.pkl"
            )
            # Load and validate he parameters from the JSON file
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
                outputDir,
            ) = loadParameters(parameterFile)

            # Filter the data based on the provided parameters
            snpProportion, snpProportionNoInterpolation, sampleMeta = filterData(
                inputCountsFile,
                inputMetaFile,
                minloci,
                minSample,
                metaFilter={
                    "Sample Composition": "SinglePlant",
                    "reference": {"not_null": True},
                },
                # {
                #     # "Study": "Supply_Chain",
                #     # "Comments": "Single Plant Sample ",
                #     "Sample Composition": "SinglePlant",
                #     # "Sample Type": ["Target Material", "GRL"],
                # },
            )
            logging.info(f"Total # of samples is: {snpProportion.shape[1]}")
            logging.info(f"Initial # of markers is: {snpProportion.shape[0]}")
            logging.info(snpProportion.head())
            logging.info(sampleMeta.head())
            # future convert to scikit pipeline...
            # dim_reduction_pipeline = Pipeline([
            #     ('pca', PCA(n_components=50)),
            #     ('umap', UMAP(n_neighbors=15, min_dist=0.0, n_components=5))
            # ])
            logging.info(f"----- Running PCA -----")
            # Step 1, reduce with PCA.
            # Check to see if PCA output already exists, if so, use that
            # May need an override method incase of filter changes...
            if not pca_marker_output.exists():
                # Update method to select markers based on some criteria other than passing through
                # at least 0.9 of the variance.
                snpProportionPCA, pca = run_pca_and_select_markers(
                    snpProportion,
                    n_keep=0.9,
                    filePrefix=filePrefix,
                    outputDir=outputDir,
                )
                snpProportionPCA.to_pickle(pca_marker_output)
                joblib.dump(
                    pca, Path(outputDir).joinpath(f".pre1_{filePrefix}_pca.joblib")
                )
            else:
                logging.info(
                    f"Skipping PCA run as dimensions have already been reduced here.. {pca_marker_output}"
                )
                snpProportionPCA = pd.read_pickle(pca_marker_output)

            logging.info("----- PCA Complete -----")
            logging.info("----- Running UMAP -----")
            # Step 2, embed with UMAP
            # Embed the filtered data using UMAP
            # ..further reduce the dimensionality of the data for clustering
            embedding, umap_reducer = embedData(snpProportionPCA, umapSeed)
            joblib.dump(
                umap_reducer,
                Path(outputDir).joinpath(f".pre2_{filePrefix}_umap.joblib"),
            )
            logging.info(f"----- UMAP Complete -----")

            logging.info("----- Running HDBSCAN -----")
            # Step 3 Cluster with HDBSCAN
            if hdbscan_output_dir.exists():
                logging.info(f"Loading hdbscan from checkpoint... {hdbscan_output_dir}")
                hdbs = joblib.load(hdbscan_output_dir)
                db_communities = hdbs.labels_
            else:
                # Cluster the embedded data using HDBSCAN
                # Current implementation uses a minimum cluster size of 3, but this could be parameterized in the future.
                hdbs, db_communities = clusteringHDBSCAN(
                    snpProportionPCA,
                    sampleMeta,
                    embedding,
                    3,
                    filePrefix,
                    admixedCutoff,
                    outputDir=outputDir,
                )

                joblib.dump(hdbs, hdbscan_output_dir)
            logging.info("----- HDBSCAN Complete -----")
            # Label the samples based on the clustering results
            output, _ = labelSamples(
                snpProportionPCA,
                sampleMeta,
                db_communities,
                embedding,
                cutHeight,
                admixedCutoff,
                filePrefix,
                snpProportionNoInterpolation,
                parameterFile,
                outputDir=outputDir,
                probabilities=hdbs.probabilities_,
            )
            # If this is put in an interface in the future that is publically accessible,
            # consider using a more secure method of storing the output.
            output.to_pickle(
                Path(outputDir).joinpath(f"{filePrefix}_clustering_output.pkl")
            )

            # Get the cluster metrics.. These should output entropy, purity
            # of the cluster or group assignments specifically.
            group_assignments = output[["cluster", "short_name", "variety"]]

            # Remove the outlier group
            group_assignments.drop(
                group_assignments[group_assignments["cluster"] == -1].index,
                inplace=True,
            )

            sampleMeta["short_name"] = sampleMeta["short_name"].astype(str)
            group_to_cluster = pd.merge(
                group_assignments[["short_name", "cluster"]],
                sampleMeta[["short_name", "reference"]],
                on="short_name",
                how="inner",  # 'inner' keeps only matching rows in both DataFrames
            )
            group_to_cluster.rename(columns={"reference": "variety"}, inplace=True)
            group_to_cluster["variety"].fillna("Group 0", inplace=True)
            total_class_counts = group_to_cluster["variety"].value_counts()
            cluster_metrics = get_cluster_metrics(group_to_cluster, total_class_counts)

            cluster_metrics.to_csv(
                Path(outputDir).joinpath(f"{filePrefix}_cluster_metrics.csv")
            )

            plot_cluster_metrics(
                cluster_metrics,
                Path(outputDir).joinpath(f"{filePrefix}_cluster_metrics.png"),
            )

            match method:
                case 1:
                    logging.info(
                        "Running with method 1: calling genotypes from counts file."
                    )
                    ## Consensus part.. Generate a consensus genotype 0,1,2
                    if not genotype_matrix_dir.exists():
                        # Update to take/load from stored file
                        genotypeMatrix, sampleMeta = generateGenotypeMatrix(
                            inputCountsFile, inputMetaFile, minloci, minSample
                        )

                        genotypeMatrix.to_pickle(genotype_matrix_dir)
                    else:
                        logging.info(
                            f"Loading genotype matrix from file. {genotype_matrix_dir}"
                        )
                        genotypeMatrix = pd.read_pickle(genotype_matrix_dir)
                case 2:
                    # ##
                    # # Method #2 use dartseq called genotypes...
                    logging.info("Running with method 2: Using called genotypes.")
                    geno_call_file = "./wheat_full/merged_genotypes_master.csv"
                    logging.info(f"loading from file: {geno_call_file}")
                    genotypeMatrix = pd.read_csv(geno_call_file, index_col="MarkerName")
                    genotypeMatrix.replace("-", np.nan, inplace=True)
                case 3:
                    logging.info("Running with method 3: Using snpProportions")
                    # # Method #3 use the snpProportions from the PCA step
                    genotypeMatrix = snpProportionPCA
                    genotypeMatrix.sort_index(inplace=True)

            # Filter the genotype matrix by the identified markers in the PCA step
            filtered_df = genotypeMatrix[
                genotypeMatrix.index.isin(snpProportionPCA.index)
            ]

            # Group the lines by the group assignments and build the consensus.
            df_geno_t = filtered_df.T

            # Map the cluster assignments to the index using the short_name
            cluster_mapping = group_assignments.set_index("short_name")["cluster"]
            df_geno_t["cluster"] = df_geno_t.index.map(cluster_mapping)

            # Group by cluster and calculate the mode
            # We use a lambda to handle ties (it picks the first mode if multiple exist) and NaNs
            consensus_matrix_t = df_geno_t.groupby("cluster").agg(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan
            )
            # Create a mapping dictionary of {cluster_num: majority_class}
            majority_map = cluster_metrics["Majority_Class"].to_dict()

            # Map the majority class directly using the cluster column
            consensus_matrix_t.insert(
                0, "Majority_Class", consensus_matrix_t.index.map(majority_map)
            )

            consensus_matrix_t.to_csv(
                Path(outputDir).joinpath(f"{filePrefix}_consensus.csv")
            )

            # Transpose back: Rows are Markers, Columns are Clusters
            # consensus_matrix = consensus_matrix_t.T

            cluster_map = group_assignments.set_index("short_name")["cluster"].to_dict()
            purity_results = []

            for line_id, raw_geno_values in df_geno_t.iterrows():
                assigned_cluster = cluster_map.get(line_id)

                if (
                    pd.isna(assigned_cluster)
                    or assigned_cluster not in consensus_matrix_t.index
                ):
                    continue

                consensus_values = consensus_matrix_t.loc[assigned_cluster]

                # 4. Pass both Pandas Series into the purity method
                purity_pct = calculate_purity_by_name(raw_geno_values, consensus_values)

                # Store the results
                purity_results.append(
                    {
                        "short_name": line_id,
                        "assigned_cluster": assigned_cluster,
                        "purity_pct": purity_pct,
                    }
                )

            # 5. Convert back to a DataFrame
            df_purity_scores = pd.DataFrame(purity_results)

            df_purity_scores.to_csv(
                Path(outputDir).joinpath(f"{filePrefix}_consensus_purity_scores.csv")
            )

        case "purityCalculationFromConsensus":
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
                outputDir,
            ) = loadParameters(parameterFile)

            # Method #1
            # genotypeMatrix, sampleMeta = generateGenotypeMatrix(
            #     inputCountsFile, inputMetaFile, minloci, minSample
            # )

            # Method #2
            geno_call_file = "./wheat_full/merged_genotypes_master.csv"
            logging.info(f"loading from file: {geno_call_file}")
            genotypeMatrix = pd.read_csv(geno_call_file, index_col="MarkerName")
            genotypeMatrix.replace("-", np.nan, inplace=True)
            sampleMeta = pd.read_csv(inputMetaFile)

            # Method #3 Using the filterData functionality.. Might need to adjust a bit.. Not sure this is performing as expected..
            # genotypeMatrix, _, sampleMeta = filterData(
            #     inputCountsFile,
            #     inputMetaFile,
            #     minloci,
            #     minSample,
            # )
            genotype_matrix_t = genotypeMatrix.T

            consensusMatrix = pd.read_csv(
                Path(outputDir).joinpath(f"{filePrefix}_consensus.csv")
            )

            consensus_df = consensusMatrix.set_index("cluster")
            consensus_df.drop(columns="Majority_Class", inplace=True)

            # filter down the full genotype matrix to only the specified ones...
            excluded_cols = {"cluster", "Majority_Class"}
            marker_cols = [
                col for col in consensusMatrix.columns if col not in excluded_cols
            ]
            geno_cols = genotype_matrix_t.columns.to_list()
            columns_to_keep = [
                col for col in marker_cols if col != "short_name" and col in geno_cols
            ]

            genotype_matrix_filtered = genotype_matrix_t[columns_to_keep]

            meta_subset = sampleMeta[["short_name", "reference"]].drop_duplicates(
                subset=["short_name"]
            )
            # so that the merge functions properly...
            meta_subset["short_name"] = (
                meta_subset["short_name"].astype(str).str.strip()
            )

            cluster_map = meta_subset.set_index("short_name")["reference"].to_dict()
            purity_results = []
            CLUSTER_COL = "Majority_Class"

            cluster_dict = dict(
                zip(
                    consensusMatrix["cluster"].astype(int), consensusMatrix[CLUSTER_COL]
                )
            )

            genotype_matrix_filtered.sort_index(inplace=True)
            genotype_matrix_filtered.to_csv(
                Path(outputDir).joinpath(f"{filePrefix}_filter_genos_temp.csv")
            )

            purity_matrix = calculate_purity_matrix(
                genotype_matrix_filtered, consensus_df
            )

            # Find the index of the highest purity match for each raw genotype row
            best_match_indices = np.argmax(purity_matrix, axis=1)
            best_purities = np.max(purity_matrix, axis=1)
            best_consensus_ids = consensus_df.index[best_match_indices]

            # Build results DataFrame
            df_purity_scores = pd.DataFrame(
                {
                    "short_name": genotype_matrix_filtered.index,
                    "assigned_cluster": [
                        cluster_dict.get(int(cid)) for cid in best_consensus_ids
                    ],
                    "best_consensus_id": pd.Series(
                        consensus_df.index[best_match_indices], dtype="Int64"
                    ),
                    "purity_pct": best_purities,
                }
            )

            logging.info(f"Purity Dataframe:\n {df_purity_scores.head()}")

            df_purity_scores.to_csv(
                Path(outputDir).joinpath(
                    f"{filePrefix}_purity_scores_from consensus.csv"
                )
            )

            # total_class_counts = df_purity_scores["variety"].value_counts()
            # get_cluster_metrics()

        case _:
            raise ValueError(
                f"pipelineName must be one of:\n 'clusteringForVarietyIdentification',\n 'revisedClusteringForVarietyIdentification',\n genotypeClusteringForPurityCalculation.\n I got {pipelineName}"
            )
    logging.info(f"Pipeline {pipelineName} completed successfully.")
    logging.info(f"--------------------------")
    # return (
    #     snpProportion,
    #     snpProportionNoInterpolation,
    #     sampleMeta,
    #     embedding,
    #     db_communities,
    #     output,
    # )


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
