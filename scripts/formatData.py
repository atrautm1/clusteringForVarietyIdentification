#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DArT output files into the correct format for this pipeline 
"""
import pandas as pd

def processCounts(inFile, outFile):
    '''
    Process a DArT tag counts file with seven header rows into the correct input format
    
    Args:
        inFile: name and path to counts file
        outFile: name and path for output file
    '''
    df = pd.read_csv(inFile, skiprows=7)
    notData = ['AlleleSequence','SNP','CallRate','OneRatioRef',
            'OneRatioSnp','FreqHomRef','FreqHomSnp','FreqHets',
            'PICRef','PICSnp','AvgPIC','AvgCountRef','AvgCountSnp',
            'RatioAvgCountRefAvgCountSnp']
    df = df.drop(columns=notData)
    df.to_csv(outFile, index=False)    
 
def processCountsSeq(inFile, outFile):  
    '''
    Process a DArT seq counts file with seven header rows into the correct input format
    
    Args:
        inFile: name and path to counts file
        outFile: name and path for output file
    '''  
    df = pd.read_csv(inFile, skiprows =7)
    notData = ['AlleleID', 'AlleleSequence', 'TrimmedSequence',
           'Chrom_Eragrostis_CogeV3', 'ChromPosTag_Eragrostis_CogeV3',
           'ChromPosSnp_Eragrostis_CogeV3', 'AlnCnt_Eragrostis_CogeV3',
           'AlnEvalue_Eragrostis_CogeV3', 'Strand_Eragrostis_CogeV3', 'SNP',
           'SnpPosition', 'CallRate', 'OneRatioRef', 'OneRatioSnp', 'FreqHomRef',
           'FreqHomSnp', 'FreqHets', 'PICRef', 'PICSnp', 'AvgPIC', 'AvgCountRef',
           'AvgCountSnp', 'RepAvg']
    df = df.drop(columns=notData)    
    df.rename(columns={'CloneID': 'MarkerName'}, inplace=True)
    df.to_csv(outFile, index=False)    