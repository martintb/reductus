"""
File loaders for NCNR 6-column ABS SANS format
"""
from __future__ import print_function
import sys
import os
import struct
import time
import math
from collections import namedtuple

try:
    # pylint: disable=unused-import
    from typing import Tuple, Dict, List, Optional, Sequence, Any, Iterator, AnyStr
    # pylint: enable=unused-import
    ignore = 0  # keep pycharm happy
except ImportError:
    pass

import numpy as np

from .sansdata import SANSDataIQ

def readNCNRABS(fpath):
    '''Read ASCII .ABS files and, if possible, extract instrument configuration 
    
    This method attempts to flexibly handle the two types of ABS files generated from the NCNR 
    Igor macros: single-configuration reduced data and multi-configuration combined files. ABS 
    files have a variable number of header lines and then 6 columns of data: q, I, dI, dq, 
    meanQ, ShadowFactor.
    
    Arguments
    ---------
    fpath: str or pathlib.Path
        Full path with filename to an ABS file
    
    Returns:
    --------
    sansIQ: SansIQData
        A dictionary with keys: q, I, dI, dq, meanQ, ShadowFactor
    
    '''
    config_dict={}
    with open(fpath,'r') as f:
        skiprows = 1
        while True:
            line = f.readline()
            skiprows+=1
            
            if 'LABEL:' in line:
                label = line.split(':')[-1]
            elif 'MON CNT' in line:
                keys1 = ['MONCNT','LAMBDA','DET ANG','DET DIST','TRANS','THICK','AVE','STEP']
                values1 = f.readline().split()
                keys2 = ['BCENT(X,Y)','A1(mm)','A2(mm)','A1A2DIST(m)','DL/L','BSTOP(mm)','DET_TYP']
                _ = f.readline()
                values2 = f.readline().split()
                config_dict = {k:v for k,v in zip(keys1+keys2,values1+values2)}
                skiprows += 2
            elif 'The 6 columns are' in line:
                break
            
    Q,I,dI,dQ,meanQ,ShadowFactor = np.loadtxt(fpath,skiprows=skiprows).T
    sansIQ = SANSDataIQ(I=I,dI=dI,Q=Q,dQ=dQ,meanQ=meanQ,ShadowFactor=ShadowFactor,metadata=config_dict)
    return sansIQ
