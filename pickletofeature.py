# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:11:58 2017

@author: user98
"""

import pickle
import numpy as np
import glob
import json

file=glob.glob("*.pickle")
with open(file[0], 'rb') as f:
         ddd=pickle.load(f)
         
a=np.zeros((len(file),ddd['feature'].shape[1]))
i=0
for fi in file:
    with open(fi, 'rb') as f:
         ddd=pickle.load(f)
         a[i,:]=ddd['feature']
         
np.save("feature.npy",a)
