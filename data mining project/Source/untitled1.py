# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:27:49 2022

@author: suraj
"""
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import cmeans

incomplete_data = pd.read_csv("C:\\Users\\suraj\\Downloads\\Studies\\Data Mining\\Incomplete datasets\\Data 1\\Data_1_AE_1%.csv", sep=',',header=None)

center, u, u0, d, _, _, _ = cmeans(incomplete_data.dropna(axis=0).T, 5, 3, error=0.001,maxiter = 100)

center1, u1, u01, d1, _, _, _ = cmeans(incomplete_data, 5, 3, error=0.001,maxiter = 100)

nan_rows = incomplete_data[incomplete_data.isnull().any(axis=1)]

new_val = fuzz.cluster.cmeans_predict(incomplete_data.fillna(value=0).T, center, 3,error=0.001,maxiter = 100)
