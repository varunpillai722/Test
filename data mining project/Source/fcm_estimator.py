import numpy as np
from skfuzzy import cmeans

from config import NAN, FCMParam
import pandas as pd
import copy

class FCMeansEstimator:
    def __init__(self, c, m, data):
        self.c = c
        self.m = m
        self.data = data
        self.complete_rows, self.incomplete_rows = self.__extract_rows()
        self.centers = self.getCenters()
        

    # Extract complete and incomplete rows
    def __extract_rows(self):
        rows, columns = len(self.data), len(self.data[0])
        complete_rows, incomplete_rows = [], []
        for i in range(rows):
            flag = False
            for j in range(columns):
                if pd.isna(self.data[i][j]):
                    flag = True
                    break
                
            if(flag):
                incomplete_rows.append(i)
            else:
                complete_rows.append(i)

        return np.array(complete_rows), np.array(incomplete_rows)

    def getCenters(self):
        complete_data = np.array([self.data[x] for x in self.complete_rows])
        centers, _, _, _, _, _, _ = cmeans(data=complete_data.transpose(), c=self.c, m=self.m, error=FCMParam.ERROR,
                                           maxiter=FCMParam.MAX_ITR, init=None)
        return centers
        
    # Estimate the missing values
    def estimate_missing_values(self):
        
        estimated_data = np.zeros((len(self.incomplete_rows),self.data.shape[1]))
        #complete_data = np.array([self.data[x] for x in self.complete_rows])
        #centers, _, _, _, _, _, _ = cmeans(data=complete_data.transpose(), c=self.c, m=self.m, error=FCMParam.ERROR,
                                           #maxiter=FCMParam.MAX_ITR, init=None)
        
        mean_arr = copy.deepcopy(self.data)
        col_mean = np.nanmean(mean_arr, axis=0)
        col_mean = np.nan_to_num(col_mean)
        inds = np.where(np.isnan(mean_arr))

        mean_arr[inds] = np.take(col_mean, inds[1])

        # Calculate distance between two points based on euclidean distance
        def calculate_distance(data_1, data_2):
            X = data_1 - data_2
            where_are_NaNs = np.isnan(X)
            X[where_are_NaNs] = 0
            #np.nan_to_num(nparr, nan=np.nanmean(nparr))
            return np.linalg.norm(X)

        # Calculate the membership value for given point
        def calculate_membership(dist_matrix, distance, m):
            numerator = np.power(distance, -2 / (1 - m))
            denominator = np.array([np.power(x, -2 / (1 - m)) for x in dist_matrix]).sum()
            return numerator / denominator
        
        row=0

        for i in self.incomplete_rows:
            estimated = 0
            dist, membership_value = [], []
            miss_ind = np.where(pd.isna(self.data[i])  )[0]
            for j in miss_ind:
                for center in self.centers:
                    dist.append(calculate_distance(data_1=np.delete(np.array(center), j),
                                                   data_2=np.delete(np.array(mean_arr[i]), j)))
                for d in dist:
                    membership_value.append(calculate_membership(dist, d, self.m))

                for k in range(self.c):
                    estimated += self.centers[k][j] * membership_value[k]

                estimated_data[row][j] = estimated
            row = row+1

        return np.array(estimated_data)