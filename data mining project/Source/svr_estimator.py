import numpy as np
from sklearn.svm import SVR

from config import NAN, SVRParam
import pandas as pd



class SVREstimator:
    def __init__(self, data):
        self.data = data
        self.complete_rows, self.incomplete_rows = self.__extract_rows()

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

    # Estimate the missing values
    def estimate_missing_value(self):
        estimated_data = np.zeros((len(self.incomplete_rows),self.data.shape[1]))
        complete_data = np.array([self.data[x] for x in self.complete_rows])
        incomplete_data = np.array([self.data[x] for x in self.incomplete_rows])
        col_mean = np.nanmean(complete_data, axis=0)
        
        for column, value in enumerate(incomplete_data.transpose()):
            ind_rows = np.where(pd.isna(value))[0]
            if len(ind_rows) > 0:
                x_train = np.delete(complete_data.transpose(), column, 0).transpose()
                y_train = np.array(complete_data[:, column])

                model = SVR(gamma='scale', C=SVRParam.C, epsilon=SVRParam.EP)
                model.fit(x_train, y_train)

                x_test = []
                x_test_temp = np.delete(incomplete_data.transpose(), column, 0).transpose()
                #where_are_NaNs = np.isnan(x_test_temp)
                #x_test_temp[where_are_NaNs] = 0
                                    
                inds = np.where(np.isnan(x_test_temp))

                x_test_temp[inds] = np.take(col_mean, inds[1])

                for i in ind_rows:
                    x_test.append(x_test_temp[i])

                predicted = model.predict(x_test_temp)

                for i, v in enumerate(ind_rows):
                    estimated_data[v][column] = predicted[i]

        return estimated_data