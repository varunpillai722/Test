import numpy as np

from fcm_estimator import FCMeansEstimator
from ga import Genetic_Algo
from svr_estimator import SVREstimator
import pandas as pd




def compute_rmse(a,b):
    return np.sqrt(np.square(np.subtract(a, b)).sum().sum())/np.sqrt(np.square(a).sum().sum())

def fillMissingValues(incomplete_data, imputedData, incomplete_rows):
    
    for i in range(len(incomplete_rows)):
        incomplete_data.iloc[incomplete_rows.iloc[i]] = incomplete_data.iloc[incomplete_rows.iloc[i]].fillna(imputedData.iloc[i])
        
    return incomplete_data
        
# Run the algorithm for estimating
def run(dataset_name):

    data = np.array(pd.read_csv("C:\\Users\\suraj\\Downloads\\Studies\\Data Mining\\Complete datasets\\Data_1.csv", sep=','))



    # Make FCM and SVR model
    incomplete_data = np.array(pd.read_csv("C:\\Users\\suraj\\Downloads\\Studies\\Data Mining\\Incomplete datasets\\Data 1\\Data_1_AE_1%.csv", sep=',',header=None))
    
    svr_estimator = SVREstimator(data=incomplete_data)
    #y = svr_estimator.estimate_missing_value()
    ga = Genetic_Algo(svr_estimator,incomplete_data)

    while True:
        
        c, m = ga.run()
        fcm_estimator = FCMeansEstimator(c=c, m=m, data=incomplete_data)
        x = fcm_estimator.estimate_missing_values()
        #error = np.power(x - y, 2).sum()
        imputedData = fillMissingValues(pd.DataFrame(incomplete_data),pd.DataFrame(x), pd.DataFrame(svr_estimator.incomplete_rows))
        #incomplete_data.fillna()
        rmse = compute_rmse(data, imputedData)
        print('\nRMSE  : ' + str(rmse))
        if(rmse < 1):
            break




if __name__ == '__main__':
    # Use the name of the database as input
    run('Iris')