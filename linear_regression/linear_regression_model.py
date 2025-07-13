import pandas as pd
import numpy as np
import joblib

class LinearRegression:
    """Construct Method To Initialize Class Variables"""
    def __init__(self,):
        self.__x_data = None
        self.__y_data = None
        self.__x_mean = None
        self.__y_mean = None
        self.__beta_0 = None
        self.__beta_1 = None
        self.__y_predicted = None
        self.sum_squared_error = 0
        self.__matrix_data = None
        self.__Beta_Hat_Vector = None
    
    """Fit Method Used To Train This Model On Your Data"""        
    def fit(self, x=None, y=None):
        
        self.__x_data = x
        self.__y_data = y
        
        self.__x_mean = np.mean(self.__x_data)
        self.__y_mean = np.mean(self.__y_data)
        
        """Validate Input Data"""
        if type(self.__x_data) is not pd.DataFrame:
            raise TypeError(f"Object Of Type {type(self.__x_data)} is not Pandas Data Frame ")
        
        if len(self.__x_data.columns) < 1:
            raise ValueError("1 or 2 Columns Expected In Data Frame But 0 was given")
        
        if len(self.__x_data.index) != len(self.__y_data.index): 
            raise ValueError("length of X must be Equal to The length of Y")
        
        """Simple Linear Regression"""
        if len(self.__x_data.columns) == 1:
            
            """Calculate beta_hat_1 numerator and denominator"""
            beta_hat_1_numer = 0
            beta_hat_1_den = 0
            
            beta_hat_1_numer += (self.__x_data.iloc[0:, 0] - self.__x_mean) * (self.__y_data.iloc[0:]  - self.__y_mean) 
            beta_hat_1_den += (self.__x_data.iloc[0:, 0] - self.__x_mean)**2 
            
            self.__beta_1 = sum(beta_hat_1_numer) / sum(beta_hat_1_den)
            self.__beta_0 = self.__y_mean - (self.__beta_1 * self.__x_mean)
            
            """Calculate Predicted Values"""
            self.__y_predicted = self.__beta_0 + (self.__beta_1 * self.__x_data.iloc[0:, 0]) 
            
            """Calculate Sum Of Squared Errors"""
            for index_i in range(len(self.__x_data.index)):
                self.sum_squared_error += self.__y_data.iloc[index_i].astype(float) - self.__y_predicted.iloc[index_i].astype(float)

        elif len(self.__x_data.columns) > 1:
            """Multi-Linear Regression"""
            self.__matrix_data = []
            """Get 1 intercep for design matrix and append each column data ass row"""
            ones_intercep = []
            for i in range(len((self.__x_data.index))):
                ones_intercep.append(1)
            """Append Above Intercep To Matrix Data"""
            self.__matrix_data.append(ones_intercep) 
            
            """Create Array For Each Column Data"""
            for i in range(len(self.__x_data.columns)):
                col = np.array(self.__x_data.iloc[0:, i])
                self.__matrix_data.append(col)
            self.__matrix_data = np.matrix(self.__matrix_data).T
   
            """Transposed X Data"""
            matrix_data_transpose = np.linalg.matrix_transpose(self.__matrix_data)
            
            """Get Vector Like Column Containing Y target Observed Values."""
            Y_observed = np.matrix(self.__y_data).T
            
            """Get Matrix Product Between X Data Transposed and  X Data"""
            X_T_X = np.linalg.matmul(matrix_data_transpose, self.__matrix_data)
        
            """Get X_T_X Inverse"""
            X_T_X_inverse = np.linalg.inv(X_T_X)

            """Get X Data and Y Vector"""
            X_T_y = np.linalg.matmul( matrix_data_transpose , Y_observed)
            
            """Calculate Coefficients:
            Column Containing Unknown Population Parameters
            """
            self.__Beta_Hat_Vector = np.linalg.matmul(X_T_X_inverse, X_T_y)
         
    def predict(self, x):
        if self.__x_data is not None:
            if len(self.__x_data.columns) == 1:
                if self.__beta_0 is None or self.__beta_1 is None:
                    raise ValueError("Can't Predict Before Fitting The Model, Please Call fit() method before you Predict")
                predicted = [self.__beta_0 + self.__beta_1 * row for row in x]
                return np.array(predicted)
            
            if len(self.__x_data.columns) > 1:
                if self.__Beta_Hat_Vector is None:
                    raise ValueError("Can't Predict Before Fitting The Model, Please Call fit() method before you Predict")
                
                """Get New Data Frame to Predict On"""
                if type(x[0]) is list:
                    for i in range(len(x)):
                        try:
                            x[i].insert(0, 1)
                        except AttributeError:
                            raise ValueError("Array Containg data Were Expected")
                else:
                    x.insert(0, 1)
                new_prediction = x
                """Calculate Y Predicted Values From Beta_Hat_Vector Coefficients"""
                predicted =  (new_prediction * self.__Beta_Hat_Vector)
                
                """Return Predicted Values as Float in List"""
                all_predicted = []
                for result in list(np.array(predicted)):
                    all_predicted.append(float(result[0]))
                return all_predicted
        elif self.__x_data is None:
            raise AttributeError("Can't Predict Before Train The Model")


"""Save Model Using Joblib"""
path_to_model = r'D:\All_Projects\Econometrics_Practice\models\linear_model.joblib'
joblib.dump(LinearRegression, path_to_model)