import pandas as pd
import joblib


def train_model(x, y):
    """Read Saved Model and train It on Any data"""
    LinearModel = joblib.load(r'D:\All_Projects\Econometrics_Practice\models\linear_model.joblib')

    """Create Object"""
    lm = LinearModel()

    """Train On New Data"""
    if type(x) is pd.DataFrame:
        lm.fit(x=x, y=y)
        return lm
    else:
        raise ValueError("You only fit this model on pandas DataFrame")

    
