from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import BinaryEncoder
import pandas as pd
import joblib


def load_data(path_to_data):
    """Load Data From Source"""
    df = pd.read_csv(path_to_data)
    return df


def bool_to_object(df):
    """Convert bool Type To (str) Object Before Encodeing"""
    df['Fertilizer_Used'] = df['Fertilizer_Used'].astype(str)
    df['Irrigation_Used'] = df['Irrigation_Used'].astype(str)
    return df


"""In the target Feature there are some negative value which was the mistake of recording data
   Yield_tons_per_hectare can not be negative
"""
def remove_fault_data(df):
    df['Yield_tons_per_hectare'] = abs(df['Yield_tons_per_hectare'])
    return df


def categorical_encoding(df):
    """Initialize Encoders"""
    binary_encoder = BinaryEncoder(cols=['Soil_Type', 'Crop']) 
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    label_encoder = LabelEncoder()
    
    """Use LabelEncoder To Encode Fertilizer_Used"""
    df['Fertilizers_Used'] = label_encoder.fit_transform(df['Fertilizer_Used'])
    
    """Encode Two Columns ('Soil_Type', 'Crop') and Save it to Original Data frame"""
    encoded_df = binary_encoder.fit_transform(df[['Soil_Type', 'Crop']])
    df = pd.concat([df, encoded_df], axis=1)
    
    """Encode Fertilizer _Used Column"""
    one_hot_encoded_df = one_hot_encoder.fit_transform(df[['Irrigation_Used', 'Weather_Condition']])
    one_hot_encoded_features = one_hot_encoder.get_feature_names_out(['Irrigation_Used', 'Weather_Condition'])
    
    """Create New Data frame and Save it To Original Data frame"""
    new_one_hot_encoded_df = pd.DataFrame(data=one_hot_encoded_df, columns=one_hot_encoded_features)
    df = pd.concat([df, new_one_hot_encoded_df], axis=1)
    
    """Remove Columns Encoded"""
    df = df.drop(['Unnamed: 0', 'Region', 'Soil_Type', 'Crop', 'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest'], axis=1)
    
    """Arrange and Save Preprocessed Data Frame"""
    df = df[['Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Crop_0', 'Crop_1',
        'Crop_2', 'Rainfall_mm', 'Temperature_Celsius', 'Fertilizers_Used',  'Irrigation_Used_False', 'Irrigation_Used_True',
        'Weather_Condition_Cloudy', 'Weather_Condition_Rainy',
        'Weather_Condition_Sunny', 'Yield_tons_per_hectare']]
    
    df.to_csv(r'D:\All_Projects\Econometrics_Practice\data\preprocessed\new_crop_yield.csv', index=False)
    
    """Save Encoder Object For Future Use"""
    joblib.dump(binary_encoder, r'D:\All_Projects\Econometrics_Practice\models\binary_encoder.joblib')
    joblib.dump(one_hot_encoder, r'D:\All_Projects\Econometrics_Practice\models\one_hot_encoder.joblib')
    joblib.dump(label_encoder, r'D:\All_Projects\Econometrics_Practice\models\label_encoder.joblib')

    return df
