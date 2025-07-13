import pandas as pd
from linear_regression.linear_regression_model import LinearRegression
from src_code.preprocessing import load_data, bool_to_object, remove_fault_data, categorical_encoding
import joblib

"""Main Program"""
def main():
    """Get Raw Data"""
    df = load_data(r'D:\All_Projects\Econometrics_Practice\data\raw_data\crop_yield.csv')
    df = bool_to_object(df)
    df = remove_fault_data(df)
    df = categorical_encoding(df)
    
    """X_data => Independent Features (Response Variables)"""
    X =  df[['Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Crop_0', 'Crop_1',
        'Crop_2', 'Rainfall_mm', 'Temperature_Celsius', 'Fertilizers_Used', 
        'Irrigation_Used_False', 'Irrigation_Used_True',
        'Weather_Condition_Cloudy', 'Weather_Condition_Rainy',
        'Weather_Condition_Sunny']]
    
    """Dependent Variable"""
    y = df['Yield_tons_per_hectare']
    
    """Read and Fit Data To Model"""
    model = LinearRegression()
    model.fit(x=X, y=y)
    
    """Prompting user To Enter Independent Features Values. 
        Transform Input Data In Order To Get Some Prediction"""
    input_from_user = {
        'Soil_Type': input("Choose Soil Type in ['Chalky', 'Loam', 'Peaty', 'Clay', 'Silt', 'Sandy']: "),
        'Crop': input("Choose Crop in ['Maize', 'Cotton', 'Barley', 'Wheat', 'Soybean', 'Rice']: "),
        'Rainfall_mm' : int(input("Enter RainFall (mm): ")), 
        'Temperature_Celsius': int(input("Enter Temperature Celsius: ")), 
        'Fertilizer_Used': input("Is Fertilizer_Used? [True, False]: "),
        'Irrigation_Used': input("Is Irrigation_Used? [True, False]: "),
        'Weather_Condition': input("Choose Weather Condition in ['Rainy', 'Cloudy', 'Sunny']: ")
    }
  
    
    """Read Encoder Models"""
    binary_encoder = joblib.load(r'D:\All_Projects\Econometrics_Practice\models\binary_encoder.joblib')
    one_hot_encoder = joblib.load(r'D:\All_Projects\Econometrics_Practice\models\one_hot_encoder.joblib')
    label_encoder = joblib.load(r'D:\All_Projects\Econometrics_Practice\models\label_encoder.joblib')
    
    """Get new Df and Transform to Numerical Values """
    df_soil_crop = pd.DataFrame({'Soil_Type': [input_from_user.get('Soil_Type', 'Clay')],
                                 'Crop': [input_from_user.get('Crop', 'Maize')]})
    df_soil_crop = binary_encoder.transform(df_soil_crop)
    
    df_irr_weather = pd.DataFrame({'Irrigation_Used': [input_from_user.get('Irrigation_Used', 'True')],
                                    'Weather_Condition': [input_from_user.get('Weather_Condition', 'Rainy')]})
    df_irr_weather = one_hot_encoder.transform(df_irr_weather)
    
    """Transform Fertilizer_Used Value To Numeric"""
    df_fer = label_encoder.transform([input_from_user.get('Fertilizer_Used', 'True')])
    
    """Combine df_irr_weather and df_fer 
        and Rainfall_mm and Temperature_Celsius Values 
        In Single DataFrame"""
    r_t_irr_weat = pd.DataFrame({
                                   'Rainfall_mm': [input_from_user.get('Rainfall_mm')], 
                                   'Temperature_Celsius': [input_from_user.get('Temperature_Celsius')], 
                                   'Fertilizers_Used': df_fer,
                                   'Irrigation_Used_False': [df_irr_weather[0][0]], 
                                   'Irrigation_Used_True': [df_irr_weather[0][1]],
                                   'Weather_Condition_Cloudy': [df_irr_weather[0][2]], 
                                   'Weather_Condition_Rainy': [df_irr_weather[0][3]],
                                   'Weather_Condition_Sunny': [df_irr_weather[0][4]]
                                   })

    """Get Full Input as Data Frame"""
    predicting_input = pd.concat([df_soil_crop, r_t_irr_weat], axis=1)
    
    """Predict On Input"""
    print(f"Yield Per Hectare Prediction: {model.predict(x=list(predicting_input.iloc[0]))}")
    
    
if __name__ == "__main__":
    main()