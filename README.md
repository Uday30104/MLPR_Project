# Power Outage Prediction Based on Weather Events

This project predicts power outages caused by different weather events using machine learning. It includes two modeling approaches, along with preprocessing and matching scripts to handle weather and outage data.

## Project Files and Descriptions

- **ML_Model.py**  
  **Attempt 1**: Trains a single Random Forest model using data from all weather events combined.

- **ML Models.py**  
  **Attempt 2**: Trains separate Random Forest models for each type of weather event (hail, wind, snow, etc.).

- **Hyperparameter_Tuning.py**  
  Script for hyperparameter optimization to improve model performance.

- **Model_comparison.py**  
  Compares different models to select the best one based on performance metrics.

- **long_lat_extract.ipynb**  
  Extracts latitude and longitude based on state information for events lacking geographical coordinates.

- **outage_matching_code.ipynb**  
  Matches outage records with the corresponding weather events.

- **weather_var_extract.ipynb**  
  Extracts relevant weather variables from open-meteo api for use in modeling.

## Notes

- Both attempts use Random Forest classifier.

