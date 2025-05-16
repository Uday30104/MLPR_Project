#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
    recall_score
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Load and Preprocess Data
df = pd.read_csv('complete_data.csv')

# Parse datetime
df['event_datetime'] = pd.to_datetime(df['event_datetime'], format='%d/%m/%y %H:%M')

# Add temporal features (kept as basic preprocessing)
df['month'] = df['event_datetime'].dt.month
df['year'] = df['event_datetime'].dt.year
df['hour'] = df['event_datetime'].dt.hour

# Handle missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols = ['state', 'county']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Standardize event_type
df['event_type'] = df['event_type'].str.lower().str.strip()

# Fix tornado mislabeling (assuming '1' and '0' are tornado events)
df['event_type'] = df['event_type'].replace({'1': 'tornado', '0': 'tornado'})

# Define features (only raw features plus month, year, hour)
feature_cols = [
    'temperature_2m', 'dew_point_2m', 'relative_humidity_2m', 'precipitation', 'rain', 'snowfall',
    'snow_depth', 'windspeed_10m', 'windspeed_100m', 'winddirection_10m', 'winddirection_100m',
    'windgusts_10m', 'surface_pressure', 'cloudcover', 'cloudcover_low', 'cloudcover_mid',
    'cloudcover_high', 'soil_temperature_0_to_7cm', 'soil_temperature_7_to_28cm',
    'soil_temperature_28_to_100cm', 'soil_temperature_100_to_255cm', 'soil_moisture_0_to_7cm',
    'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm',
    'month', 'year', 'hour', 'state', 'county', 'event_latitude', 'event_longitude'
]
target_col = 'caused_power_outage'

# Scale features
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# 2. Split Data by Event Type
event_types = ['high_wind', 'heavy_snow', 'tornado', 'thunderstorm', 'hail']
event_dfs = {event: df[df['event_type'] == event].copy() for event in event_types}

# 3. Hyperparameter Space for Random Forest
param_dist = {
    'n_estimators': np.arange(100, 500, 50),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# 4. Train and Evaluate Models
models = {}
recall_scorer = make_scorer(recall_score, pos_label=1)
all_y_true, all_y_pred, all_y_proba = [], [], []

for event in event_types:
    print(f"\nTraining model for {event}...")
    
    # Get data for this event
    event_df = event_dfs[event]
    
    # Check if data exists
    if event_df.empty:
        print(f"Warning: No data available for {event}. Skipping model training.")
        continue
    
    X = event_df[feature_cols]
    y = event_df[target_col]
    
    # Ensure enough data for splitting
    if len(event_df) < 5:
        print(f"Warning: Insufficient data for {event} ({len(event_df)} samples). Skipping model training.")
        continue
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        print(f"Error splitting data for {event}: {e}. Skipping model training.")
        continue
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Initialize Random Forest classifier
    rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf_clf,
        param_distributions=param_dist,
        n_iter=50,
        scoring=recall_scorer,
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit model
    random_search.fit(X_train, y_train)
    
    # Store best model
    best_model = random_search.best_estimator_
    models[event] = best_model
    
    # Evaluate on test set with threshold tuning
    y_proba = best_model.predict_proba(X_test)[:, 1]
    threshold = 0.3  # Lower threshold to increase recall
    y_pred = (y_proba >= threshold).astype(int)
    
    # Store for combined metrics
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)
    
    print(f"\nBest Parameters for {event}:", random_search.best_params_)
    print(f"Classification Report for {event}:\n", classification_report(y_test, y_pred))
    print(f"AUC-ROC Score for {event}:", roc_auc_score(y_test, y_proba))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix for {event}")
    plt.show()

# 5. Combined Evaluation
if all_y_true:
    print("\n=== COMBINED EVALUATION ===")
    
    # Combined AUC-ROC
    combined_auc = roc_auc_score(all_y_true, all_y_proba)
    print(f"Combined AUC-ROC Score: {combined_auc:.2f}")
    
    # Combined Classification Report
    print("\nCombined Classification Report:\n")
    print(classification_report(all_y_true, all_y_pred, target_names=['no_outage', 'outage']))
    
    # Combined Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no_outage', 'outage'])
    disp.plot(cmap='Blues')
    plt.title('Combined Confusion Matrix')
    plt.show()
else:
    print("\nNo models were trained, so combined evaluation is not possible.")

# 6. Prediction Pipeline
def predict_power_outage(new_data):
    predictions = []
    
    if 'event_type' not in new_data.columns:
        raise ValueError("New data must include 'event_type' column")
    
    # Scale new data
    new_data[feature_cols] = scaler.transform(new_data[feature_cols])
    
    for _, row in new_data.iterrows():
        event = row['event_type'].lower().strip()
        if event not in models:
            predictions.append({"error": f"No model trained for event type: {event}"})
            continue
        
        X_new = row[feature_cols].values.reshape(1, -1)
        model = models[event]
        proba = model.predict_proba(X_new)[0][1]
        pred = int(proba >= 0.3)
        
        predictions.append({
            'event_type': event,
            'predicted_outage': pred,
            'probability': proba
        })
    
    return predictions

# 7. Save Models
for event, model in models.items():
    joblib.dump(model, f"rf_model_{event}.pkl")

# 8. Check Event Type Distribution
print("\nEvent Type Distribution:")
print(df['event_type'].value_counts())

# 9. Inspect Weather Features
for event in event_types:
    print(f"\nWeather Feature Stats for {event}:")
    if event in ['high_wind', 'tornado']:
        print(df[df['event_type'] == event][['windspeed_10m', 'windgusts_10m']].describe())
    elif event == 'heavy_snow':
        print(df[df['event_type'] == event][['snowfall', 'snow_depth']].describe())
    elif event in ['thunderstorm', 'hail']:
        print(df[df['event_type'] == event][['precipitation', 'rain']].describe())


# In[ ]:




