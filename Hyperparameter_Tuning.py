from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from tqdm import tqdm

# Search space
param_dist = {
    'n_estimators': np.arange(100, 400, 50),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Generate sample of 30 random parameter combinations
param_list = list(ParameterSampler(param_dist, n_iter=30, random_state=42))

results = []
print("🔄 Starting manual hyperparameter evaluation...\n")

for i, params in enumerate(tqdm(param_list)):
    model = RandomForestClassifier(random_state=42, class_weight='balanced', **params)
    scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=3, n_jobs=-1)
    mean_score = scores.mean()

    print(f"▶️ Trial {i+1:02d} | AUC: {mean_score:.5f} | Params: {params}")
    results.append({'params': params, 'mean_auc': mean_score})

# Convert results to DataFrame and sort
results_df = pd.DataFrame(results).sort_values(by='mean_auc', ascending=False)
print("\n🏁 Best Result:")
print(results_df.iloc[0])

# Save results
results_df.to_csv("manual_hyperparameter_results.csv", index=False)
print("\n📁 Results saved to: manual_hyperparameter_results.csv")
