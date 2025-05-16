import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Step 1: Drop non-numeric columns
df_numeric = df.select_dtypes(include=['number']).copy()

# Drop rows with missing values
df_numeric = df_numeric.dropna()


# Step 2: Define features and target
X = df_numeric.drop('caused_power_outage', axis=1)
y = df_numeric['caused_power_outage']

# Step 3: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# -------------- No SMOTE --------------

# Step 4: Train-test split (without SMOTE)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Train Random Forest (without SMOTE)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Predictions (without SMOTE)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for AUC-ROC

# Step 7: Performance metrics (without SMOTE)
print("---- Without SMOTE ----")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_prob))

# Step 8: Plot Confusion Matrix (without SMOTE)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Outage", "Outage"], yticklabels=["No Outage", "Outage"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (Without SMOTE)')
plt.show()
#--------------with smote--------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Stratified to ensure equal distribution of classes
)

# Step 4: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Scale test set using the same scaler as train set

# Step 5: Apply SMOTE to the training set only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Step 6: Train Random Forest on the resampled training set
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Step 7: Predictions on the original test set (no SMOTE applied to test set)
y_pred_smote = model.predict(X_test_scaled)
y_pred_prob_smote = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for AUC-ROC

# Step 8: Performance metrics (with SMOTE on the training set only)
print("---- With SMOTE (on train set only) ----")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_smote))
print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_prob_smote))

# Step 9: Plot Confusion Matrix (with SMOTE on train set only)
conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)
sns.heatmap(conf_matrix_smote, annot=True, fmt="d", cmap="Blues", xticklabels=["No Outage", "Outage"], yticklabels=["No Outage", "Outage"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (With SMOTE on Train Set Only)')
plt.show()
