import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
df = pd.read_csv('/content/harry_putter.csv')

# Step 2: Prepare features (X) and target (y)
features = [
    'temperature_2m', 'dewpoint_2m', 'relative_humidity_2m', 'precipitation',
    'rain', 'snowfall', 'snow_depth', 'wind_speed_10m', 'wind_speed_100m',
    'wind_direction_10m', 'wind_direction_100m', 'wind_gusts_10m',
    'surface_pressure', 'cloud_cover', 'cloud_cover_low',
    'cloud_cover_mid', 'cloud_cover_high',
    'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm',
    'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm',
    'soil_temperature_0_to_7cm', 'soil_temperature_7_to_28cm'
]

X = df[features]
y = df['caused_power_outage']

# Step 3: Train-test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_raw)
X_test_imputed = imputer.transform(X_test_raw)

# Step 5: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Step 6: Define all models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

# Step 7: Train and Evaluate each model
results = {}

for model_name, model in models.items():
    print(f"🔵 Training {model_name}...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        # Some models like SVM without probability=True don't have predict_proba
        y_proba = model.decision_function(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results[model_name] = {
        "Confusion Matrix": cm,
        "Accuracy": acc,
        "ROC-AUC": roc_auc,
        "Classification Report": classification_report(y_test, y_pred, output_dict=True)
    }

# Step 8: Display results
for model_name, metrics in results.items():
    print(f"\n\n==============================")
    print(f"📈 Results for {model_name}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='coolwarm',
                xticklabels=['No Outage (0)', 'Outage (1)'],
                yticklabels=['No Outage (0)', 'Outage (1)'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    report_df = pd.DataFrame(metrics['Classification Report']).transpose()
    print(report_df)
