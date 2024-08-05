import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Example dataset
# Replace this with your actual dataset
n_samples = 1000
np.random.seed(0)
group = np.random.randint(1, 20, size=n_samples)
x = np.random.normal(0, 1, size=n_samples)
y = np.random.binomial(1, 1 / (1 + np.exp(-(0.5 * x + 0.1 * group))), size=n_samples)

# Creating a DataFrame
data = pd.DataFrame({'Group': group, 'Predictor': x, 'Response': y})

# Features and target
X = data[['Group', 'Predictor']]
y = data['Response']

# Standardize the predictor variable
scaler = StandardScaler()
X['Predictor'] = scaler.fit_transform(X[['Predictor']])

# Define the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# GroupKFold cross-validation
group_kfold = GroupKFold(n_splits=5)
cross_val_scores = cross_val_score(rf_model, X, y, cv=group_kfold, groups=data['Group'], scoring='roc_auc')

# Train the model on the full dataset
rf_model.fit(X, y)

# Predictions
y_pred_prob = rf_model.predict_proba(X)[:, 1]
auc = roc_auc_score(y, y_pred_prob)

print(f'Cross-validated AUC scores: {cross_val_scores}')
print(f'Mean AUC: {np.mean(cross_val_scores)}')
print(f'Training AUC: {auc}')

# If you want to make predictions on new data:
# new_data = pd.DataFrame({'Group': [new_group_values], 'Predictor': [new_predictor_values]})
# new_data['Predictor'] = scaler.transform(new_data[['Predictor']])
# new_predictions = rf_model.predict_proba(new_data)[:, 1]
