import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv('C:/Users/antl/Samsung_AI_Challenge/dataset/train.csv')
test_df = pd.read_csv('C:/Users/antl/Samsung_AI_Challenge/testset/test.csv')

# Features and target
X = train_df.iloc[:, 1:-1]  # Features (x_0 to x_10)
y = train_df.iloc[:, -1]  # Target (label)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)

# Train the model with early stopping
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=True)

# Predict on the test data
X_test = test_df.iloc[:, 1:]  # Test features
y_pred = model.predict(X_test)

# Identify top 10% of predicted values
threshold = np.percentile(y_pred, 90)
top_10_percent_mask = y_pred >= threshold

# Apply additional conditions to mark as 1 or 0
additional_conditions_mask = (
    (X_test['x_0'] < 0.95) |
    (X_test['x_1'] > -0.16) |
    (X_test['x_4'] < -0.35) |
    (X_test['x_5'] > -1.65) |
    (X_test['x_6'] > 0.52) |
    (X_test['x_7'] > -0.1)
)

zero_conditions_mask = (
    (X_test['x_3'] < 0.85) |
    (X_test['x_7'] < -0.175) |
    (X_test['x_8'] > 0.68) |
    (X_test['x_9'] < 0.25)
)

# Create the final submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_df.iloc[:, 0],  # Assuming first column is ID in test.csv
    'y': np.where(top_10_percent_mask & additional_conditions_mask, 1, 
                  np.where(zero_conditions_mask, 0, np.where(top_10_percent_mask, 1, 0)))
})

# Save the submission file
submission_df.to_csv('C:/Users/antl/Samsung_AI_Challenge/results/xgboost_combiined_submission_v3.csv', index=False)

print(f"Top 10% threshold: {threshold:.4f}")
print(f"Number of samples in top 10%: {sum(top_10_percent_mask)}")
