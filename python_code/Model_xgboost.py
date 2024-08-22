"""
 * Project : Samsung AI Challenge -  Creating and testing AI models
 * Program Purpose and Features :
 * - load testset and Predict labels
 * Author : HG Kim
 * First Write Date : 2024.08.22
 * ============================================================
 * Program history
 * ============================================================
 * Author    		Date		    Version		History
    HG Kim          2024.08.22      xgboost.v1  xgboost를 활용한 모델 생성
    HG Kim          2024.08.22      xgboost.v2  xgboost의 파라미터 값 조정
    HG Kim          2024.08.22      xgboost.v3  xgboost의 학습 에폭을 조정 (Early Stopping 방법을 사용해 학습률이 저조하면 중단.)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv('C:/Users/antl/Samsung_AI_Challenge/dataset/train.csv')
test_df = pd.read_csv('C:/Users/antl/Samsung_AI_Challenge/testset/test.csv')

# Features and target
X = train_df.iloc[:, 1:-1]  # Features (x_0 to x_10)
y = train_df.iloc[:, -1]  # Target (label)

# Split the data into training and validation sets for better performance monitoring
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)

# Train the model
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

# Predict on the test data
X_test = test_df.iloc[:, 1:]  # Test features
y_pred = model.predict(X_test)

# Identify top 10% of predicted values
threshold = np.percentile(y_pred, 90)
top_10_percent_mask = y_pred >= threshold

# Results
top_10_percent_predictions = y_pred[top_10_percent_mask]
print(f"Top 10% threshold: {threshold:.4f}")
print(f"Number of samples in top 10%: {sum(top_10_percent_mask)}")

# Create submission file
submission_df = pd.DataFrame({
    'ID': test_df.iloc[:, 0],  # Assuming first column is ID in test.csv
    'y': np.where(top_10_percent_mask, 1, 0)  # 1 for top 10%, 0 otherwise
})

submission_df.to_csv('C:/Users/antl/Samsung_AI_Challenge/results/xgboost_submission_v1.csv', index=False)
