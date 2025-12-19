import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# 1. Load data
data_path = "data/flights.csv"
df = pd.read_csv(data_path)

# 2. Features and target
X = df.drop("price", axis=1)
y = df["price"]

# 3. Categorical feature encoding
categorical_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# 4. Build model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluate
mae = mean_absolute_error(y_test, y_pred)

# 9. Save results
os.makedirs("results", exist_ok=True)

with open("results/metrics.txt", "w") as f:
    f.write(f"Mean Absolute Error (MAE): {mae:.2f}")

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Flight Prices")
plt.savefig("results/prediction_plot.png")
plt.close()

print("Training complete.")
print(f"MAE: {mae:.2f}")

