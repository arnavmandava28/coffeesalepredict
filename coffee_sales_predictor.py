import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# --- Step 1: Load & preprocess ---
file_path_1 = r"C:\Users\hyjin\Downloads\archive (6)\index_1.csv"
file_path_2 = r"C:\Users\hyjin\Downloads\archive (6)\index_2.csv"

df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)
df = pd.concat([df1, df2], ignore_index=True)

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', infer_datetime_format=True)
df = df.dropna(subset=['datetime'])
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour

df_grouped = df.groupby(['month', 'hour']).size().reset_index(name='sales_count')

X = df_grouped[['month', 'hour']]
y = df_grouped['sales_count'].values.reshape(-1, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add bias column
X_train_scaled = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_scaled = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# --- Step 2: Gradient Descent Implementation ---
m, n = X_train_scaled.shape

np.random.seed(42)
theta = np.random.randn(n, 1)  # initial parameters

alpha = 0.01  # learning rate
epochs = 1000

# Cost function: MSE
def compute_cost(X, y, theta):
    preds = X @ theta
    errors = preds - y
    return (1/(2*len(y))) * np.sum(errors**2)

# Starting statistics
print("\n--- Starting Statistics ---")
print(f"Initial theta (parameters): \n{theta.ravel()}")
print(f"Hyperparameters: learning rate = {alpha}, epochs = {epochs}")
print(f"Starting cost: {compute_cost(X_train_scaled, y_train, theta):.4f}")

# Training loop
cost_history = []
theta_history = []

for epoch in range(epochs):
    gradients = (1/m) * X_train_scaled.T @ (X_train_scaled @ theta - y_train)
    theta = theta - alpha * gradients
    cost = compute_cost(X_train_scaled, y_train, theta)
    cost_history.append(cost)
    
    # Save snapshots for plotting hypothesis evolution
    if epoch in [0, 1, 2, 5, 10, 50, 100, 500, 999]:
        theta_history.append((epoch, theta.copy()))

# Final statistics
print("\n--- Final Statistics ---")
print(f"Final theta (parameters): \n{theta.ravel()}")
print(f"Final cost: {cost_history[-1]:.4f}")

# --- Step 3: Plots ---
# 1. Training cost over epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), cost_history, linewidth=2)
plt.title("Training Cost (MSE) over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig("cost_over_epochs.png")
print("Saved cost_over_epochs.png")

# 2. Hypothesis evolution (prediction lines)
plt.figure(figsize=(10, 6))
months = np.linspace(X['month'].min(), X['month'].max(), 12)
hours_fixed = np.full_like(months, 12)

X_line = np.c_[months, hours_fixed]
X_line_scaled = scaler.transform(X_line)
X_line_scaled = np.c_[np.ones((X_line_scaled.shape[0], 1)), X_line_scaled]

colors = plt.cm.viridis(np.linspace(0,1,len(theta_history)))
for color, (epoch, theta_snapshot) in zip(colors, theta_history):
    preds = X_line_scaled @ theta_snapshot
    plt.plot(months, preds, color=color, label=f"Epoch {epoch}")

plt.scatter(X['month'], y, c='gray', alpha=0.5, label="Data")
plt.title("Hypothesis Evolution (Predicted Sales vs Month at Hour=12)")
plt.xlabel("Month")
plt.ylabel("Predicted Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hypothesis_evolution.png")
print("Saved hypothesis_evolution.png")

# 3. Theta evolution over epochs
theta_array = np.array([t for _, t in theta_history]).squeeze()
plt.figure(figsize=(8,5))
for i, name in enumerate(['Bias', 'Month', 'Hour']):
    plt.plot([e for e,_ in theta_history], theta_array[:,i], label=name, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Theta Value")
plt.title("Theta Parameters Evolution over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("theta_evolution.png")
print("Saved theta_evolution.png")



# --- Step 4: Make a New Prediction ---
def predict_sales(month, hour, scaler, theta):
    X_new = np.array([[month, hour]])
    X_new_scaled = scaler.transform(X_new)
    X_new_scaled = np.c_[np.ones((X_new_scaled.shape[0], 1)), X_new_scaled]
    prediction = X_new_scaled @ theta
    return int(max(0, round(prediction[0, 0])))

predicted_sales = predict_sales(7, 11, scaler, theta)
print(f"\nPredicted coffee sales for month=7, hour=11 → {predicted_sales} units")
