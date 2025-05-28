import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data for historical contracts and macroeconomic features
n_days = 100  # Example number of days
n_contracts = 2  # March and April contracts
n_features = 3  # Features like Prices, Open Interest, Volatility

# Create random data for each contract (March, April) and features (Prices, OI, Volatility)
historical_tensor = np.random.rand(n_days, n_contracts, n_features)

# Set up the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the X, Y, Z coordinates for the 3D plot
# X-axis will be 'Days'
x = np.arange(n_days)

# Y-axis will be 'Features'
y = np.arange(n_features)

# Create a meshgrid for X and Y
X, Y = np.meshgrid(x, y)

# Plot each contract's data as a surface
for i in range(n_contracts):
    # Transpose the data to match the X, Y shape
    Z = historical_tensor[:, i, :].T
    ax.plot_surface(X, Y, Z, label=f'Contract {i + 1}', alpha=0.7)

# Set labels and title
ax.set_title('Historical Contract Data Tensor 3D Visualization')
ax.set_xlabel('Days')
ax.set_ylabel('Features')
ax.set_zlabel('Values')

plt.show()
