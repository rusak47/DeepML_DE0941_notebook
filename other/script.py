import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import torch

# 1. Generate sample data using sklearn
X, y = sklearn.datasets.make_classification(n_samples=100, n_features=2,
                                            n_redundant=0, n_informative=2,
                                            n_clusters_per_class=1, random_state=42)

# 2. Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)


# 3. Simple neural network with PyTorch
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


# 4. Train the model
model = SimpleNet()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor).squeeze()
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# 5. Make predictions and convert back to numpy for plotting
with torch.no_grad():
    predictions = model(X_tensor).squeeze().numpy()

# 6. Visualize results using matplotlib
plt.figure(figsize=(10, 4))

# Plot original data
plt.subplot(1, 2, 1)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(scatter)

# Plot predictions
plt.subplot(1, 2, 2)
scatter = plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis')
plt.title('Model Predictions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(scatter)

plt.tight_layout()
plt.show()

# Print some statistics using numpy
print(f"Prediction range: {np.min(predictions):.3f} to {np.max(predictions):.3f}")
print(f"Mean prediction: {np.mean(predictions):.3f}")

print('Hello World! All libraries we need are loaded!')