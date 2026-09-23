import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample data
# [Annual Income (in thousands), Spending Score (0-100)]
data = np.array([
    [15, 39],
    [16, 42],
    [17, 45],
    [18, 41],
    [20, 40],
    [25, 50],
    [28, 55],
    [30, 52],
    [32, 58],
    [35, 60],
    [60, 20],
    [65, 25],
    [70, 22],
    [72, 28],
    [75, 30],
    [80, 35],
    [85, 75],
    [88, 80],
    [90, 78],
    [95, 85]
])

# Create K-Means model
# We want to divide customers into 3 groups
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model
kmeans.fit(data)

# Get cluster labels
labels = kmeans.labels_

# Get cluster centers
centers = kmeans.cluster_centers_

# Display results
print("Cluster labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

# Predict the cluster for a new customer
new_customer = np.array([[50, 65]])
predicted_cluster = kmeans.predict(new_customer)

print("\nNew customer data:")
print(new_customer)

print("Predicted cluster:",
      predicted_cluster[0])

# Plot the clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    data[:, 0],
    data[:, 1],
    c=labels,
    cmap="viridis",
    s=100
)

# Plot cluster centers
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    color="red",
    marker="X",
    s=250,
    label="Cluster Centers"
)

plt.scatter(
    new_customer[:, 0],
    new_customer[:, 1],
    color="black",
    marker="*",
    s=250,
    label="New Customer"
)

plt.xlabel("Annual Income (₹ thousands)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means Clustering")
plt.legend()
plt.grid(True)
plt.show()