import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')


class MyRegressionKNN:

    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))    

    def manhattan_distance(self, p1, p2):
        return np.sum(np.abs((p1 - p2)))    

    def __init__(self, k, dist_metric='euclidean'): 
        self.k = k
        self.dist_metric = dist_metric
   
    def predict(self, X_train, y_train, X_test):
        predictions = []

        for row_test in X_test:
            distances_targets = []

            for row_X_train, target in zip(X_train, y_train):
                if self.dist_metric == 'euclidean':
                    distance = self.euclidean_distance(row_X_train, row_test)
                elif self.dist_metric == 'manhattan':
                    distance = self.manhattan_distance(row_X_train, row_test)

                distances_targets.append((distance, target)) 

            k_sorted = sorted(distances_targets, key=lambda x: x[0])[:self.k]

            # MAIN CHANGE → mean instead of voting
            targets = [target for _, target in k_sorted]
            prediction = np.mean(targets)

            predictions.append(prediction)

        self.predictions = np.array(predictions)
        return self.predictions  


# =========================
# DATA (continuous values)
# =========================
X = np.array([
    [1, 2, 1], [2, 3, 2], [3, 3, 1],
    [6, 5, 6], [7, 7, 7], [8, 6, 5],
    [5, 2, 3], [6, 3, 4], [7, 2, 5]
])

# continuous targets (NOT classes anymore)
y = np.array([10, 12, 11, 30, 35, 32, 20, 22, 25])

new_point = np.array([[5, 4, 4]])
k = 3

# =========================
# MODEL
# =========================
model = MyRegressionKNN(k=3)
prediction = model.predict(X, y, new_point)[0]

# =========================
# FIND NEIGHBORS (same logic)
# =========================
distances = []
for point, target in zip(X, y):
    d = np.linalg.norm(point - new_point[0])
    distances.append((d, point, target))

distances.sort(key=lambda x: x[0])
neighbors = distances[:k]

# =========================
# PLOT
# =========================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use color map (continuous!)
scatter = ax.scatter(
    X[:, 0], X[:, 1], X[:, 2],
    c=y,
    cmap='viridis',
    s=50
)

# prediction point
ax.scatter(
    new_point[0][0], new_point[0][1], new_point[0][2],
    color='white',
    s=220,
    edgecolor='black',
    linewidth=1.5,
    label=f"Prediction: {prediction:.2f}"
)

# lines to neighbors
for dist, point, target in neighbors:
    ax.plot(
        [new_point[0][0], point[0]],
        [new_point[0][1], point[1]],
        [new_point[0][2], point[2]],
        linestyle='dashed',
        linewidth=2,
        color='white',
        alpha=0.7
    )

# labels
ax.set_title("KNN Regression - Neighbors in 3D", fontsize=14)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")

# color bar (important for regression)
cbar = plt.colorbar(scatter)
cbar.set_label("Target Value")

ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.05, 1)
)

plt.tight_layout()
plt.show()
