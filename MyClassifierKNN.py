import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MyClassifierKNN:

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
                if self.dist_metric == 'manhattan':
                    distance = self.manhattan_distance(row_X_train, row_test)
                distance_target = distance, target
                distances_targets.append(distance_target) 

            k_sorted_distances_targets = sorted(distances_targets)[ :self.k]
            targets = pd.Series(target for distance, target in k_sorted_distances_targets)
            prediction = targets.value_counts().idxmax()
            predictions.append(prediction)
        self.predictions = np.array(predictions)
        return self.predictions  
    
    def my_score(self, y_test):
        correct = (sum(y_test==self.predictions))
        tp = sum(self.predictions[y_test==1] == 1)
        tn = sum(self.predictions[y_test==0] == 0)
        fp = sum(self.predictions[y_test==0] == 1)
        fn = sum(self.predictions[y_test==1] == 0)     
        
        self.accuracy = correct / len(y_test)
        self.recall = tp / (tp + fn)
        self.precision = tp / (tp + fp)  
    
#==============================================================

plt.style.use('dark_background')

# Example 3D dataset (you can replace with your own)
X = np.array([
    [1, 2, 1], [2, 3, 2], [3, 3, 1],
    [6, 5, 6], [7, 7, 7], [8, 6, 5],
    [5, 2, 3], [6, 3, 4], [7, 2, 5]
])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# New point (the one we predict)
new_point = np.array([5, 4, 4])

k = 3

# ---- compute distances ----
distances = []
for point, label in zip(X, y):
    d = np.linalg.norm(point - new_point)
    distances.append((d, point, label))

distances.sort(key=lambda x: x[0])
neighbors = distances[:k]

# ---- prediction (majority vote) ----
labels = [label for _, _, label in neighbors]
prediction = max(set(labels), key=labels.count)


# ---- Plot ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 🎨 Better modern colors (high contrast on dark)
colors = {
    0: '#4cc9f0',   # cyan
    1: '#f72585',   # pink
    2: '#b9e769'    # light green
}

# Plot all data
for label in np.unique(y):
    ax.scatter(
        X[y == label][:, 0],
        X[y == label][:, 1],
        X[y == label][:, 2],
        color=colors[label],
        label=f"Class {label}",
        s=40,
        alpha=0.9
    )

# Prediction point (make it pop)
ax.scatter(
    new_point[0], new_point[1], new_point[2],
    color='white',
    s=220,
    edgecolor='black',
    linewidth=1.5,
    label=f"Prediction: {prediction}"
)

# Draw dashed lines to neighbors
for dist, point, label in neighbors:
    ax.plot(
        [new_point[0], point[0]],
        [new_point[1], point[1]],
        [new_point[2], point[2]],
        linestyle='dashed',
        linewidth=2,
        color=colors[label],
        alpha=0.9
    )

# Labels
ax.set_title("KNN - Nearest Neighbors in 3D", fontsize=14)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")

# 🔥 Move legend OUTSIDE
ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    frameon=True
)

plt.tight_layout()
plt.show()