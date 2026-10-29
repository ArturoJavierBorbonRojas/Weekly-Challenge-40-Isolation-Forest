import numpy as np

# Weekly Challenge 40: Unsupervised Anomaly Detection - Isolation Forest from Scratch
# Author: Ing. Arturo Javier Borbon Rojas

print("Initializing Isolation Forest Engine...")
print("-" * 65)

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.is_leaf = False

    def fit(self, X, current_height=0):
        self.size = X.shape[0]
        
        # Stop growing if height limit is reached or node has 1 or 0 samples
        if current_height >= self.height_limit or self.size <= 1:
            self.is_leaf = True
            return self
            
        # Select a random feature
        n_features = X.shape[1]
        self.split_feature = np.random.randint(0, n_features)
        
        # Find min and max of the selected feature and pick a random split threshold
        f_min = X[:, self.split_feature].min()
        f_max = X[:, self.split_feature].max()
        
        if f_min == f_max:
            self.is_leaf = True
            return self
            
        self.split_value = np.random.uniform(f_min, f_max)
        
        # Split the data
        left_mask = X[:, self.split_feature] < self.split_value
        right_mask = ~left_mask
        
        # Recursively build branches
        self.left = IsolationTree(self.height_limit).fit(X[left_mask], current_height + 1)
        self.right = IsolationTree(self.height_limit).fit(X[right_mask], current_height + 1)
        
        return self

    def path_length(self, x, current_height=0):
        if self.is_leaf:
            # Adjustment factor for unbuilt branches (c(n) mathematically standardizes the path)
            return current_height + self._c(self.size)
            
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, current_height + 1)
        else:
            return self.right.path_length(x, current_height + 1)

    def _c(self, n):
        """Mathematical adjustment for the average path length of unsuccessful searches in a BST."""
        if n > 2:
            return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
        elif n == 2:
            return 1.0
        else:
            return 0.0

class IsolationForest:
    def __init__(self, n_trees=100, subsample_size=256):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.trees = []
        self.c_limit = 0

    def fit(self, X):
        print(f"Growing {self.n_trees} random Isolation Trees...")
        n_samples = X.shape[0]
        self.subsample_size = min(self.subsample_size, n_samples)
        
        # Mathematical height limit for the trees: log2(subsample_size)
        height_limit = int(np.ceil(np.log2(self.subsample_size)))
        self.c_limit = IsolationTree(0)._c(self.subsample_size)
        
        for i in range(self.n_trees):
            # 1. Randomly subsample the data
            indices = np.random.choice(n_samples, self.subsample_size, replace=False)
            X_sub = X[indices]
            
            # 2. Grow the random tree
            tree = IsolationTree(height_limit).fit(X_sub)
            self.trees.append(tree)
            
            if (i + 1) % 25 == 0:
                print(f"   ➔ Grown {i + 1} trees...")

    def predict(self, X, threshold=0.6):
        """Returns anomaly scores and boolean flags (True if anomaly)."""
        scores = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            # Average path length across all trees in the forest
            path_lengths = [tree.path_length(X[i]) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            
            # Mathematical anomaly score: 2^(-E[h(x)] / c(n))
            scores[i] = 2.0 ** (-avg_path_length / self.c_limit)
            
        return scores, scores > threshold

# --- Educational Dataset (Normal clusters vs Hidden Anomalies) ---
# Features: [Absences (0-30), Socio-emotional Stress (1-10), Grades (0-10)]
np.random.seed(42)

# Generate 200 normal/regular students
normal_students = np.random.normal(loc=[5, 3, 8.5], scale=[2, 1, 0.5], size=(200, 3))

# Generate 3 extremely atypical profiles (e.g., Perfect grades but max stress & high absences)
anomalies = np.array([
    [25, 9.5, 9.8], # Total burnout anomaly
    [0, 1.0, 3.0],  # Apathetic anomaly (Present and calm, but failing)
    [15, 8.0, 8.5]  # Borderline anomaly
])

X_data = np.vstack([normal_students, anomalies])

# Train the Forest
iso_forest = IsolationForest(n_trees=100, subsample_size=64)
iso_forest.fit(X_data)

print("-" * 65)
print("EVALUATING DATASET FOR ANOMALIES:")

# Score the known anomalies and a few normal students
test_indices = [0, 1, 200, 201, 202] 
X_test = X_data[test_indices]
scores, is_anomaly = iso_forest.predict(X_test, threshold=0.55)

for i, idx in enumerate(test_indices):
    profile_type = "Normal Student" if idx < 200 else "Injected Anomaly"
    status = "🚨 EXTREME RISK (Isolated)" if is_anomaly[i] else "Regular Pattern"
    
    print(f"\n{profile_type} [Absences: {X_test[i][0]:.1f}, Stress: {X_test[i][1]:.1f}, Grades: {X_test[i][2]:.1f}]")
    print(f"   ➔ Anomaly Score: {scores[i]:.4f} | {status}")