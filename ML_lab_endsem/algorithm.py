import pandas as pd
import numpy as np
import math

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):

    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = "label"

    Xtrain = dftrain[featurecols]
    ytrain = dftrain[targetcol]

    Xval = dfval[featurecols]
    yval = dfval[targetcol]

    return (Xtrain, ytrain, Xval, yval)


def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))

def softmax(logits):
    logits = np.asarray(logits)
    max_logit = np.max(logits, axis=1, keepdims=True)
    ex = np.exp(logits - max_logit)
    s = np.sum(ex, axis=1, keepdims=True)
    return ex / s

def grad_hess_multiclass(y, logits, n_classes):
    probs = softmax(logits)
    G = probs.copy()
    idx = (np.arange(y.shape[0]), y.astype(int))
    G[idx] -= 1.0
    H = probs * (1.0 - probs)
    return G, H



#XG Boost with col-subsampling


class DecisionStump:
    def __init__(self, lamb=1.0, max_depth=3, depth=0,
                 feature_subsample_ratio=0.04, random_state=None, gamma=0.0):
        self.feature_index = None
        self.threshold = None
        self.value = None
        self.left = None
        self.right = None
        self.lamb = lamb
        self.max_depth = max_depth
        self.depth = depth
        self.feature_subsample_ratio = feature_subsample_ratio
        self.gamma = gamma
        self.random_state = int(random_state) if random_state is not None else None
        self.rng = np.random.RandomState(self.random_state)

    def _calc_leaf_value(self, grad, hess):
        denom = np.sum(hess) + self.lamb
        if denom == 0:
            return 0.0
        return -np.sum(grad) / denom

    def fit(self, X, grad, hess):
        if self.depth >= self.max_depth or X.shape[0] == 0:
            self.value = self._calc_leaf_value(grad, hess)
            return
        m, n = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        n_features_to_use = max(1, int(self.feature_subsample_ratio * n))
        feature_indices = self.rng.choice(n, n_features_to_use, replace=False)
        for feature_index in feature_indices:
            col = X[:, feature_index]
            order = np.argsort(col)
            col_sorted = col[order]
            grad_sorted = grad[order]
            hess_sorted = hess[order]
            unique_vals, idx_first = np.unique(col_sorted, return_index=True)
            if unique_vals.shape[0] == 1:
                continue
            grad_cumsum = np.cumsum(grad_sorted)
            hess_cumsum = np.cumsum(hess_sorted)
            split_positions = idx_first[1:] - 1
            if split_positions.size == 0:
                continue
            G_left = grad_cumsum[split_positions]
            H_left = hess_cumsum[split_positions]
            G_total = grad_cumsum[-1]
            H_total = hess_cumsum[-1]
            G_right = G_total - G_left
            H_right = H_total - H_left
            denom_left = H_left + self.lamb
            denom_right = H_right + self.lamb
            denom_total = H_left + H_right + self.lamb
            denom_left = np.where(denom_left == 0, 1e-12, denom_left)
            denom_right = np.where(denom_right == 0, 1e-12, denom_right)
            denom_total = np.where(denom_total == 0, 1e-12, denom_total)
            gain_vec = 0.5 * (
                (G_left ** 2) / denom_left +
                (G_right ** 2) / denom_right -
                ((G_left + G_right) ** 2) / denom_total
            )
            idx_best = np.argmax(gain_vec)
            local_best_gain = gain_vec[idx_best]
            if local_best_gain > best_gain and local_best_gain > self.gamma:
                best_gain = float(local_best_gain)
                split_pos = split_positions[idx_best]
                v_left = col_sorted[split_pos]
                v_right = col_sorted[split_pos + 1]
                threshold = 0.5 * (v_left + v_right)
                best_feature = feature_index
                best_threshold = float(threshold)
        if best_feature is None:
            self.value = self._calc_leaf_value(grad, hess)
            return
        self.feature_index = best_feature
        self.threshold = best_threshold
        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask
        left_seed = self.rng.randint(0, 2**31 - 1)
        right_seed = self.rng.randint(0, 2**31 - 1)
        self.left = DecisionStump(
            lamb=self.lamb,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            feature_subsample_ratio=self.feature_subsample_ratio,
            random_state=left_seed,
            gamma=self.gamma
        )
        self.right = DecisionStump(
            lamb=self.lamb,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            feature_subsample_ratio=self.feature_subsample_ratio,
            random_state=right_seed,
            gamma=self.gamma
        )
        self.left.fit(X[left_mask], grad[left_mask], hess[left_mask])
        self.right.fit(X[right_mask], grad[right_mask], hess[right_mask])

    def predict(self, X):
        if X.shape[0] == 0:
            return np.array([])
        if self.feature_index is None:
            return np.full(X.shape[0], self.value)
        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask
        ypred = np.zeros(X.shape[0])
        if np.any(left_mask):
            ypred[left_mask] = self.left.predict(X[left_mask])
        if np.any(right_mask):
            ypred[right_mask] = self.right.predict(X[right_mask])
        return ypred

class XGBoostMulticlass:
    def __init__(self, n_classes, n_estimators=10, learning_rate=0.1, max_depth=3, lamb=1.0,
                 feature_subsample_ratio=0.04, gamma=0.0, random_state=None, learning_rate_decay=None):
        self.n_classes = int(n_classes)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.lamb = float(lamb)
        self.feature_subsample_ratio = float(feature_subsample_ratio)
        self.gamma = float(gamma)
        self.random_state = int(random_state) if random_state is not None else None
        self.learning_rate_decay = learning_rate_decay
        self.rng = np.random.RandomState(self.random_state)
        self.trees = [] 
        self.base_score = 0.0
        self.logits = None
        self.proba_threshold = None

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=None):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        m, n = X.shape
        self.logits = np.zeros((m, self.n_classes))
        best_val_loss = np.inf
        rounds_no_improve = 0
        for it in range(self.n_estimators):
            G, H = grad_hess_multiclass(y, self.logits, self.n_classes)
            trees_this_round = []
            for k in range(self.n_classes):
                gk = G[:, k]
                hk = H[:, k]
                seed = self.rng.randint(0, 2**31 - 1)
                stump = DecisionStump(lamb=self.lamb, max_depth=self.max_depth, depth=0,
                                      feature_subsample_ratio=self.feature_subsample_ratio,
                                      random_state=seed, gamma=self.gamma)
                stump.fit(X, gk, hk)
                trees_this_round.append(stump)
                eta = self.learning_rate
                if self.learning_rate_decay is not None:
                    eta = eta * (self.learning_rate_decay ** it)
                self.logits[:, k] += eta * stump.predict(X)
            self.trees.append(trees_this_round)
            if X_val is not None and y_val is not None and early_stopping_rounds is not None:
                val_logits = self._predict_logits(X_val)
                probs = softmax(val_logits)
                eps = 1e-12
                p = np.clip(probs, eps, 1 - eps)
                val_loss = -np.mean(np.log(p[np.arange(p.shape[0]), y_val.astype(int)]))
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                    if rounds_no_improve >= early_stopping_rounds:
                        break
        self.proba_threshold = None

    def _predict_logits(self, X):
        X = np.asarray(X)
        m = X.shape[0]
        logits = np.zeros((m, self.n_classes))
        for it, trees_this_round in enumerate(self.trees):
            for k, stump in enumerate(trees_this_round):
                eta = self.learning_rate
                if self.learning_rate_decay is not None:
                    eta = eta * (self.learning_rate_decay ** it)
                logits[:, k] += eta * stump.predict(X)
        return logits

    def predict_proba(self, X):
        logits = self._predict_logits(X)
        return softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


#PCA

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

#KNN



class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []

        for x in X:
          
            distances = np.array([self._euclidean_distance(x, x_train)
                                  for x_train in self.X_train])

            k_idx = np.argsort(distances)[:self.k]

            k_neighbor_labels = self.y_train[k_idx]

            most_common = np.bincount(k_neighbor_labels).argmax()
            predictions.append(most_common)

        return np.array(predictions)


#SVM


class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_p=0.001, n_iters=2000):
        self.lr = learning_rate
        self.lambda_p = lambda_p
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
          
            margins = y * (X.dot(self.w) + self.b)


            misclassified = margins < 1

            dw = self.lambda_p * self.w - np.dot(X[misclassified].T, y[misclassified]) / n_samples
            db = -np.sum(y[misclassified]) / n_samples

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def decision_function(self, X):
        return X.dot(self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))
    

class MulticlassSVM:
    def __init__(self, learning_rate=0.01, lambda_p=0.001, n_iters=2000):
        self.lr = learning_rate
        self.lambda_p = lambda_p
        self.n_iters = n_iters
        self.models = []
        self.n_classes = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.models = []
        
        for c in range(self.n_classes):
            y_binary = np.where(y == c, 1, -1)
            svm = LinearSVM(
                learning_rate=self.lr,
                lambda_p=self.lambda_p,
                n_iters=self.n_iters
            )
            svm.fit(X, y_binary)
            self.models.append(svm)

    def predict(self, X):
        scores = np.column_stack([model.decision_function(X) for model in self.models])
        return np.argmax(scores, axis=1)


#XGBOOST GREEDY ALGORITHM

class DecisionStump:
    def __init__(self, lamb=1.0, max_depth=3, depth=0):
        self.feature_index = None
        self.threshold = None
        self.value = None
        self.left = None
        self.right = None
        self.lamb = lamb
        self.max_depth = max_depth
        self.depth = depth

    def fit(self, X, grad, hess):
        if self.depth >= self.max_depth or X.shape[0] == 0:
            self.value = -np.sum(grad) / (np.sum(hess) + self.lamb)
            return

        m, n = X.shape
        best_gain = -float('inf')

        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                grad_left = np.sum(grad[left_mask])
                grad_right = np.sum(grad[right_mask])

                hess_left = np.sum(hess[left_mask])
                hess_right = np.sum(hess[right_mask])

                gain = 0.5 * (
                    grad_left**2 / (hess_left + self.lamb) +
                    grad_right**2 / (hess_right + self.lamb) -
                    (grad_left + grad_right)**2 /
                    (hess_left + hess_right + self.lamb)
                )

                if gain > best_gain:
                    best_gain = gain
                    self.feature_index = feature_index
                    self.threshold = threshold

        if self.feature_index is None:
            self.value = -np.sum(grad) / (np.sum(hess) + self.lamb)
            return

        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask

        self.left = DecisionStump(self.lamb, self.max_depth, self.depth+1)
        self.right = DecisionStump(self.lamb, self.max_depth, self.depth+1)

        self.left.fit(X[left_mask], grad[left_mask], hess[left_mask])
        self.right.fit(X[right_mask], grad[right_mask], hess[right_mask])

    def predict(self, X):
        if X.shape[0] == 0:
            return np.array([])

        if self.feature_index is None:
            return np.full(X.shape[0], self.value)

        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask
        out = np.zeros(X.shape[0])
        out[left_mask] = self.left.predict(X[left_mask])
        out[right_mask] = self.right.predict(X[right_mask])
        return out



class XGBoostGreedy:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3, lamb=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lamb = lamb
        self.estimators = [] 
        self.n_classes = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        m, n = X.shape
        self.n_classes = len(np.unique(y))

        logits = np.zeros((m, self.n_classes))


        for _ in range(self.n_estimators):
            G, H = grad_hess_multiclass(y, logits, self.n_classes)

            round_estimators = []
            for c in range(self.n_classes):
                stump = DecisionStump(lamb=self.lamb, max_depth=self.max_depth)
                stump.fit(X, G[:, c], H[:, c])
                round_estimators.append(stump)

                logits[:, c] += self.learning_rate * stump.predict(X)

            self.estimators.append(round_estimators)

    def predict_proba(self, X):
        X = np.asarray(X)
        m = X.shape[0]
        logits = np.zeros((m, self.n_classes))

        for round_estimators in self.estimators:
            for c, stump in enumerate(round_estimators):
                logits[:, c] += self.learning_rate * stump.predict(X)

        return softmax(logits)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

