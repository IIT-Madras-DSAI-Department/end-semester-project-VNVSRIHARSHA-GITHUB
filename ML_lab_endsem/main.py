import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,f1_score
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


x_train , y_train , x_val , y_val = read_data()

model = XGBoostMulticlass( n_estimators= 50, learning_rate=0.3, max_depth= 4, lamb=1.0,
                 feature_subsample_ratio=0.04, gamma=0.1, random_state=None,
                 learning_rate_decay=None,n_classes = 10)


model.fit(x_train,y_train)
ypred = model.predict(x_val)
#print("n_estimators= 50, learning_rate=0.3, max_depth=4, lamb=1.0, feature_subsample_ratio=0.04, gamma=0.1")
#print(f"accuracy_score : {accuracy_score(y_val,ypred)}")
#print(f"f1_score : {f1_score(y_val,ypred,average='macro')}")
