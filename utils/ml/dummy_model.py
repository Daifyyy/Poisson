import numpy as np

class DummyModel:
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        home_recent = X[:, 0]
        away_recent = X[:, 1]
        elo_diff = X[:, 2]
        xg_diff = X[:, 3]
        home = 0.4 + 0.1*home_recent - 0.1*away_recent + 0.002*elo_diff + 0.05*xg_diff
        away = 0.4 + 0.1*away_recent - 0.1*home_recent - 0.002*elo_diff - 0.05*xg_diff
        draw = 1 - home - away
        probs = np.vstack([home, draw, away]).T
        probs = np.clip(probs, 0.01, 0.98)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


class SimpleLabelEncoder:
    def __init__(self):
        self.classes_ = np.array(['H', 'D', 'A'])

    def inverse_transform(self, indices):
        return [self.classes_[i] for i in indices]
