import numpy as np
import pandas as pd
from scipy.spatial import distance


class MyKNNReg:
    def __init__(self, k: int = 3, train_size: int = None) -> None:
        self.k = k
        self.train_size = train_size

    def __str__(self) -> str:
        return f'MyKNNReg class: k={self.k}'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_size = X.shape[0], X.shape[1]
        self.X = X.copy()  # Сохраняем копию, чтобы не менять исходный DataFrame
        self.y = y.copy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = []

        for i in range(len(X)):
            distances = []
            point1 = X.iloc[i]

            for j in range(len(self.X)):
                point2 = self.X.iloc[j]

                euclid = distance.euclidean(point1, point2)
                distances.append((euclid, self.y[j]))

            k_distances = sorted(distances, key=lambda x: x[0])[:self.k]
            k_classes = sum(el[1] for el in k_distances) / len(k_distances)

            pred.append(k_classes)

        return np.array(pred)
