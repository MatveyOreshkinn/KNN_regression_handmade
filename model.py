import numpy as np
import pandas as pd
from scipy.spatial import distance


class MyKNNReg:
    def __init__(self,
                 k: int = 3,
                 train_size: int = None,
                 metric: str = 'euclidean',
                 weight: str = 'uniform') -> None:
        self.k = k
        self.train_size = train_size
        self.metric = metric
        self.weight = weight

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

                if self.metric == 'euclidean':
                    d = distance.euclidean(point1, point2)
                elif self.metric == 'chebyshev':
                    d = distance.chebyshev(point1, point2)
                elif self.metric == 'manhattan':
                    d = distance.cityblock(point1, point2)
                elif self.metric == 'cosine':
                    d = distance.cosine(point1, point2)
                distances.append((d, self.y[j]))

            k_distances = sorted(distances, key=lambda x: x[0])[:self.k]
            k_classes = [el[1] for el in k_distances]

            if self.weight == 'uniform':
                k_classes_mean = sum(k_classes) / len(k_distances)
                pred.append(k_classes_mean)

            elif self.weight == 'rank':
                total_weight = 0
                weights = []
                for i, (dist, value) in enumerate(k_distances):
                    rank = i+1
                    weight = 1 / rank
                    total_weight += weight
                    weights.append(weight)
                weighted_sum = 0
                for i, (dist, value) in enumerate(k_distances):
                    weighted_sum += (weights[i] / total_weight) * value
                pred.append(weighted_sum)

            elif self.weight == 'distance':
                weighted_sum = 0
                weights = []
                for dist, value in k_distances:
                    weight = 1 / dist
                    weighted_sum += weight
                    weights.append(weight)

                predict = 0
                for i, (dist, value) in enumerate(k_distances):
                    predict += (weights[i] / weighted_sum) * value
                pred.append(predict)
        return np.array(pred)
