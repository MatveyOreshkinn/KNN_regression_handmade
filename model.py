import numpy as np
import pandas as pd


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
