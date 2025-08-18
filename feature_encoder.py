import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEncoder:
    def __init__(self, dense_cols, sparse_cols, oov_index='auto'):
        self.dense_cols = dense_cols
        self.sparse_cols = sparse_cols
        self.oov_index = oov_index

        self.scaler = StandardScaler()
        self.encoders = {col: LabelEncoder() for col in sparse_cols}
        self.fitted = False

    def fit(self, df):
        df = df.copy()
        df[self.dense_cols] = df[self.dense_cols].fillna(0)
        self.scaler.fit(df[self.dense_cols])

        for col in self.sparse_cols:
            self.encoders[col].fit(df[col].astype(str))

        self.fitted = True

    def transform(self, df, is_train=False):
        assert self.fitted, "FeatureEncoder must be fitted first."
        df = df.copy()

        # Dense features
        df[self.dense_cols] = df[self.dense_cols].fillna(0)
        dense = self.scaler.transform(df[self.dense_cols])

        # Sparse features
        sparse_array = []
        for col in self.sparse_cols:
            encoder = self.encoders[col]
            col_data = df[col].astype(str).values
            if is_train:
                encoded = encoder.transform(col_data)
            else:
                encoded = np.array([
                    encoder.transform([val])[0] if val in encoder.classes_ else len(encoder.classes_)
                    for val in col_data
                ])
            sparse_array.append(encoded)

        sparse = np.vstack(sparse_array).T  # shape [batch, num_sparse]
        return dense, sparse

    def get_sparse_cardinalities(self, include_oov=True):
        return [
            len(enc.classes_) + (1 if include_oov else 0)
            for enc in self.encoders.values()
        ]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.scaler, self.encoders), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.scaler, self.encoders = pickle.load(f)
        self.fitted = True