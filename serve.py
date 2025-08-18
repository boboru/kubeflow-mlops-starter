import argparse
import io
import os
import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kserve import (
    InferRequest,
    InferResponse,
    Model,
    ModelServer,
    logging,
    model_server,
)
from kserve.utils.utils import get_predict_response, get_predict_input

from feature_encoder import FeatureEncoder
from model import DCNv2


# KServe Custom model
class DCNv2Model(Model):
    def __init__(
        self, name: str, model_path: str, encoder_path: str, dense_cols, sparse_cols
    ):
        super().__init__(name)
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.dense_cols = dense_cols
        self.sparse_cols = sparse_cols
        self.ready = False
        self.device = "cpu"
        self.load()

    def load(self) -> bool:
        # Load feature encoder
        self.encoder = FeatureEncoder(self.dense_cols, self.sparse_cols)
        self.encoder.load(self.encoder_path)

        # Load model
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        dense_dim = len(self.dense_cols)
        sparse_cardinalities = self.encoder.get_sparse_cardinalities()
        self.model = DCNv2(dense_dim, sparse_cardinalities)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        self.ready = True
        return self.ready

    async def predict(
        self,
        payload: Union[Dict, InferRequest],
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> InferResponse:
        
        instances = get_predict_input(payload)

        # preprocessing
        dense_np, sparse_np = self.encoder.transform(instances)
        dense_tensor = torch.tensor(dense_np, dtype=torch.float32, device=self.device)
        sparse_tensor = torch.tensor(sparse_np, dtype=torch.long, device=self.device)

        # inference
        with torch.no_grad():
            preds = self.model(dense_tensor, sparse_tensor).numpy()

        # return v2 response
        return get_predict_response(payload, preds, self.name)


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model-path",
    default=os.environ.get("MODEL_PATH"),
    help="Path to the trained DCNv2 model .pt file",
)
parser.add_argument(
    "--encoder-path",
    default=os.environ.get("ENCODER_PATH"),
    help="Path to the fitted FeatureEncoder pickle file",
)

parser.add_argument(
    "--dense-cols",
    nargs="+",
    default=os.environ.get("DENSE_COLS", "").split(',')
    if os.environ.get("DENSE_COLS")
    else None,
    help="List of dense feature column names",
)
parser.add_argument(
    "--sparse-cols",
    nargs="+",
    default=os.environ.get("SPARSE_COLS", "").split(',')
    if os.environ.get("SPARSE_COLS")
    else None,
    help="List of sparse feature column names",
)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)

    model = DCNv2Model(
        name="dcnv2",
        model_path=args.model_path,
        encoder_path=args.encoder_path,
        dense_cols=args.dense_cols,
        sparse_cols=args.sparse_cols,
    )
    ModelServer().start([model])
