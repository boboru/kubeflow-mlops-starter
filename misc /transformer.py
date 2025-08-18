import argparse
import pickle

import pandas as pd
from kserve import InferInput, InferRequest, Model, ModelServer, model_server, logging


class TabularTransformer(Model):
    def __init__(self, name: str, feature_encoder_path: str):
        super().__init__(name)
        self.feature_encoder_path = feature_encoder_path
        self.encoder = None

    def load(self):
        with open(self.feature_encoder_path, "rb") as f:
            self.feature_encoder = pickle.load(f)
        self.ready = True

    def preprocess(self, request: InferRequest, headers=None) -> InferRequest:
        raw_data = request.inputs[0].data

        df = pd.DataFrame(raw_data)
        dense_np, sparse_np = self.feature_encoder.transform(df)

        # transform to InferInput (dense and sparse)
        infer_inputs = [
            InferInput(
                name="dense",
                datatype="FP32",
                shape=list(dense_np.shape),
                data=dense_np.tolist(),
            ),
            InferInput(
                name="sparse",
                datatype="INT64",
                shape=list(sparse_np.shape),
                data=sparse_np.tolist(),
            ),
        ]

        # transform to a new InferRequest
        infer_request = InferRequest(
            model_name=self.model_name, infer_inputs=infer_inputs
        )

        return infer_request


parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)
    model = TabularTransformer(
        args.model_name,
        feature_encoder_path=args.feature_encoder_path,
    )
    ModelServer().start([model])