from ts.torch_handler.base_handler import BaseHandler
import torch

class PredictionHandler(BaseHandler):
    """
    Rewrite inference function to support dense and sparse features
    """
    def inference(self, inputs, *args, **kwargs):
        with torch.inference_mode():
            dense_tensor = torch.tensor(inputs[0]["data"], dtype=torch.float32, device=self.device)
            sparse_tensor = torch.tensor(inputs[1]["data"], dtype=torch.long, device=self.device)

        return self.model(dense_tensor, sparse_tensor)






