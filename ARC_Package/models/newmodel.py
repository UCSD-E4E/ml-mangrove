from ModelClass import ModelClass
import torch

class newmodel(ModelClass):
    def on_batch_begin(self, learn, model_input_batch: torch.Tensor, model_target_batch: torch.Tensor):
        # Implement your batch transformation logic here
        raise NotImplementedError("This method needs to be implemented.")

    def transform_input(self, xb: torch.Tensor) -> torch.Tensor:
        # Implement your input transformation logic here
        raise NotImplementedError("This method needs to be implemented.")

    def transform_input_multispectral(self, xb: torch.Tensor) -> torch.Tensor:
        # Implement your multispectral input transformation logic here
        raise NotImplementedError("This method needs to be implemented.")

    def get_model(self, data, backbone=None, **kwargs):
        # Implement your model retrieval logic here
        raise NotImplementedError("This method needs to be implemented.")

    def loss(self, model_output, *model_target):
        # Implement your loss calculation logic here
        raise NotImplementedError("This method needs to be implemented.")

    def post_process(self, pred: torch.Tensor, thres: float = 0.5) -> torch.Tensor:
        # Implement your post-processing logic here
        raise NotImplementedError("This method needs to be implemented.")
