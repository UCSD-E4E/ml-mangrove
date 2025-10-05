from torch import Tensor
from torch.nn import Module
from abc import ABC, abstractmethod
# Based on https://developers.arcgis.com/python/latest/guide/add-model-using-model-extension/

class ModelClass(ABC):
    def __init__(self, state_dict=None, weights=None):
        """
        Custom Model class to define the model architecture, loss function, and input transformations.
        Args:
            state_dict: path to pretrained state_dict to be used for the model.
            weights: Pretrained weights to be used for the model.
        """
        self.name = self.__class__.__name__
        self.description = f"{self.name} model for pixel classification"
        self.model = None
        self.state_dict = state_dict
        self.weights = weights

    @abstractmethod
    def on_batch_begin(self, learn, model_input_batch: Tensor, model_target_batch: Tensor):
        """
        Function to transform the input data and the targets in accordance to the model for training.
        Args:
            learn: a fastai learner object
            model_input_batch: fastai transformed batch of input images - tensor of shape [B, C, H, W]
                with values in the range -1 and 1.
            model_target_batch: fastai transformed batch of targets. The targets will be of different type and shape for object detection and pixel classification.
        """
        pass

    @abstractmethod
    def transform_input(self, xb: Tensor) -> Tensor:
        """
        Function to transform the inputs for inferencing.
        Args:
            xb: batch of input images: tensor of shape [N, C, H, W],
             where N - batch size C - number of channels (bands) in the image H - height of the image W - width of the image
        """
        pass

    @abstractmethod
    def transform_input_multispectral(self, xb: Tensor) -> Tensor:
        """
        Function to transform the multispectral inputs for inferencing.
        Args:
            xb: batch of input images: tensor of shape [N, C, H, W],
             where N - batch size C - number of channels (bands) in the image H - height of the image W - width of the image
        """
        pass

    @abstractmethod
    def get_model(self, data, backbone=None, **kwargs) -> Module:
        """
        Function used to define the model architecture.
        Args:
            data: DataBunch object created in the prepare_data function
            backbone: weights to be used for the encoder
            kwargs: Additional key word arguments to be passed to the model. Should include state_dict if pretrained weights are to be used.
        """
        pass

    @abstractmethod
    def loss(self, model_output, *model_target):
        """
        Function to define the loss calculations.
        Args:
            model_output: Raw output of the model for a batch of images
            model_target: Ground truth target one_batch_begin function
        """
        pass

    @abstractmethod
    def post_process(self, pred: Tensor, thres: float = 0.5) -> Tensor:
        """
        Function to post process the output of the model in validation/infrencing mode.
        Args:
            pred: Raw output of the model for a batch of images
            thres: Confidence threshold to be used to filter the predictions
        """
        pass