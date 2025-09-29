
import torch
from fastai.learner import Learner
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
from torch.nn import Module
# Based on https://developers.arcgis.com/python/latest/guide/add-model-using-model-extension/

class SegFormer():
    def __init__(self, weights="nvidia/segformer-b0-finetuned-ade-512-512", state_dict=None):
        """
        Custom Model class to define the model architecture, loss function and input transformations.
        Args:
            weights: Pretrained weights to be used for the model. Default is "nvidia/segformer-b0-finetuned-ade-512-512
            state_dict: Path to the trained state dictionary to be loaded into the model. Default is None
        """
        self.model = None
        self.weights = weights
        self.state_dict = state_dict

    def on_batch_begin(self, learn: Learner, model_input_batch: torch.Tensor, model_target_batch: torch.Tensor):
        """
        Function to transform the input data and the targets in accordance to the model for training.
        Args:
            learn: a fastai learner object
            model_input_batch: fastai transformed batch of input images - tensor of shape [B, C, H, W]
                with values in the range -1 and 1.
            model_target_batch: fastai transformed batch of targets. The targets will be of different type and shape for object detection and pixel classification.
        """
        return model_input_batch, model_target_batch

    def transform_input(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Function to transform the inputs for inferencing.
        Args:
            xb: fastai transformed batch of input images: tensor of shape [N, C, H, W],
             where N - batch size C - number of channels (bands) in the image H - height of the image W - width of the image
        """
        return xb[:, :3, :, :]

    def transform_input_multispectral(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Function to transform the multispectral inputs for inferencing.
        Args:
            xb: fastai transformed batch of input images: tensor of shape [N, C, H, W],
             where N - batch size C - number of channels (bands) in the image H - height of the image W - width of the image
        """
        return xb[:, :3, :, :]

    def get_model(self, data, **kwargs):
        """
        Function used to define the model architecture.
        Args:
            data: DataBunch object created in the prepare_data function
            kwargs: Additional key word arguments to be passed to the model
        """
        if self.model is not None:
            return self.model
    
        class SegFormerModel(Module):
            """
            SegFormer model for semantic segmentation.
            Uses a pretrained SegFormer backbone and replaces the decode head to upsample to the input image size.
            
            https://github.com/NVlabs/SegFormer
            
            """
            def __init__(self, num_classes=1, input_image_size=128, weights="nvidia/segformer-b2-finetuned-ade-512-512"):
                super(SegFormerModel, self).__init__()
                self.num_classes = num_classes
                self.input_image_size = input_image_size

                self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                    weights,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )

                for param in self.segformer.parameters():
                    param.requires_grad = False

                output_feature_size = self.segformer.config.decoder_hidden_size

                # Replace the decode head to upsample to input image size
                self.segformer.decode_head.classifier = nn.Sequential( # type: ignore
                nn.ConvTranspose2d(output_feature_size, output_feature_size // 2, kernel_size=4, stride=2, padding=1), 
                nn.BatchNorm2d(output_feature_size // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(output_feature_size // 2, output_feature_size // 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(output_feature_size // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_feature_size // 4, num_classes, kernel_size=3, padding=1),
            )
                
                for param in self.segformer.decode_head.classifier.parameters(): # type: ignore
                    param.requires_grad = True

            def forward(self, image):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image = image[:, :3, :, :]
                output = self.segformer(image).logits
                if output.shape[2] != image.shape[2] or output.shape[3] != image.shape[3]:
                    output = nn.functional.interpolate(output, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
                return output
            
            def freeze_backbone(self):
                for param in self.segformer.parameters():
                    param.requires_grad = False
                for param in self.segformer.decode_head.classifier.parameters(): # type: ignore
                    param.requires_grad = True
            
            def train_backbone(self):
                for param in self.segformer.parameters():
                    param.requires_grad = True

        weights = kwargs.get("weights", self.weights)
        num_classes = len(data.classes) if hasattr(data, 'classes') else 1
        train_backbone = kwargs.get("train_backbone", False)

        segformer = SegFormerModel(num_classes=num_classes, input_image_size=data.train_ds[0][0].shape[1], weights=weights)
        if train_backbone:
            segformer.train_backbone()

        if self.state_dict is not None:
            kwargs["state_dict"] = self.state_dict
        if kwargs.get("state_dict", None) is not None:
            obj = torch.load(kwargs["state_dict"], map_location=torch.device('cpu'))
            if isinstance(obj, dict):
                if all(isinstance(v, torch.Tensor) for v in obj.values()):
                    state_dict = obj
                elif 'state_dict' in obj:
                    state_dict = obj['state_dict']
                elif 'model_state_dict' in obj:
                    state_dict = obj['model_state_dict']
                segformer.load_state_dict(state_dict)
        for param in segformer.decode_head.classifier.parameters(): # type: ignore
            param.requires_grad = True
        
        self.model = segformer
        return self.model

    def loss(self, model_output, *model_target):
        """
        Function to define the loss calculations.
        Args:
            model_output: Raw output of the model for a batch of images
            model_target: Ground truth target one_batch_begin function
        """
        logits = model_output.logits  # [N, C, H, W]
        targets = model_target[0].long()  # ground truth
        return F.cross_entropy(logits, targets, ignore_index=255)

    def post_process(self, pred: torch.Tensor, thres: float) -> torch.Tensor:
        """
        Function to post process the output of the model in validation/infrencing mode.
        Args:
            pred: Raw output of the model for a batch of images
            thres: Confidence threshold to be used to filter the predictions
            post_processed_pred: tensor of shape [N, 1, H, W] or a List/Tuple of N tensors of shape [1, H, W], where N - batch size H - height of the image W - width of the image
        """
        if pred.shape[1] == 1:
            # Binary segmentation → threshold
            return (torch.sigmoid(pred) > thres).long().squeeze(1)
        else:
            # Multi-class segmentation → argmax
            return torch.argmax(pred, dim=1, keepdim=True)