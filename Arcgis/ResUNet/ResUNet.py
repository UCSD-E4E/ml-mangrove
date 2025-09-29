
import torch
from fastai.learner import Learner
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torch.nn import Conv2d, Module

class ResUNet():
    def __init__(self, weights=None, state_dict=None):
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

    def get_model(self, data, backbone=None, **kwargs):
        """
        Function used to define the model architecture.
        Args:
            data: DataBunch object created in the prepare_data function
            backbone: Pretrained ResNet model to be used as the encoder. Default is ResNet18 with ImageNet weights.
            kwargs: Additional key word arguments to be passed to the model
        """
        if self.model is not None:
            return self.model
        
        class ResNet_UNet(Module):
            """
            UNet architecture with ResNet encoder.
            Defaults to ResNet18 with ImageNet weights.
            """
            def __init__(self, ResNet = None, input_image_size=224, num_classes=1):
                super(ResNet_UNet, self).__init__()
                if ResNet is None:
                    ResNet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                for param in ResNet.parameters():
                    param.requires_grad = False
                
                # Encoder
                self.layer1 = nn.Sequential(
                    ResNet.conv1,
                    ResNet.bn1,
                    nn.ReLU(),
                    ResNet.maxpool,
                    ResNet.layer1,
                )
                self.layer2 = ResNet.layer2
                self.layer3 = ResNet.layer3
                self.layer4 = ResNet.layer4

                # Define feature dimensions
                dummy_input = torch.randn(1, 3, input_image_size, input_image_size)
                x = self.layer1(dummy_input)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                feature_dim = x.shape[1]
                half_dim = feature_dim // 2
                quarter_dim = feature_dim // 4
                eighth_dim = feature_dim // 8
                sixteenth_dim = feature_dim // 16
                
                # Center
                self.center = Decoder(feature_dim, int(feature_dim // 1.5), half_dim)
                # Skip connections
                self.skip_conv1 = Conv2d(half_dim, half_dim, kernel_size=1)
                self.skip_conv2 = Conv2d(quarter_dim, quarter_dim, kernel_size=1)
                self.skip_conv3 = Conv2d(eighth_dim, eighth_dim, kernel_size=1)
                #decoder
                self.decoder1 = Decoder(feature_dim, half_dim, quarter_dim)
                self.decoder2 = Decoder(half_dim, quarter_dim, eighth_dim)
                self.classification_head = nn.Sequential(
                    Upsample(quarter_dim, eighth_dim, sixteenth_dim),
                    Conv2d(sixteenth_dim, num_classes, kernel_size=2, padding=1),
                    nn.Upsample(
                    size=(input_image_size, input_image_size),
                    mode="bilinear",
                    align_corners=False,
                    ),
                    Conv2d(num_classes, num_classes, kernel_size=3, padding=1) # smooth output
                )

            def forward(self, image):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image = image[:, :3, :, :]
                # Encode
                x1 = self.layer1(image)
                x2 = self.layer2(x1)
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)
                # Center
                x = self.center(x4)
                # Decode
                x = torch.cat((x, self.skip_conv1(x3)), dim=1)
                x = self.decoder1(x)
                x = torch.cat((x, self.skip_conv2(x2)), dim=1)
                x = self.decoder2(x)
                x = torch.cat((x, self.skip_conv3(x1)), dim=1)
                return self.classification_head(x)
        class Decoder(torch.nn.Module):
            """
            Helper class for the UNet architecture.
            Uses convolutional layers to upsample the input.
            Includes dropout layer to prevent overfitting.
            """
            def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
                super(Decoder, self).__init__()
                self.decoder = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
                )

            def forward(self, x) -> torch.Tensor:
                return self.decoder(x)
        class Upsample(Module):
            """
            Helper class for the UNet architecture.
            Uses convolutional layers to upsample the input.
            """
            def __init__(self, in_channels, mid_channels, out_channels):
                super(Upsample, self).__init__()
                self.upsampler = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
                )
            def forward(self, x) -> torch.Tensor:
                return self.upsampler(x)
        
        num_classes = len(data.classes) if hasattr(data, 'classes') else 1
        input_image_size = data.train_ds[0][0].shape[1] if hasattr(data, 'train_ds') else 224
        self.model = ResNet_UNet(ResNet=backbone, input_image_size=input_image_size, num_classes=num_classes)
        
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
                self.model.load_state_dict(state_dict)
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