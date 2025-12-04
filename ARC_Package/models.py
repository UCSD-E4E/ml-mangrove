import torch
from torch import Tensor
from abc import ABC, abstractmethod
import torch.nn as nn
from torch.nn import Module, Conv2d
import torch.nn.functional as F
from transformers import SegformerModel, SegformerDecodeHead
from torch.amp.autocast_mode import autocast
from torchvision.models import ResNet18_Weights, resnet18, resnet34, ResNet34_Weights, ResNet50_Weights, resnet50, resnet101, ResNet101_Weights, ResNet152_Weights, resnet152
# Based on https://developers.arcgis.com/python/latest/guide/add-model-using-model-extension/

__all__ = ['ModelClass', 'SegFormer', 'ResNetUNet']

class ModelClass(ABC):
    def __init__(self, state_dict=None, weights=None):
        """
        Custom Model class to define the model architecture, loss function, and input transformations.
        Args:
            state_dict: path to pretrained state_dict to be used for the model.
            weights: Pretrained weights to be used for the model.
        """
        self.model = None
        self.state_dict = state_dict
        self.weights = weights
        self.name = "BaseModel"
        self.description = "Base model class"

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
    def get_model(self, backbone=None, image_size=512, **kwargs) -> Module:
        """
        Function used to define the model architecture.
        Args:
            backbone: weights to be used for the encoder
            image_size: Size of the input image
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
    def post_process(self, pred: Tensor, thres: float = 0.5, nodata_value=255) -> Tensor:
        """
        Function to post process the output of the model in validation/infrencing mode.
        Args:
            pred: Raw output of the model for a batch of images
            thres: Confidence threshold to be used to filter the predictions
        """
        pass

class ResNetUNet(ModelClass):
    def __init__(self, state_dict=None, weights=None):
        """
        Custom Model class to define the model architecture, loss function and input transformations.
        Args:
            state_dict: path to pretrained state_dict to be used for the model.
            weights: Pretrained weights to be used for the model."
        """
        super().__init__(state_dict=state_dict, weights=weights)
        self.name = "ResUNet"
        self.description = "UNet model for pixel classification"
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def on_batch_begin(self, learn, model_input_batch: torch.Tensor, model_target_batch: torch.Tensor):
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
        # Ensure 4D
        if xb.ndim == 3:
            xb = xb.unsqueeze(0)

        xb = xb[:, :3].float()
        if xb.max() > 1.5:
            xb = xb / 255.0
        xb = xb.clamp(0.0, 1.0)

        return (xb - self.mean.to(xb.device)) / self.std.to(xb.device)

    def transform_input_multispectral(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Function to transform the multispectral inputs for inferencing.
        Args:
            xb: fastai transformed batch of input images: tensor of shape [N, C, H, W],
             where N - batch size C - number of channels (bands) in the image H - height of the image W - width of the image
        """
        if len(xb.shape) == 3:
            xb = xb.unsqueeze(0)

        # Normalize using ImageNet stats
        if xb.max() > 1.5:
            xb = xb / 255.0
        
        return (xb - self.mean.to(xb.device)) / self.std.to(xb.device)

    def get_model(self, backbone="resnet18", image_size=512, num_classes=1, **kwargs):
        """
        Function used to define the model architecture.
        Args:
            data: DataBunch object created in the prepare_data function
            backbone: Pretrained ResNet model to be used as the encoder. Default is ResNet18 with ImageNet weights.
                \n The options are:
                resnet18, resnet34, resnet50, resnet101, resnet152
            kwargs: Additional key word arguments to be passed to the model
        """
        if self.model is not None:
            return self.model
        
        class ResNet_UNet(Module):
            """
            UNet architecture with ResNet encoder.
            Defaults to ResNet18 with ImageNet weights.
            """
            def __init__(self, backbone = "resnet18", input_image_size=512, num_classes=1):
                super(ResNet_UNet, self).__init__()
                if backbone == "resnet18":
                    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                elif backbone == "resnet34":
                    backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
                elif backbone == "resnet50":
                    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                elif backbone == "resnet101":
                    backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
                elif backbone == "resnet152":
                    backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
                else:
                    raise ValueError(f"Backbone not supported. Supported backbones are: resnet18, resnet34, resnet50, resnet101, resnet152.")
                
                for param in backbone.parameters():
                    param.requires_grad = False
                
                # Encoder
                self.num_classes = num_classes
                self.layer1 = nn.Sequential(
                    backbone.conv1,
                    backbone.bn1,
                    nn.ReLU(),
                    backbone.maxpool,
                    backbone.layer1,
                )
                self.layer2 = backbone.layer2
                self.layer3 = backbone.layer3
                self.layer4 = backbone.layer4

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
                with autocast(next(self.layer1.parameters()).device.type):
                    # Ensure 4D
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
        
        self.model = ResNet_UNet(backbone=backbone, input_image_size=image_size, num_classes=num_classes)
        
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


    def post_process(self, pred: torch.Tensor, thres: float = 0.5) -> torch.Tensor:
        """
        Post-process raw model outputs for binary or multi-class segmentation.
        Always returns [N, H, W] label masks.
        """
        pred = pred.detach()

        # Ensure 4D shape: [N, C, H, W]
        if pred.ndim == 2:           # [H, W]
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 3:         # could be [N, H, W] or [1, H, W]
            if pred.shape[0] > 1:    # [N, H, W]
                pred = pred.unsqueeze(1)  # -> [N, 1, H, W]
            else:
                pred = pred.unsqueeze(0)  # -> [1, 1, H, W]


        if pred.shape[1] == 1:
            pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        else:
            pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        
        # Apply output processing
        N, C, H, W = pred.shape
        if C == 1: # Binary segmentation
            prob = torch.sigmoid(pred)
            mask = (prob > thres).long()
            return mask.squeeze(1)      # -> [N, H, W]
        else: # Multi-class segmentation
            return torch.argmax(pred, dim=1)  # -> [N, H, W]

class SegFormer(ModelClass):
    def __init__(self, state_dict=None, weights=None):
        """
        Custom Model class to define the model architecture, loss function and input transformations.
        Args:
            state_dict: path to pretrained state_dict to be used for the model.
            weights: Pretrained weights to be used for the model.
        """
        super().__init__(state_dict=state_dict, weights=weights)
        self.name = "Segformer"
        self.description = "Segformer model for pixel classification"
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    
    def on_batch_begin(self, learn, model_input_batch: torch.Tensor, model_target_batch: torch.Tensor):
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
        # Ensure 4D
        if xb.ndim == 3:
            xb = xb.unsqueeze(0)

        xb = xb.float()
        xb = xb[:, :3]
        if xb.max() > 1.5:
            xb = xb / 255.0
        xb = xb.clamp(0.0, 1.0)

        return (xb - self.mean.to(xb.device)) / self.std.to(xb.device)

    def transform_input_multispectral(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Function to transform the multispectral inputs for inferencing.
        Args:
            xb: fastai transformed batch of input images: tensor of shape [N, C, H, W],
             where N - batch size C - number of channels (bands) in the image H - height of the image W - width of the image
        """
        if len(xb.shape) == 3:
            xb = xb.unsqueeze(0)
        xb = xb[:, :3, :, :]
    
        # Normalize using ImageNet stats
        max_val = xb.max()
        if max_val > 1.5:
            xb = xb / 255.0
        
        return (xb - self.mean.to(xb.device)) / self.std.to(xb.device)

    def get_model(self, backbone="nvidia/segformer-b2-finetuned-ade-512-512", image_size=512, **kwargs) -> Module:
        """
        Function used to define the model architecture.
        Args:
            data: DataBunch object created in the prepare_data function
            kwargs: Additional key word arguments to be passed to the model
        """
        self.backbone = backbone
        if self.model is not None:
            return self.model
        
        class SegformerDecodeCore(nn.Module):
            """
            Recreate SegFormer decode head up to (but not including) classifier.
            Uses safe reshape/permute operations to avoid .view() stride errors.
            """
            def __init__(self, decode_head):
                super().__init__()
                # reuse the modules from HF decode head
                self.linear_c = decode_head.linear_c       # ModuleList of SegformerMLP
                self.linear_fuse = decode_head.linear_fuse # Conv2d
                self.batch_norm = decode_head.batch_norm
                self.activation = decode_head.activation
                self.dropout = decode_head.dropout

            def forward(self, features, debug=False):
                # features: list/tuple of 4 tensors: [(B,C1,H1,W1), (B,C2,H2,W2), ...]
                # target spatial size: use features[0] (largest spatial)
                target_h, target_w = features[0].shape[-2], features[0].shape[-1]
                projected = []
                for i, proj_mlp in enumerate(self.linear_c):
                    x = features[i]  # (B, Ci, Hi, Wi)
                    if (x.shape[-2], x.shape[-1]) != (target_h, target_w):
                        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
                    
                    # safe flatten+linear: (B, Ci, H*W) -> (B, H*W, Ci) -> proj -> (B, H*W, hidden) -> back
                    b, c, h, w = x.shape
                    x_flat = x.reshape(b, c, h*w).permute(0, 2, 1)   # (B, H*W, Ci)
                    # Ensure contiguous before Linear just in case
                    if not x_flat.is_contiguous():
                        x_flat = x_flat.contiguous()
                    x_proj = proj_mlp.proj(x_flat)                   # (B, H*W, hidden)
                    x_proj = x_proj.permute(0, 2, 1).reshape(b, -1, h, w)  # (B, hidden, H, W)
                    x_proj = x_proj.contiguous() if not x_proj.is_contiguous() else x_proj
                    projected.append(x_proj)

                    if debug:
                        print(f"proj[{i}] -> shape {x_proj.shape}, contiguous={x_proj.is_contiguous()}")
                
                x = torch.cat(projected, dim=1)  # (B, sum(hidden), H, W)
                x = self.linear_fuse(x)
                x = self.batch_norm(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = x.contiguous() if not x.is_contiguous() else x
                if debug:
                    print("fused ->", x.shape, "contig=", x.is_contiguous())
                return x
    
        class SegFormerModel(Module):
            """
            SegFormer model for semantic segmentation.
            Uses a pretrained SegFormer backbone and replaces the decode head to upsample to the input image size.
            
            https://github.com/NVlabs/SegFormer
            """
            def __init__(self, num_classes=1, input_image_size=512, backbone="nvidia/segformer-b2-finetuned-ade-512-512"):
                super().__init__()
                self.num_classes = num_classes
                self.input_image_size = input_image_size
                self.weights = backbone

                # --- backbone: MiT encoder only ---
                self.backbone = SegformerModel.from_pretrained(backbone)
                config = self.backbone.config

                # --- create a decode_head from config and reuse its components ---
                hf_decode_head = SegformerDecodeHead(config)
                hf_decode_head.classifier = nn.Identity() # type: ignore
                self.decode_core = SegformerDecodeCore(hf_decode_head)

                # --- classifier head ---
                self.hidden_size = config.decoder_hidden_size
                self.classifier = nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 2, kernel_size=4, stride=2, padding=1), 
                    nn.BatchNorm2d(self.hidden_size // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(self.hidden_size // 2, self.hidden_size // 4, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(self.hidden_size // 4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.hidden_size // 4, num_classes, kernel_size=3, padding=1),
        )
            def _get_backbone_features(self, x):
                # call backbone asking for hidden_states
                # HuggingFace SegformerModel returns hidden_states as a tuple of stage outputs when output_hidden_states=True
                outputs = self.backbone(x, output_hidden_states=True)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    hs = list(outputs.hidden_states)
                    # Many HF versions include embeddings; the stage outputs are often the last 4 entries.
                    if len(hs) >= 4:
                        hs = hs[-4:]
                    return hs
                # Fallback: if outputs.last_hidden_state is already a fused map, try to adapt
                if hasattr(outputs, "last_hidden_state"):
                    # some versions may return (B, seq_len, hidden) — attempt to reshape to (B, hidden, H, W)
                    lh = outputs.last_hidden_state
                    # If it looks like a feature map already (4D), return as single element list
                    if lh.dim() == 4:
                        return [lh]
                raise RuntimeError("Couldn't extract backbone hidden states. Inspect `outputs` returned by SegformerModel.")

            def forward(self, image):
                # enforce 3-channel
                if image.shape[1] > 3:
                    image = image[:, :3, :, :]
                with autocast(next(self.classifier.parameters()).device.type):
                    features = self._get_backbone_features(image)
                    core = self.decode_core(features)  # (B, hidden_size, H/4, W/4)
                    out = self.classifier(core)

                # upsample to input size
                if out.shape[-2:] != image.shape[-2:]:
                    out = F.interpolate(out, size=image.shape[-2:], mode="bilinear", align_corners=False)
                return out

        num_classes = kwargs.get("num_classes", 1)

        segformer = SegFormerModel(num_classes=num_classes, input_image_size=image_size, backbone=self.backbone)

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
        else:
            print("No state_dict found. Initializing model with random weights.")
        
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

    def post_process(self, pred: torch.Tensor, thres: float = 0.5) -> torch.Tensor:
        """
        Post-process raw model outputs for binary or multi-class segmentation.
        Always returns [N, H, W] label masks.
        """
        pred = pred.detach()
        # Ensure 4D shape: [N, C, H, W]
        if pred.ndim == 2:           # [H, W]
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 3:         # could be [N, H, W] or [1, H, W]
            if pred.shape[0] > 1:    # [N, H, W]
                pred = pred.unsqueeze(1)  # -> [N, 1, H, W]
            else:
                pred = pred.unsqueeze(0)  # -> [1, 1, H, W]

        # Apply output processing
        N, C, H, W = pred.shape
        if C == 1: # Binary segmentation
            prob = torch.sigmoid(pred)
            mask = (prob > thres).long()
            return mask.squeeze(1)      # -> [N, H, W]
        else: # Multi-class segmentation
            return torch.argmax(pred, dim=1)  # -> [N, H, W]