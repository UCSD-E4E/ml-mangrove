from segmentation import SegmentationInterpreter, compare_methods
from transformers import SegformerForSemanticSegmentation
import torch
import os

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
device = "cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu"

model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model = model.to(device).eval() # type: ignore
interp = SegmentationInterpreter(model)  # auto-detects CNN vs. Transformer

path = os.path.join(os.path.dirname(__file__), "photo.png")

# Single method — defaults to Integrated Gradients
result = interp.interpret(path, class_idx=1, class_name="wall", smooth_sigma=4)
result.show()           # opens: original | heatmap | overlay panel
result.save("out.png")
result.heatmap          # raw (H, W) float32 ndarray in [0, 1]

# All methods at once (integrated_gradients, lime, shap)
results = interp.interpret_all(path, class_idx=1)
compare_methods(results, image=path)
