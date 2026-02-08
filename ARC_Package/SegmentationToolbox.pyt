# -*- coding: utf-8 -*-
"""
Semantic Segmentation Toolbox (Dynamic EMD-Driven)

Drop this file into a folder with:
    - ModelClasses.py   (defines ModelClass, ResNetUNet, SegFormer, etc.)
    - One or more .emd  (ESRI model definition files)
    - Corresponding .pth model weight files

ArcGIS Pro will detect this as a Python toolbox (.pyt).
"""

import arcpy  # type: ignore
import os
import re
import json
import sys
import gc
import tempfile
import numpy as np
from osgeo import gdal, gdalconst  # type: ignore
import torch

sys.path.insert(0, os.path.dirname(__file__))
import models  # local import of model classes
from models import __all__ as model_class_names, ModelClass

# ------------------------------------------------------------------------------
# Toolbox path / imports
# ------------------------------------------------------------------------------
def process_raster(input_path, output_path,
                   model_class, model,
                   device, tile_size, overlap, batch_size,
                   nodata_value, extract_bands, threshold=0.5,
                   use_tta: bool = False) -> str:
    """
    Process raster with tiling using smooth blending, keeping all heavy
    computation in PyTorch tensors (optionally on GPU).
    """

    model.eval()
    use_amp = (device.type == "cuda")

    # ------------------------------------------------------------------
    # Output handling (GDB vs TIFF)
    # ------------------------------------------------------------------
    is_gdb = ('.gdb' in output_path.lower()) or ('.sde' in output_path.lower())
    if is_gdb:
        temp_dir = tempfile.gettempdir()
        temp_output = os.path.join(temp_dir, "temp_output.tif")
        arcpy.AddMessage(f"  Creating temporary file: {temp_output}")
        actual_output = temp_output
    else:
        actual_output = output_path

    # ------------------------------------------------------------------
    # Open input raster
    # ------------------------------------------------------------------
    src_ds = gdal.Open(input_path, gdalconst.GA_ReadOnly)
    if src_ds is None:
        raise ValueError(f"Cannot open raster: {input_path}")

    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    total_bands = src_ds.RasterCount

    if extract_bands:
        band_indices = [b for b in extract_bands if 1 <= b <= total_bands]
    else:
        band_indices = list(range(1, min(total_bands, 3) + 1))

    n_bands = len(band_indices)
    arcpy.AddMessage(f"  Size: {width} x {height} pixels")
    arcpy.AddMessage(f"  Bands used: {n_bands} (indices: {band_indices})")

    # Pre-fetch band handles & nodata
    bands = []
    for bidx in band_indices:
        band = src_ds.GetRasterBand(bidx)
        bands.append(band)

    # ------------------------------------------------------------------
    # Create output raster (always GeoTIFF here)
    # ------------------------------------------------------------------
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(
        actual_output, width, height, 1, gdal.GDT_Byte,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
    )
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.SetNoDataValue(int(nodata_value))

    # ------------------------------------------------------------------
    # Tiling setup
    # ------------------------------------------------------------------
    stride = tile_size - overlap
    n_tiles_x = int(np.ceil(width / stride))
    n_tiles_y = int(np.ceil(height / stride))
    total_tiles = n_tiles_x * n_tiles_y

    arcpy.AddMessage(f"  Processing {total_tiles} tiles ({n_tiles_x} x {n_tiles_y})")
    if use_tta:
        arcpy.AddMessage("  TTA enabled")
    arcpy.SetProgressor("step", "Processing tiles...", 0, total_tiles, 1)

    # ------------------------------------------------------------------
    # Determine num_classes (single dummy forward)
    # ------------------------------------------------------------------
    with torch.no_grad():
        dummy = torch.zeros((1, n_bands, tile_size, tile_size),
                            dtype=torch.float32, device=device)
        dummy_out = model(dummy)

        if isinstance(dummy_out, (tuple, list)):
            dummy_out = dummy_out[0]

        if dummy_out.ndim == 3:
            num_classes = 1
        else:
            num_classes = dummy_out.shape[1]

        del dummy, dummy_out

    # ------------------------------------------------------------------
    # Global accumulators
    # ------------------------------------------------------------------
    accum = torch.zeros((num_classes, height, width),
                        dtype=torch.float32, device=device)
    weight = torch.zeros((height, width),
                         dtype=torch.float32, device=device)

    # Edge confidence tapering (precomputed, reused)
    edge_weight_mask_tensor = compute_edge_weight(
        tile_size=tile_size,
        padding=overlap,
        device=device
    )

    # TTA transforms indices
    tta_transforms = [0]
    if use_tta:
        # 4 rotations: 0, 90, 180, 270
        tta_transforms = [0, 1, 2, 3]

    processed = 0

    # Pre-allocate tile_logits once & reuse (avoid re-alloc per tile)
    tile_logits = torch.zeros(
        (num_classes, tile_size, tile_size),
        dtype=torch.float32,
        device=device
    )

    # ------------------------------------------------------------------
    # Main tile loop (torch accumulation)
    # ------------------------------------------------------------------
    with torch.no_grad():
        for tile_idx in range(total_tiles):
            try:
                ty, tx = divmod(tile_idx, n_tiles_x)
                x_off = tx * stride
                y_off = ty * stride
                x_size = min(tile_size, width - x_off)
                y_size = min(tile_size, height - y_off)

                # -----------------------------
                # Read tile (NumPy, minimal)
                # -----------------------------
                tile_data_np = np.zeros((n_bands, tile_size, tile_size),
                                        dtype=np.float32)
                tile_mask_np = np.ones((tile_size, tile_size),
                                       dtype=np.float32)

                for i, band in enumerate(bands):
                    data = band.ReadAsArray(x_off, y_off, x_size, y_size)
                    if data is not None:
                        tile_data_np[i, :y_size, :x_size] = data
                        if nodata_value is not None:
                            valid = (data != nodata_value).astype(np.float32)
                            tile_mask_np[:y_size, :x_size] *= valid

                # Skip tile if all NoData
                if tile_mask_np.max() == 0 or tile_data_np.max() == 0:
                    processed += 1
                    if processed % 10 == 0:
                        arcpy.SetProgressorPosition(processed)
                    continue

                # -----------------------------
                # Convert to torch tensors
                # -----------------------------
                tile_data = torch.from_numpy(tile_data_np).to(
                    device=device, dtype=torch.float32
                )  # [C, H, W]
                tile_mask = torch.from_numpy(tile_mask_np).to(
                    device=device, dtype=torch.float32
                )  # [H, W]

                # Zero out invalid pixels in the input
                tile_data *= tile_mask.unsqueeze(0)

                # -----------------------------
                # TTA loop
                # -----------------------------
                tile_logits.zero_()  # reuse preallocated tensor

                for tta_idx in tta_transforms:
                    # Apply transform in tensor space
                    t_in = apply_tta_transform(tile_data, tta_idx)  # [C,H,W]

                    # ModelClass normalization
                    t_in = model_class.transform_input(t_in)
                    # Ensure [1,C,H,W] on device
                    if t_in.ndim == 3:
                        t_in = t_in.unsqueeze(0)
                    t_in = t_in.to(device=device, dtype=torch.float32)

                    # AMP only if CUDA & requested
                    with torch.amp.autocast_mode.autocast(
                        device_type=device.type,
                        enabled=use_amp
                    ):
                        out = model(t_in)  # [1,C',H,W] or similar

                    if isinstance(out, (tuple, list)):
                        out = out[0]

                    # Ensure shape [C, H, W]
                    if out.ndim == 2:
                        out = out.unsqueeze(0)  # [1,H,W]
                    if out.ndim == 3:
                        pass  # [C,H,W]
                    elif out.ndim == 4:
                        out = out.squeeze(0)  # [C,H,W]
                    else:
                        raise RuntimeError(
                            f"Unexpected model output ndim={out.ndim}"
                        )

                    # Reverse TTA
                    out = reverse_tta_transform(out, tta_idx)  # [C,H,W]

                    # Align channels with num_classes
                    if out.shape[0] == 1 and num_classes > 1:
                        out = out.repeat(num_classes, 1, 1)
                    elif out.shape[0] != num_classes:
                        if num_classes == 1:
                            out = out[0:1, :, :]
                        else:
                            raise RuntimeError(
                                f"Model output channels ({out.shape[0]}) "
                                f"!= num_classes ({num_classes})"
                            )

                    tile_logits += out

                # Average across TTA
                tile_logits /= float(len(tta_transforms))

                # -----------------------------
                # Blending weights (torch)
                # -----------------------------
                y_end = min(y_off + tile_size, height)
                x_end = min(x_off + tile_size, width)
                tile_h = y_end - y_off
                tile_w = x_end - x_off
                tile_mask_sub = tile_mask[:tile_h, :tile_w]
                
                # dont blend if no data in tile
                has_nodata = (tile_mask_sub < 1.0).any().item()
                if has_nodata:
                    tile_weight = tile_mask_sub
                else:
                    tile_weight = edge_weight_mask_tensor[:tile_h, :tile_w] * tile_mask_sub

                # Apply weight to logits
                weighted_logits = tile_logits[:, :tile_h, :tile_w] * tile_weight.unsqueeze(0)

                # Accumulate logits and weights
                accum[:, y_off:y_end, x_off:x_end] += weighted_logits
                weight[y_off:y_end, x_off:x_end] += tile_weight

                processed += 1

                if processed % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if processed % 10 == 0:
                    arcpy.SetProgressorPosition(processed)
                    arcpy.AddMessage(
                        f"  Progress: {processed}/{total_tiles} "
                        f"({100.0 * processed / total_tiles:.1f}%)"
                    )

            except Exception as e:
                arcpy.AddWarning(f"  Failed to process tile {tile_idx}: {e}")
                processed += 1

    arcpy.ResetProgressor()

    # ------------------------------------------------------------------
    # Final blending & post-processing (torch)
    # ------------------------------------------------------------------
    valid = (weight > 0.0)  # [H,W]

    blended_t = torch.zeros_like(accum)  # [C,H,W]
    blended_t[:, valid] = accum[:, valid] / weight[valid]

    # ModelClass post_process expects [N,C,H,W] or similar
    pred_t = model_class.post_process(
        blended_t.unsqueeze(0),  # [1,C,H,W]
        thres=threshold
    )

    # Ensure [H,W]
    while pred_t.ndim > 2:
        pred_t = pred_t.squeeze(0)

    pred_t = pred_t.to(torch.int64)

    # Apply nodata where there was no valid weight
    pred_t = pred_t.clone()
    pred_t[~valid] = int(nodata_value)

    # Convert once to NumPy for GDAL
    final_np = pred_t.cpu().numpy().astype(np.uint8)

    dst_band.WriteArray(final_np)
    dst_band.FlushCache()
    dst_ds.FlushCache()

    # Close GDAL datasets before using ArcPy geoprocessing
    dst_band = None
    dst_ds = None
    src_ds = None

    arcpy.AddMessage(f"Processed {total_tiles} tiles with tensorized blending")

    # ------------------------------------------------------------------
    # If target is a GDB/SDE, copy the temp TIFF into it
    # ------------------------------------------------------------------
    if is_gdb:
        try:
            arcpy.AddMessage(f"Copying temporary TIFF to geodatabase:\n  {output_path}")

            # Overwrite any existing dataset with same name if needed
            if arcpy.Exists(output_path):
                arcpy.management.Delete(output_path)

            arcpy.management.CopyRaster(
                in_raster=actual_output,
                out_rasterdataset=output_path,
                pixel_type="8_BIT_UNSIGNED",
                nodata_value=nodata_value,
                colormap_to_RGB="NONE"
            )

            arcpy.AddMessage("✓ Copied classification raster into geodatabase")
            actual_output = output_path  # return the GDB path

        except Exception as e:
            arcpy.AddWarning(f"⚠ Failed to copy TIFF into geodatabase: {e}")

    return actual_output

def compute_edge_weight(tile_size: int, padding: int, device):
    """
    Create a [tile_size, tile_size] edge weight mask (torch) that tapers
    weights toward the tile borders within `padding` pixels.
    """
    import torch
    weight = torch.ones((tile_size, tile_size),
                        dtype=torch.float32, device=device)
    if padding <= 0:
        return weight

    max_p = min(padding, tile_size // 2)

    # vertical fade (top & bottom)
    for i in range(max_p):
        factor = float(i + 1) / float(max_p)
        weight[i, :] *= factor
        weight[tile_size - 1 - i, :] *= factor

    # horizontal fade (left & right)
    for j in range(max_p):
        factor = float(j + 1) / float(max_p)
        weight[:, j] *= factor
        weight[:, tile_size - 1 - j] *= factor

    return weight

def apply_tta_transform(x: torch.Tensor, idx: int) -> torch.Tensor:
    """
    Apply dihedral-style transform in torch.
    x: [C,H,W].
    """
    # rotations
    if idx == 0: return x
    elif idx == 1: return torch.rot90(x, k=1, dims=(1, 2))
    elif idx == 2: return torch.rot90(x, k=2, dims=(1, 2))
    elif idx == 3: return torch.rot90(x, k=3, dims=(1, 2))
    # flips
    elif idx == 4: return torch.flip(x, dims=(2,))          # horizontal
    elif idx == 5: return torch.flip(x, dims=(1,))          # vertical
    elif idx == 6: return torch.rot90(torch.flip(x, dims=(2,)), k=1, dims=(1, 2))
    elif idx == 7: return torch.rot90(torch.flip(x, dims=(1,)), k=1, dims=(1, 2))
    else: return x

def reverse_tta_transform(x: torch.Tensor, idx: int) -> torch.Tensor:
    """
    Inverse of apply_tta_transform. x: [C,H,W]
    """
    # inverse of 0 is 0
    if idx == 0: return x
    # inverse of rot90(k) is rot90(4-k)
    elif idx == 1: return torch.rot90(x, k=3, dims=(1, 2))
    elif idx == 2: return torch.rot90(x, k=2, dims=(1, 2))
    elif idx == 3: return torch.rot90(x, k=1, dims=(1, 2))
    elif idx == 4: return torch.flip(x, dims=(2,))
    elif idx == 5: return torch.flip(x, dims=(1,))
    elif idx == 6:
        # inverse of flip_h + rot90 is rot270 + flip_h
        x = torch.rot90(x, k=3, dims=(1, 2))
        return torch.flip(x, dims=(2,))
    elif idx == 7:
        # inverse of flip_v + rot90 is rot270 + flip_v
        x = torch.rot90(x, k=3, dims=(1, 2))
        return torch.flip(x, dims=(1,))
    else: return x

try:
    TOOLBOX_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback if __file__ is not defined
    TOOLBOX_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

if TOOLBOX_DIR not in sys.path:
    sys.path.insert(0, TOOLBOX_DIR)

# ------------------------------------------------------------------------------
# Model Registry
# ------------------------------------------------------------------------------

class ModelRegistry:
    """Scans a folder for EMD files and provides query helpers."""

    def __init__(self):
        self.models = []  # list of dicts (emd contents + metadata)

    def scan_folder(self, folder: str):
        """Scan folder (recursively) for .emd and parse them."""
        self.models = []
        if not folder:
            return self.models

        folder = os.path.abspath(folder)

        for root, _, files in os.walk(folder):
            for fn in files:
                if not fn.lower().endswith(".emd"):
                    continue

                emd_path = os.path.join(root, fn)
                try:
                    emd = load_emd(emd_path)
                    # Attach metadata
                    emd["emd_path"] = emd_path

                    # ModelFile may be relative; store full .pth path
                    model_file = emd.get("ModelFile")
                    if model_file is not None:
                        if os.path.isabs(model_file):
                            pth_path = model_file
                        else:
                            pth_path = os.path.join(root, model_file)
                    else:
                        pth_path = None
                    emd["pth_path"] = pth_path

                    self.models.append(emd)
                except Exception as e:
                    # For toolbox: be quiet, but you could log if needed
                    print(f"Error parsing EMD {emd_path}: {e}")
        return self.models

    # ---- Query functions ----

    def list_tasks(self):
        """All unique task strings (as in emd['Task'])."""
        return sorted({m.get("Task", "").strip() for m in self.models if "Task" in m})

    def list_architectures(self, task=None):
        """Unique ModelConfiguration, optionally filtered by task."""
        if task:
            archs = {m.get("ModelConfiguration", "").strip()
                     for m in self.models
                     if m.get("Task", "").strip() == task}
        else:
            archs = {m.get("ModelConfiguration", "").strip()
                     for m in self.models if "ModelConfiguration" in m}
        return sorted(a for a in archs if a)

    def list_models(self, task=None, architecture=None):
        """Return a list of EMD dicts matching optional task & architecture."""
        result = []
        for m in self.models:
            if task and m.get("Task", "").strip() != task:
                continue
            if architecture and m.get("ModelConfiguration", "").strip() != architecture:
                continue
            result.append(m)
        return result

    def list_backbones(self, task=None, architecture=None):
        """Unique ModelBackbone values for given task & architecture."""
        backs = set()
        for m in self.models:
            if task and m.get("Task", "").strip() != task:
                continue
            if architecture and m.get("ModelConfiguration", "").strip() != architecture:
                continue
            b = m.get("ModelBackbone")
            if b:
                backs.add(b)
        return sorted(backs)

    def get_classes(self, model_emd):
        """Return class names from EMD['Classes']."""
        classes = model_emd.get("Classes", [])
        names = []
        for c in classes:
            name = c.get("Name")
            if isinstance(name, str):
                names.append(name)
        return names

    def get_image_size(self, model_emd):
        """Get image size (height) from EMD, default 512."""
        return int(model_emd.get("ImageHeight", 512))

    def get_extract_bands(self, model_emd):
        """Return ExtractBands list (1-based indices) or None."""
        bands = model_emd.get("ExtractBands")
        if isinstance(bands, list) and bands:
            return [int(b) for b in bands]
        return None

    def get_pth_path(self, model_emd):
        return model_emd.get("pth_path")

# EMD parsing helpers
def _clean_emd_json(text: str) -> str:
    """Clean EMD text into valid JSON:
    - Remove comments
    - Remove trailing commas
    - Fix missing commas between JSON string pairs
    """

    # Remove single-line // comments
    text = re.sub(r'//.*', '', text)
    # Remove /* ... */ comments
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)

    # Remove trailing commas before ] or }
    text = re.sub(r',\s*(\]|\})', r'\1', text)

    # Fix missing commas between JSON string pairs:
    #   "key": "value"  "next": "value"
    # -> "key": "value", "next": "value"
    text = re.sub(r'\"(\s*:\s*\"[^\"]*\"\s*)\"', r'\1, \"', text)

    return text


def load_emd(path: str) -> dict:
    """Load an EMD file into a Python dict (robust to ESRI quirks)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = _clean_emd_json(raw)
    return json.loads(cleaned)

# ------------------------------------------------------------------------------
# Toolbox
# ------------------------------------------------------------------------------

class Toolbox(object):
    def __init__(self):
        self.label = "Semantic Segmentation"
        self.alias = "semantic_segmentation_toolbox"
        self.tools = [
            Classify,
            ModelInfo
        ]

# ------------------------------------------------------------------------------
# Classify Tool (main raster classification)
# ------------------------------------------------------------------------------

class Classify(object):
    """Main classification tool for raster processing"""

    def __init__(self):
        self.label = "Classify Raster"
        self.description = "Perform semantic segmentation on a raster using a trained model"
        self.canRunInBackground = False

        # Dynamic registry of models from EMDs
        self.registry = ModelRegistry()

    # ------------------------------------------------------------------
    # Parameter definitions
    # ------------------------------------------------------------------
    def getParameterInfo(self):
        """Define the tool parameters."""
        params = []

        # 0. Folder Containing Models
        p_model_folder = arcpy.Parameter(
            displayName="Folder Containing Models",
            name="model_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        p_model_folder.value = TOOLBOX_DIR  # default to toolbox dir
        params.append(p_model_folder)

        # 1. Model Task (from EMD["Task"])
        p_task = arcpy.Parameter(
            displayName="Model Task",
            name="model_task",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p_task.filter.type = "ValueList"
        p_task.filter.list = []
        params.append(p_task)

        # 2. Model Architecture (from EMD["ModelConfiguration"])
        p_arch = arcpy.Parameter(
            displayName="Model Architecture",
            name="model_architecture",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p_arch.filter.type = "ValueList"
        p_arch.filter.list = []
        params.append(p_arch)

        # 3. Model File (.pth) (from EMD["ModelFile"])
        p_model_file = arcpy.Parameter(
            displayName="Trained Model File (.pth)",
            name="model_file",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p_model_file.filter.type = "ValueList"
        p_model_file.filter.list = []
        params.append(p_model_file)

        # 4. Input Raster
        p_in_raster = arcpy.Parameter(
            displayName="Input Raster",
            name="input_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )
        params.append(p_in_raster)

        # 5. Output Raster
        p_out_raster = arcpy.Parameter(
            displayName="Output Classified Raster (.tif or GDB)",
            name="output_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        params.append(p_out_raster)

        # --- Processing Options Category ---

        # 6. Tile Size
        p_tile_size = arcpy.Parameter(
            displayName="Tile Size (pixels)",
            name="tile_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        p_tile_size.value = 512
        p_tile_size.category = "Processing Options"
        params.append(p_tile_size)

        # 7. Tile Overlap
        p_tile_overlap = arcpy.Parameter(
            displayName="Tile Overlap (pixels)",
            name="tile_overlap",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        p_tile_overlap.value = 64
        p_tile_overlap.category = "Processing Options"
        params.append(p_tile_overlap)

        # 8. Batch Size
        p_batch = arcpy.Parameter(
            displayName="Batch Size",
            name="batch_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        p_batch.value = 8
        p_batch.filter.type = "Range"
        p_batch.filter.list = [1, 64]
        p_batch.category = "Processing Options"
        params.append(p_batch)

        # 9. TTA
        p_tta = arcpy.Parameter(
            displayName="TTA (Test Time Augmentation)",
            name="tta_enabled",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        p_tta.value = False
        p_tta.category = "Processing Options"
        params.append(p_tta)

        

        # 10. Use GPU
        p_use_gpu = arcpy.Parameter(
            displayName="Use GPU if available",
            name="use_gpu",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        p_use_gpu.value = True
        p_use_gpu.category = "Processing Options"
        params.append(p_use_gpu)

        # --- Model Configuration Category ---

        # 11. Pretrained Backbone (from EMD["ModelBackbone"])
        p_backbone = arcpy.Parameter(
            displayName="Pretrained Backbone",
            name="pretrained_backbone",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        p_backbone.filter.type = "ValueList"
        p_backbone.filter.list = []
        p_backbone.value = None
        p_backbone.category = "Model Configuration"
        params.append(p_backbone)

        # --- Output Configuration Category ---

        # 12. Class Names
        p_class_names = arcpy.Parameter(
            displayName="Class Names (comma-separated)",
            name="class_names",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        p_class_names.category = "Output Configuration"
        params.append(p_class_names)

        # 13. NoData Value
        p_nodata = arcpy.Parameter(
            displayName="NoData Value",
            name="nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        p_nodata.value = 255
        p_nodata.category = "Output Configuration"
        params.append(p_nodata)

        # Initial attempt to populate lists from toolbox dir
        try:
            folder = p_model_folder.valueAsText or TOOLBOX_DIR
            self.registry.scan_folder(folder)
            tasks = self.registry.list_tasks()
            p_task.filter.list = tasks
            if tasks:
                p_task.value = tasks[0]

            archs = self.registry.list_architectures(task=p_task.valueAsText) if tasks else []
            p_arch.filter.list = archs
            if archs:
                p_arch.value = archs[0]

            models_emd = self.registry.list_models(task=p_task.valueAsText, architecture=p_arch.valueAsText) if archs else []
            model_files = [m.get("ModelFile") for m in models_emd if m.get("ModelFile")]
            p_model_file.filter.list = model_files
            if model_files:
                p_model_file.value = model_files[0]

            if models_emd:
                first_model = models_emd[0]
                # Classes
                cls = self.registry.get_classes(first_model)
                if cls:
                    p_class_names.value = ",".join(cls)
                # Tile size
                p_tile_size.value = self.registry.get_image_size(first_model)
                # Backbones
                backs = self.registry.list_backbones(p_task.valueAsText, p_arch.valueAsText)
                p_backbone.filter.list = backs
                if backs:
                    p_backbone.value = backs[0]
        except Exception:
            # Don't crash toolbox at definition time
            pass

        return params

    # ------------------------------------------------------------------
    def isLicensed(self):
        return True

    # ------------------------------------------------------------------
    def updateParameters(self, parameters):
        """Update parameters dynamically based on folder / task / architecture selection."""
        p_folder = parameters[0]
        p_task = parameters[1]
        p_arch = parameters[2]
        p_model_file = parameters[3]
        p_tile_size = parameters[6]
        p_batch = parameters[8]
        p_backbone = parameters[11]
        p_class_names = parameters[12]

        # Helper: refresh registry from folder
        def refresh_registry():
            folder = p_folder.valueAsText
            if folder:
                self.registry.scan_folder(folder)
            else:
                self.registry.models = []

        # 1. Folder changed -> rescan, update tasks, archs, model files, backbones, classes
        if p_folder.altered:
            try:
                refresh_registry()
                tasks = self.registry.list_tasks()
                p_task.filter.list = tasks
                if tasks:
                    if p_task.valueAsText not in tasks:
                        p_task.value = tasks[0]
                else:
                    p_task.value = None

                archs = self.registry.list_architectures(p_task.valueAsText) if p_task.valueAsText else []
                p_arch.filter.list = archs
                if archs:
                    if p_arch.valueAsText not in archs:
                        p_arch.value = archs[0]
                else:
                    p_arch.value = None

                models_emd = self.registry.list_models(
                    p_task.valueAsText,
                    p_arch.valueAsText
                )
                model_files = [m.get("ModelFile") for m in models_emd if m.get("ModelFile")]
                p_model_file.filter.list = model_files
                if model_files and p_model_file.valueAsText not in model_files:
                    p_model_file.value = model_files[0]

                if models_emd:
                    first_model = models_emd[0]
                    # Update classes if user hasn't edited
                    cls = self.registry.get_classes(first_model)
                    p_class_names.value = ",".join(cls)
                    # Update tile size if user hasn't edited
                    if not p_tile_size.altered:
                        p_tile_size.value = self.registry.get_image_size(first_model)
                    # Update backbones
                    backs = self.registry.list_backbones(p_task.valueAsText, p_arch.valueAsText)
                    p_backbone.filter.list = backs
                    if backs and p_backbone.valueAsText not in backs:
                        p_backbone.value = backs[0]
                else:
                    p_backbone.filter.list = []
                    if not p_backbone.altered:
                        p_backbone.value = None
            except Exception:
                pass

        # 2. Task changed -> refresh archs, model files, backbones, classes
        if p_task.altered and not p_folder.altered:
            try:
                refresh_registry()
                archs = self.registry.list_architectures(p_task.valueAsText)
                p_arch.filter.list = archs
                if archs:
                    if p_arch.valueAsText not in archs:
                        p_arch.value = archs[0]
                else:
                    p_arch.value = None

                models_emd = self.registry.list_models(
                    p_task.valueAsText,
                    p_arch.valueAsText
                )
                model_files = [m.get("ModelFile") for m in models_emd if m.get("ModelFile")]
                p_model_file.filter.list = model_files
                if model_files and p_model_file.valueAsText not in model_files:
                    p_model_file.value = model_files[0]

                if models_emd:
                    first_model = models_emd[0]
                    cls = self.registry.get_classes(first_model)
                    p_class_names.value = ",".join(cls)
                    if not p_tile_size.altered:
                        p_tile_size.value = self.registry.get_image_size(first_model)
                    backs = self.registry.list_backbones(p_task.valueAsText, p_arch.valueAsText)
                    p_backbone.filter.list = backs
                    if backs and p_backbone.valueAsText not in backs:
                        p_backbone.value = backs[0]
                else:
                    p_backbone.filter.list = []
                    if not p_backbone.altered:
                        p_backbone.value = None
            except Exception:
                pass

        # 3. Architecture changed -> refresh model files, backbones, classes, tile size
        if p_arch.altered and not p_folder.altered:
            try:
                refresh_registry()
                models_emd = self.registry.list_models(
                    p_task.valueAsText,
                    p_arch.valueAsText
                )
                model_files = [m.get("ModelFile") for m in models_emd if m.get("ModelFile")]
                p_model_file.filter.list = model_files
                if model_files and p_model_file.valueAsText not in model_files:
                    p_model_file.value = model_files[0]

                if models_emd:
                    first_model = models_emd[0]
                    cls = self.registry.get_classes(first_model)
                    p_class_names.value = ",".join(cls)
                    if not p_tile_size.altered:
                        p_tile_size.value = self.registry.get_image_size(first_model)
                    backs = self.registry.list_backbones(p_task.valueAsText, p_arch.valueAsText)
                    p_backbone.filter.list = backs
                    if backs and p_backbone.valueAsText not in backs:
                        p_backbone.value = backs[0]
                else:
                    p_backbone.filter.list = []
                    if not p_backbone.altered:
                        p_backbone.value = None
            except Exception:
                pass

        # 4. Model file changed -> update classes / tile size / backbone from that specific EMD
        if p_model_file.altered:
            try:
                refresh_registry()
                models_emd = self.registry.list_models(
                    p_task.valueAsText,
                    p_arch.valueAsText
                )
                selected = None
                mf_name = p_model_file.valueAsText
                for m in models_emd:
                    if m.get("ModelFile") == mf_name:
                        selected = m
                        break
                if selected:
                    cls = self.registry.get_classes(selected)
                    p_class_names.value = ",".join(cls)
                    if not p_tile_size.altered:
                        p_tile_size.value = self.registry.get_image_size(selected)
                    # Backbones for that combo
                    backs = self.registry.list_backbones(p_task.valueAsText, p_arch.valueAsText)
                    p_backbone.filter.list = backs
                    if backs and not p_backbone.altered:
                        default_backbone = selected.get("ModelBackbone", backs[0])
                        if default_backbone in backs:
                            p_backbone.value = default_backbone
                        else:
                            p_backbone.value = backs[0]
            except Exception:
                pass

        return

    # ------------------------------------------------------------------
    def updateMessages(self, parameters):
        """Validate parameters and provide helpful messages."""
        p_tile_size = parameters[6]
        p_overlap = parameters[7]

        if p_tile_size.value and p_tile_size.value < 128:
            p_tile_size.setWarningMessage("Tile size < 128 may produce poor results")
        if p_overlap.value and p_tile_size.value:
            if p_overlap.value >= p_tile_size.value / 2:
                p_overlap.setErrorMessage("Overlap must be less than half the tile size")
        return

    # ------------------------------------------------------------------
    def execute(self, parameters, messages):
        model = None
        try:
            import torch
            import numpy as np
            from osgeo import gdal, gdalconst  # type: ignore

            # Read parameters
            model_folder_directory = parameters[0].valueAsText
            model_task = parameters[1].valueAsText
            model_architecture = parameters[2].valueAsText
            model_file_name = parameters[3].valueAsText
            input_raster = parameters[4].valueAsText
            output_raster = parameters[5].valueAsText
            tile_size = parameters[6].value or 512
            tile_overlap = parameters[7].value or 64
            batch_size = parameters[8].value or 4
            tta = parameters[9].value or False
            use_gpu = parameters[10].value
            backbone_param = parameters[11].valueAsText
            user_class_names_str = parameters[12].valueAsText
            nodata_value = parameters[13].value or 255

            arcpy.AddMessage("=" * 80)
            arcpy.AddMessage("Semantic Segmentation Tool")
            arcpy.AddMessage("=" * 80)
            arcpy.AddMessage(f"Model Task: {model_task}")
            arcpy.AddMessage(f"Model Architecture: {model_architecture}")
            arcpy.AddMessage(f"Selected Model: {model_file_name}")

            # Validate paths
            if not os.path.exists(input_raster):
                arcpy.AddError(f"Input raster not found: {input_raster}")
                return

            # Refresh registry and locate selected EMD entry
            self.registry.scan_folder(model_folder_directory)
            candidates = self.registry.list_models(
                task=model_task,
                architecture=model_architecture
            )
            selected_emd = None
            for m in candidates:
                if m.get("ModelFile") == model_file_name:
                    selected_emd = m
                    break

            if not selected_emd:
                # As fallback, search across all models
                for m in self.registry.models:
                    if m.get("ModelFile") == model_file_name:
                        selected_emd = m
                        break

            if not selected_emd:
                arcpy.AddError("Could not find a matching EMD entry for the selected model file.")
                return

            pth_path = self.registry.get_pth_path(selected_emd)
            if not pth_path or not os.path.exists(pth_path):
                arcpy.AddError(f"Model weight file not found: {pth_path}")
                return

            # Backbone: prefer parameter (user may override), else EMD
            emd_backbone = selected_emd.get("ModelBackbone")
            backbone = backbone_param or emd_backbone

            # Classes: if user left default / unmodified, override from EMD
            emd_classes = self.registry.get_classes(selected_emd)
            if emd_classes:
                # If user didn't alter, or appears generic, replace
                if (user_class_names_str is None or
                    user_class_names_str.strip() in ("", "Class0,Class1")):
                    class_names = emd_classes
                else:
                    class_names = [c.strip() for c in user_class_names_str.split(",")]
            else:
                class_names = [c.strip() for c in user_class_names_str.split(",")]

            # Tile size: if user didn't alter, use EMD height
            emd_img_size = self.registry.get_image_size(selected_emd)
            if not parameters[6].altered:
                tile_size = emd_img_size

            extract_bands = self.registry.get_extract_bands(selected_emd)

            arcpy.AddMessage(f"Backbone: {backbone}")
            arcpy.AddMessage(f"Model weights: {pth_path}")
            arcpy.AddMessage(f"Classes: {', '.join(class_names)}")
            if extract_bands:
                arcpy.AddMessage(f"Extract bands: {extract_bands}")
            arcpy.AddMessage(f"Tile size: {tile_size}, overlap: {tile_overlap}, batch size: {batch_size}")

            # Setup device
            arcpy.AddMessage("\n[1/5] Setting up compute device...")
            device = self._setup_device(use_gpu, messages)

            # Load model
            arcpy.AddMessage("\n[2/5] Loading model...")
            arcpy.AddMessage(f"Architecture: {model_architecture}")
            arcpy.AddMessage(f"Model file: {pth_path}")

            model_wrapper = self._load_model_class(model_architecture, messages)
            model = self._build_model(
                model_wrapper,
                class_names,
                tile_size,
                backbone,
                pth_path,
                device,
                messages
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Process raster
            arcpy.AddMessage("\n[3/5] Reading input raster...")
            arcpy.AddMessage(f"Input: {input_raster}")

            arcpy.AddMessage("\n[4/5] Running semantic segmentation...")
            arcpy.AddMessage(f"Tile size: {tile_size}x{tile_size}")
            arcpy.AddMessage(f"Overlap: {tile_overlap} pixels")
            arcpy.AddMessage(f"Batch size: {batch_size}")

            output_raster = process_raster(input_raster, output_raster,
                model_wrapper, model, device,
                tile_size, tile_overlap, batch_size,
                nodata_value, extract_bands,
                use_tta=tta
            )

            parameters[5].value = output_raster  # set output param

            arcpy.AddMessage("\n[5/5] Building pyramids and statistics...")
            self._finalize_output(output_raster, class_names, messages)

            arcpy.AddMessage("\n" + "=" * 80)
            arcpy.AddMessage("✓ Classification Complete!")
            arcpy.AddMessage(f"Output: {output_raster}")
            arcpy.AddMessage("=" * 80)

        except KeyboardInterrupt:
            arcpy.AddWarning("\nProcessing cancelled by user")

        except MemoryError:
            arcpy.AddError("\n✗ Out of memory!")
            arcpy.AddError("Try reducing tile_size or batch_size")

        except Exception as e:
            arcpy.AddError(f"\n✗ Error: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            raise

        finally:
            if model is not None:
                del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _setup_device(self, use_gpu, messages):
        import torch

        if use_gpu and torch.cuda.is_available():
            try:
                test_tensor = torch.randn(10, 10).cuda()
                del test_tensor
                torch.cuda.empty_cache()

                device = torch.device('cuda')
                arcpy.AddMessage(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
                arcpy.AddMessage(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            except Exception as e:
                arcpy.AddWarning(f"GPU test failed: {e}")
                arcpy.AddWarning("Falling back to CPU mode")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            if use_gpu and not torch.cuda.is_available():
                arcpy.AddWarning("GPU requested but not available, using CPU")
            else:
                arcpy.AddMessage("✓ Using CPU for processing")
        return device

    def _load_model_class(self, model_architecture, messages) -> ModelClass:
        """Load model wrapper class based on ModelConfiguration name."""
        try:
            if model_architecture not in model_class_names:
                raise ImportError(f"Model class '{model_architecture}' not found in ModelClasses module")

            wrapper_class = getattr(models, model_architecture)
            model_wrapper = wrapper_class()
            return model_wrapper

        except Exception as e:
            arcpy.AddError(f"Failed to load model class: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            raise

    # ------------------------------------------------------------------
    def _build_model(self, model_class: ModelClass, class_names, img_size,
                     backbone, weights, device, messages):
        import torch
        import numpy as np

        arcpy.AddMessage("  Building model ...")

        class DummyDataset:
            def __init__(self, img_size):
                self.dummy_img = np.zeros((3, img_size, img_size), dtype=np.float32)
            def __getitem__(self, idx):
                return (torch.from_numpy(self.dummy_img), None)

        class DummyData:
            def __init__(self, classes, img_size=512):
                self.classes = classes
                self.train_ds = DummyDataset(img_size)

        try:
            model = model_class.get_model(backbone=backbone, state_dict=weights, image_size=img_size, num_classes=len(class_names))
        except Exception as e:
            arcpy.AddError(f"Failed to build model: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            raise

        gc.collect()

        arcpy.AddMessage("  Moving model to device...")
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        arcpy.AddMessage("✓ Model loaded successfully")
        arcpy.AddMessage(f"  Classes: {', '.join(class_names)}")
        arcpy.AddMessage(f"  Device: {device}")

        try:
            total_params = sum(p.numel() for p in model.parameters())
            arcpy.AddMessage(f"  Parameters: {total_params:,}")
        except Exception:
            pass

        return model

    def _finalize_output(self, output_path, class_names, messages):
        """Build pyramids, statistics, and RAT if possible."""
        try:
            arcpy.AddMessage("✓ Building Raster Attribute Table...")
            # Create RAT (overwrite if exists)
            
            arcpy.management.BuildRasterAttributeTable(output_path, "Overwrite")

            # Ensure ClassName field exists
            fields = [f.name for f in arcpy.ListFields(output_path)]
            if "ClassName" not in fields:
                arcpy.management.AddField(output_path, "ClassName", "TEXT")

            # Update RAT: match VALUE -> class_names[index]
            with arcpy.da.UpdateCursor(output_path, ["VALUE", "ClassName"]) as cursor:
                for value, cls in cursor:
                    if 0 <= value < len(class_names):
                        cursor.updateRow([value, class_names[value]])
                    else:
                        cursor.updateRow([value, "Unknown"])

            arcpy.AddMessage("✓ Added class names to Raster Attribute Table")
        except Exception as e:
            arcpy.AddWarning(f"⚠ Failed to write class names to RAT: {e}")
        
        try:
            is_gdb = '.gdb' in output_path.lower() or '.sde' in output_path.lower()

            if is_gdb:
                arcpy.AddMessage("✓ Output saved to geodatabase")
                try:
                    arcpy.management.CalculateStatistics(output_path)
                    arcpy.AddMessage("✓ Calculated statistics")
                except Exception:
                    arcpy.AddMessage("  (Statistics will be calculated on first display)")
            else:
                try:
                    arcpy.management.BuildPyramids(output_path)
                    arcpy.AddMessage("✓ Built pyramids")
                except Exception as e:
                    arcpy.AddMessage(f"  Could not build pyramids: {e}")

                try:
                    arcpy.management.CalculateStatistics(output_path)
                    arcpy.AddMessage("✓ Calculated statistics")
                except Exception as e:
                    arcpy.AddMessage(f"  Could not calculate statistics: {e}")

            try:
                arcpy.management.BuildRasterAttributeTable(output_path, "Overwrite")
                 # Add field with class names (if not exists)
                fields = [f.name for f in arcpy.ListFields(output_path)]
                if "ClassName" not in fields:
                    arcpy.management.AddField(output_path, "ClassName", "TEXT")

                # Update RAT: assign class names based on raster codes
                with arcpy.da.UpdateCursor(output_path, ["VALUE", "ClassName"]) as cur:
                    for value, cls in cur:
                        if 0 <= value < len(class_names):
                            cur.updateRow([value, class_names[value]])
                        else:
                            # For safety on unexpected values
                            cur.updateRow([value, "Unknown"])

                arcpy.AddMessage("✓ Built raster attribute table")
            except Exception as e:
                arcpy.AddMessage(f"  Could not build attribute table: {e}")

        except Exception as e:
            arcpy.AddWarning(f"Post-processing warning: {e}")
            arcpy.AddMessage("  Classification completed successfully despite post-processing issues")

# ------------------------------------------------------------------------------
# ModelInfo Tool (basic pth inspection)
# ------------------------------------------------------------------------------

class ModelInfo(object):
    """Tool to inspect model information"""

    def __init__(self):
        self.label = "Inspect Model Information"
        self.description = "Display information about a trained model"
        self.canRunInBackground = False

    def getParameterInfo(self):
        params = []

        p_model_file = arcpy.Parameter(
            displayName="Model File (.pth)",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )
        p_model_file.filter.list = ['pth', 'pt']
        params.append(p_model_file)

        return params

    def execute(self, parameters, messages):
        try:
            import torch

            model_file = parameters[0].valueAsText

            arcpy.AddMessage("=" * 80)
            arcpy.AddMessage("Model Information")
            arcpy.AddMessage("=" * 80)
            arcpy.AddMessage(f"\nModel File: {model_file}")
            arcpy.AddMessage(f"File Size: {os.path.getsize(model_file) / 1e6:.2f} MB")

            checkpoint = torch.load(model_file, map_location='cpu')

            arcpy.AddMessage("\nCheckpoint Contents:")
            if isinstance(checkpoint, dict):
                for key in checkpoint.keys():
                    if key != 'state_dict':
                        arcpy.AddMessage(f"  - {key}")

                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                total_params = sum(p.numel() for p in state_dict.values())
                arcpy.AddMessage(f"\nTotal Parameters: {total_params:,}")

                for key in state_dict.keys():
                    if 'classifier' in key and 'weight' in key:
                        num_classes = state_dict[key].shape[0]
                        arcpy.AddMessage(f"Number of Classes (from classifier): {num_classes}")
                        break

                arcpy.AddMessage("\nModel Layers (top-level):")
                layer_count = {}
                for key in state_dict.keys():
                    layer_type = key.split('.')[0]
                    layer_count[layer_type] = layer_count.get(layer_type, 0) + 1

                for layer, count in sorted(layer_count.items()):
                    arcpy.AddMessage(f"  {layer}: {count} parameters")

            arcpy.AddMessage("\n" + "=" * 80)

        except Exception as e:
            arcpy.AddError(f"Error inspecting model: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())