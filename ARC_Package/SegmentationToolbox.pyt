import arcpy # type: ignore
import os
import sys

import json
import re
from pathlib import Path

try:
    TOOLBOX_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback if __file__ is not defined
    TOOLBOX_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

if TOOLBOX_DIR not in sys.path:
    sys.path.insert(0, TOOLBOX_DIR)

import ModelClasses as models
from ModelClasses import *
from ModelClasses import ModelClass
import torch
import numpy as np
import gc

# Define supported models and their configurations
MODEL_CONFIGS = {
    "SegFormer": {
        "module": "models",
        "class_name": "SegFormer",
        "default_backbone": "nvidia/segformer-b2-finetuned-ade-512-512",
        "backbone_options": [
            "nvidia/segformer-b0-finetuned-ade-512-512",
            "nvidia/segformer-b1-finetuned-ade-512-512",
            "nvidia/segformer-b2-finetuned-ade-512-512",
            "nvidia/segformer-b3-finetuned-ade-512-512",
            "nvidia/segformer-b4-finetuned-ade-512-512",
            "nvidia/segformer-b5-finetuned-ade-640-640"
        ],
        "recommended_tile_size": 512,
        "recommended_batch_size": 16,
        "supports_multispectral": False
    },
    "ResNetUNet": {
        "module": "models",
        "class_name": "ResNetUNet",
        "default_backbone": "resnet18",
        "backbone_options": [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152"
        ],
        "recommended_tile_size": 224,
        "recommended_batch_size": 16,
        "supports_multispectral": False
    }
}

class Toolbox(object):
    def __init__(self):
        self.label = "Semantic Segmentation"
        self.alias = "semantic segmentation toolbox"
        self.tools = [
            Classify,
            ModelInfo,
            ValidateModel
        ]

class Classify(object):
    """Main classification tool for raster processing"""
    
    def __init__(self):
        self.label = "Classify Raster"
        self.description = "Perform semantic segmentation on a raster using a trained model"
        self.canRunInBackground = False
        # Define supported models and their configurations
        self.model_configs = MODEL_CONFIGS
        
    def _grab_and_parse_emds(self, task, arch, directory_path):
                """Grabs all .emd files in the mangrove_classifier folder and extracts the corresponding .pth files for models acceptable for the current classification task."""
            
                directory_path = directory_path
                file_list = []
                emd_file_count = 0
                task_dict = {'Mangroves': 'mangrove',
                                'Human Infrastructure': 'human'}
                arch_dict = {'ResNetUNet': 'ResNetUNet',
                                'SegFormer': 'SegFormer'}
            
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file_path.split('.')[-1] == 'emd':
                            emd_file_count += 1
                            # Read file
                            try:    
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    s = f.read()
                                    s = re.sub(r',\s*(\]|})', r'\1', s)
                                    
                            except IOError as exc:
                                return [f"Error reading file {file_path.name}: {exc}"]

                            json_string = json.loads(s)
                            
                            #Parse if file is appropriate for task/arch
                            is_appropriate_task = any([task_dict[task] in pair["Name"].lower() for pair in json_string["Classes"]])
                            
                            # file_list.append(f"Is appropriate?: {[task_dict[task] in pair["Name"].lower() for pair in json_string["Classes"]]}")
                            
                            is_appropriate_arch = json_string["ModelConfiguration"] == arch_dict[arch]
                            
                            # file_list.append(f"Is appropriate arch?: {json_string["Classes"] == arch_dict[arch]}")
                            
                            if (is_appropriate_task and is_appropriate_arch):
                                file_list.append(json_string["ModelFile"])
                                
                            # file_list.append(file)
                file_list = sorted(file_list) 
                if len(file_list) == 0: 
                    if emd_file_count < 1:
                        file_list = ['No models found. Check model folder.']
                    else:
                        file_list = ['No models found. Change task/architecture.']
                return file_list

    def getParameterInfo(self):
        """Define the tool parameters."""
        params = []
        
        # Folder Containing Models Selection
        params.append(arcpy.Parameter(
            displayName="Folder Containing Models",
            name="model_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"))
        params[0].value = "."
        
        # Model Task Selection
        params.append(arcpy.Parameter(
            displayName="Model Task",
            name="model_task",
            datatype="GPString",
            parameterType="Required",
            direction="Input"))
        params[1].filter.type = "ValueList"
        params[1].filter.list = ["Mangroves", "Human Infrastructure"]
        params[1].value = "Mangroves"
        
        # Model Architecture Selection
        params.append(arcpy.Parameter(
            displayName="Model Architecture",
            name="model_architecture",
            datatype="GPString",
            parameterType="Required",
            direction="Input"))
        params[2].filter.type = "ValueList"
        params[2].filter.list = list(self.model_configs.keys())
        params[2].value = "SegFormer"
        
        # Model File
        params.append(arcpy.Parameter(
            displayName="Trained Model File (.pth)",
            name="model_file",
            datatype="GPString",
            parameterType="Required",
            direction="Input"))
        # Populate initial list of .pth files from default model folder
        
        try:
            default_task = params[1].value 
            default_arch = params[2].value 
            initial_files = []
        
            initial_files = self._grab_and_parse_emds(default_task, default_arch, params[0].valueAsText)
        
        except Exception:
            initial_files = ['Error: files were not gathered.']
            
        params[3].filter.type = "ValueList"
        params[3].filter.list = initial_files
        params[3].value = initial_files[0] if initial_files else None
        
        # Input Raster
        params.append(arcpy.Parameter(
            displayName="Input Raster",
            name="input_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"))
        
        # Output Raster
        params.append(arcpy.Parameter(
            displayName="Output Classified Raster (.tif or geodatabase)",
            name="output_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"))
        
        # Processing Options Category
        params.append(arcpy.Parameter(
            displayName="Tile Size (pixels)",
            name="tile_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"))
        params[6].value = 512
        params[6].category = "Processing Options"
        
        params.append(arcpy.Parameter(
            displayName="Tile Overlap (pixels)",
            name="tile_overlap",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"))
        params[7].value = 64
        params[7].category = "Processing Options"
        
        params.append(arcpy.Parameter(
            displayName="Batch Size",
            name="batch_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"))
        params[8].value = 8
        params[8].filter.type = "Range"
        params[8].filter.list = [1, 64]
        params[8].category = "Processing Options"
        
        # Use GPU
        params.append(arcpy.Parameter(
            displayName="Use GPU if available",
            name="use_gpu",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"))
        params[9].value = True
        params[9].category = "Processing Options"

        # Model Configuration Category
        params.append(arcpy.Parameter(
            displayName="Pretrained Backbone",
            name="pretrained_backbone",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"))
        params[10].filter.type = "ValueList"
        params[10].value = None  # Default will be set dynamically
        params[10].filter.list = []  # Will be populated dynamically
        params[10].category = "Model Configuration"
        
        # Output Configuration Category
        params.append(arcpy.Parameter(
            displayName="Class Names (comma-separated)",
            name="class_names",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"))
        params[11].value = "Human Artifact,Vegetation,Building,Car,Low Vegetation,Road"
        params[11].category = "Output Configuration"
        
        # NoData Value
        params.append(arcpy.Parameter(
            displayName="NoData Value",
            name="nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"))
        params[12].value = 255
        params[12].category = "Output Configuration"
        
        # CHANGED 
        # TEST OPTION A
        # params.append(arcpy.Parameter(
        #     displayName="TEST A (folder)",
        #     name="test_a",
        #     datatype="DEFolder",
        #     parameterType="Optional",
        #     direction="Input"))
        
        # # TEST OPTION B 
        # params.append(arcpy.Parameter(
        #     displayName="Model (.pth)",
        #     name="Model",
        #     datatype="DEFile",
        #     parameterType="Required",
        #     direction="Input"))
        # params[11].filter.type = "ValueList"
        # # params[0].filter.list = list(self.model_configs.keys())
        # params[11].filter.list = ['pth', 'pt']
        
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Update parameters dynamically based on model selection"""
        
        
        # When model folder changes, refresh available model files.
        if parameters[0].altered:
            try:
                task_value = parameters[1].valueAsText or parameters[1].value
                arch_value = parameters[2].valueAsText or parameters[2].value
                model_files = self._grab_and_parse_emds(task_value, arch_value, parameters[0].valueAsText)
                parameters[3].filter.list = model_files
                cur = parameters[3].value
                if cur not in model_files:
                    parameters[3].value = model_files[0] if model_files else None
            except Exception:
                pass
        
        
        # When model task changes, refresh available model files.
        if parameters[1].altered:
            try:
                task_value = parameters[1].valueAsText or parameters[1].value
                arch_value = parameters[2].valueAsText or parameters[2].value
                model_files = self._grab_and_parse_emds(task_value, arch_value, parameters[0].valueAsText)
                parameters[3].filter.list = model_files
                cur = parameters[3].value
                if cur not in model_files:
                    parameters[3].value = model_files[0] if model_files else None
            except Exception:
                pass
        
        
        # When model architecture changes, update related parameters
        if parameters[2].altered:
            model_name = parameters[2].valueAsText
            
            if model_name in self.model_configs:
                config = self.model_configs[model_name]
                
                # Update available model files
                try:
                    task_value = parameters[1].valueAsText or parameters[1].value
                    arch_value = parameters[2].valueAsText or parameters[2].value
                    model_files = self._grab_and_parse_emds(task_value, arch_value, parameters[0].valueAsText)
                    parameters[3].filter.list = model_files
                    cur = parameters[3].value
                    if cur not in model_files:
                        parameters[3].value = model_files[0] if model_files else None
                except Exception:
                    pass
                
                # Update pretrained backbone dropdown options
                parameters[10].filter.list = config["backbone_options"]

                # Set default backbone if current value not in new list
                if parameters[10].value not in config["backbone_options"]:
                    parameters[10].value = config["default_backbone"]

                # Update recommended datasizes
                if not parameters[6].altered:
                    parameters[6].value = config["recommended_tile_size"]
                if not parameters[8].altered:
                    parameters[8].value = config["recommended_batch_size"]
                    
        return

    def updateMessages(self, parameters):
        """Validate parameters and provide helpful messages"""
        if parameters[6].value and parameters[6].value < 128:
            parameters[6].setWarningMessage("Tile size < 128 may produce poor results")
        if parameters[7].value and parameters[6].value:
            if parameters[7].value >= parameters[6].value / 2:
                parameters[7].setErrorMessage("Overlap must be less than half the tile size")
        return

    def execute(self, parameters, messages):
        model = None
        """The source code of the tool."""
        try:
            import torch
            import gc
            # Get parameters
            # CHANGED WAS ORIGINALLY 0-8
            model_folder_directory = parameters[0].valueAsText
            model_architecture = parameters[2].valueAsText
            model_file = os.path.join(model_folder_directory, parameters[3].valueAsText)
            input_raster = parameters[4].valueAsText
            output_raster = parameters[5].valueAsText
            tile_size = parameters[6].value or 512
            tile_overlap = parameters[7].value or 64
            batch_size = parameters[8].value or 4
            use_gpu = parameters[9].value
            backbone = parameters[10].valueAsText
            class_names = [c.strip() for c in parameters[11].valueAsText.split(',')]
            nodata_value = parameters[12].value or 255
            
            arcpy.AddMessage("=" * 80)
            arcpy.AddMessage("Semantic Segmentation Tool")
            arcpy.AddMessage("=" * 80)
            arcpy.AddMessage(f"Model Architecture: {model_architecture}")
            
            # Validate model architecture
            if model_architecture not in self.model_configs:
                arcpy.AddError(f"Unsupported model architecture: {model_architecture}")
                return
            
            # Validate inputs
            if not os.path.exists(input_raster):
                arcpy.AddError(f"Input raster not found: {input_raster}")
                return
            if not os.path.exists(model_file):
                arcpy.AddError(f"Model file not found: {model_file}")
                return
            
            # Setup device
            arcpy.AddMessage("\n[1/5] Setting up compute device...")
            device = self._setup_device(use_gpu, messages)
            
            # Load model
            arcpy.AddMessage("\n[2/5] Loading model...")
            arcpy.AddMessage(f"Architecture: {model_architecture}")
            arcpy.AddMessage(f"Model file: {model_file}")
            model_class = self._load_model_class(model_architecture, messages)
            model = self._build_model(model_class, class_names, tile_size, backbone, model_file, device, messages)

            # Clear memory
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
            
            self._process_raster(
                input_raster, output_raster, model_class, model, device,
                tile_size, tile_overlap, batch_size, nodata_value, messages
            )
            
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
            # Cleanup
            if model is not None:
                del model
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def _setup_device(self, use_gpu, messages):
        """Setup compute device (CPU or GPU)"""
        import torch
        
        if use_gpu and torch.cuda.is_available():
            try:
                # Test if GPU actually works
                test_tensor = torch.randn(10, 10).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                
                device = torch.device('cuda')
                arcpy.AddMessage(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
                arcpy.AddMessage(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                
                # Set conservative GPU settings
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
                arcpy.AddMessage("✓ Using CPU for processing (recommended)")
        
        return device

    def _load_model_class(self, model_architecture, messages) -> ModelClass:
        """Load the model based on architecture"""
        try:
            # Get model class dynamically
            config = self.model_configs[model_architecture]
            model_class_name = config["class_name"]
            
            # Get the model class from models module
            try:
                modelclass = getattr(models, model_class_name)
            except AttributeError:
                raise ImportError(f"Model class '{model_class_name}' not found in models module")
            
            # Initialize model wrapper
            model_wrapper = modelclass()
            return model_wrapper

        except Exception as e:
            arcpy.AddError(f"Failed to load model class: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            raise

    def _build_model(self, model_class: ModelClass, class_names, img_size, backbone, weights, device, messages) -> torch.nn.Module:
        arcpy.AddMessage("  Building model ...")
        
        # Create dummy data object
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
            # Get model architecture
            dummy_data = DummyData(class_names, img_size=img_size)
            model = model_class.get_model(dummy_data, backbone=backbone, state_dict=weights)
        except Exception as e:
            arcpy.AddError(f"Failed to build model: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            raise
        
        del dummy_data
        gc.collect()
        
        arcpy.AddMessage("  Moving model to device...")
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        arcpy.AddMessage(f"✓ Model loaded successfully")
        arcpy.AddMessage(f"  Classes: {', '.join(class_names)}")
        arcpy.AddMessage(f"  Device: {device}")
        
        try:
            total_params = sum(p.numel() for p in model.parameters())
            arcpy.AddMessage(f"  Parameters: {total_params:,}")
        except:
            pass

        return model

    def _process_raster(self, input_path, output_path, model_class: ModelClass, model: torch.nn.Module, device: torch.device,
                       tile_size: int, overlap: int, batch_size: int, nodata_value, messages):
        """Process raster with tiling"""
        import torch
        from osgeo import gdal, gdalconst # type: ignore
        import numpy as np
        import gc
        import tempfile
        import os
        
        # Check if output is geodatabase
        is_gdb = '.gdb' in output_path.lower() or '.sde' in output_path.lower()
        
        # If geodatabase, create temporary GeoTIFF first
        if is_gdb:
            temp_dir = tempfile.gettempdir()
            temp_output = os.path.join(temp_dir, "temp_output.tif")
            arcpy.AddMessage(f"  Creating temporary file: {temp_output}")
            actual_output = temp_output
        else:
            actual_output = output_path
        
        # Open input
        src_ds = gdal.Open(input_path, gdalconst.GA_ReadOnly)
        if src_ds is None:
            raise ValueError(f"Cannot open raster: {input_path}")
        
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        n_bands = min(src_ds.RasterCount, 3)
        
        arcpy.AddMessage(f"  Size: {width} x {height} pixels")
        arcpy.AddMessage(f"  Bands: {n_bands}")
        
        # Create output
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            actual_output, width, height, 1, gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
        )
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())
        dst_band = dst_ds.GetRasterBand(1)
        dst_band.SetNoDataValue(nodata_value)
        
        # Calculate tiles
        stride = tile_size - overlap
        n_tiles_x = int(np.ceil(width / stride))
        n_tiles_y = int(np.ceil(height / stride))
        total_tiles = n_tiles_x * n_tiles_y
        
        arcpy.AddMessage(f"  Processing {total_tiles} tiles ({n_tiles_x} x {n_tiles_y})")
        arcpy.SetProgressor("step", "Processing tiles...", 0, total_tiles, 1)
        
        processed = 0
        
        # Process tiles
        for tile_idx in range(total_tiles):
            try:
                ty = tile_idx // n_tiles_x
                tx = tile_idx % n_tiles_x
                
                x_off = tx * stride
                y_off = ty * stride
                x_size = min(tile_size, width - x_off)
                y_size = min(tile_size, height - y_off)
                
                # Read tile
                tile_data = np.zeros((n_bands, tile_size, tile_size), dtype=np.float32)
                for b in range(n_bands):
                    band = src_ds.GetRasterBand(b + 1)
                    data = band.ReadAsArray(x_off, y_off, x_size, y_size)
                    if data is not None:
                        tile_data[b, :y_size, :x_size] = data
                
                # Process tile
                tile_tensor = torch.from_numpy(tile_data).float()
                tile_tensor = model_class.transform_input(tile_tensor).to(device)

                with torch.no_grad():
                    output = model.forward(tile_tensor)
                    prediction = model_class.post_process(output)

                # Extract valid region
                pred = prediction[0, :y_size, :x_size]
                
                # Write result
                pred = pred.squeeze(0) if pred.dim() == 3 else pred
                pred = pred.cpu().numpy().astype(np.uint8)
                dst_band.WriteArray(pred, x_off, y_off)
                processed += 1
                
                # Cleanup
                del tile_tensor, output, prediction, pred
                if processed % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Update progress
                if processed % 10 == 0:
                    arcpy.SetProgressorPosition(processed)
                    arcpy.AddMessage(f"  Progress: {processed}/{total_tiles} tiles ({100*processed/total_tiles:.1f}%)")
          
            except Exception as e:
                arcpy.AddWarning(f"  Failed to process tile {tile_idx}: {e}")
                failed_tile = np.full((y_size, x_size), nodata_value, dtype=np.uint8)
                dst_band.WriteArray(failed_tile, x_off, y_off)
                processed += 1
        
        # Cleanup and close files
        arcpy.ResetProgressor()
        dst_band.FlushCache()
        dst_ds.FlushCache()
        dst_band = None
        dst_ds = None
        src_ds = None
        
        arcpy.AddMessage(f"✓ Processed {total_tiles} tiles")
        
        # If geodatabase output, copy temp file
        if is_gdb:
            arcpy.AddMessage(f"  Copying to geodatabase...")
            try:
                arcpy.management.CopyRaster(
                    temp_output,
                    output_path,
                    pixel_type="8_BIT_UNSIGNED",
                    nodata_value=nodata_value
                )
                arcpy.AddMessage(f"✓ Saved to geodatabase: {output_path}")
                
                # Clean up temp file
                try:
                    os.remove(temp_output)
                except:
                    pass
                    
            except Exception as e:
                arcpy.AddError(f"Failed to save to geodatabase: {e}")
                arcpy.AddError(f"Temporary output saved at: {temp_output}")
                raise
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _finalize_output(self, output_path, class_names, messages):
        # TODO: Add class names to raster attribute table
        """Build pyramids and statistics"""
        try:
            is_gdb = '.gdb' in output_path.lower() or '.sde' in output_path.lower()
            
            if is_gdb:
                arcpy.AddMessage("✓ Output saved to geodatabase")
                try:
                    arcpy.management.CalculateStatistics(output_path)
                    arcpy.AddMessage("✓ Calculated statistics")
                except:
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
                arcpy.AddMessage("✓ Built raster attribute table")
            except Exception as e:
                arcpy.AddMessage(f"  Could not build attribute table: {e}")
                
        except Exception as e:
            arcpy.AddWarning(f"Post-processing warning: {e}")
            arcpy.AddMessage("  Classification completed successfully despite post-processing issues")

class ModelInfo(object):
    """Tool to inspect model information"""
    
    def __init__(self):
        self.label = "Inspect Model Information"
        self.description = "Display information about a trained model"
        self.canRunInBackground = False
        self.model_configs = MODEL_CONFIGS

    def getParameterInfo(self):
        params = []
        
        params.append(arcpy.Parameter(
            displayName="Model File (.pth)",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"))
        params[0].filter.list = ['pth', 'pt']
        
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
                        arcpy.AddMessage(f"Number of Classes: {num_classes}")
                        break
                
                arcpy.AddMessage("\nModel Layers:")
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

class ValidateModel(object):
    """Tool to validate model can be loaded"""
    def __init__(self):
        self.label = "Validate Model"
        self.description = "Test if a model can be loaded successfully"
        self.canRunInBackground = False
        # Define supported models and their configurations
        self.model_configs = MODEL_CONFIGS

    def getParameterInfo(self):
        params = []
        
        params.append(arcpy.Parameter(
            displayName="Model File (.pth)",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"))
        params[0].filter.list = ['pth', 'pt']
        
        params.append(arcpy.Parameter(
            displayName="Number of Classes",
            name="num_classes",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"))
        params[1].value = 6

        params.append(arcpy.Parameter(
            displayName="Model Architecture",
            name="model_architecture",
            datatype="GPString",
            parameterType="Required",
            direction="Input"))
        params[2].filter.type = "ValueList"
        params[2].filter.list = list(self.model_configs.keys())
        params[2].value = "SegFormer"
        
        # Model Configuration Category
        params.append(arcpy.Parameter(
            displayName="Pretrained Backbone",
            name="pretrained_backbone",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"))
        params[3].filter.type = "ValueList"
        params[3].value = self.model_configs["SegFormer"]["default_backbone"]
        params[3].filter.list = list(self.model_configs["SegFormer"]["backbone_options"])
        params[3].category = "Model Configuration"
        
        params.append(arcpy.Parameter(
            displayName="Image Size (pixels)",
            name="image_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"))
        params[4].value = 512
        params[4].category = "Model Configuration"

        return params

    def updateParameters(self, parameters):
        """Update parameters dynamically based on model selection"""
        if parameters[2].altered:
            model_name = parameters[2].valueAsText

            if model_name in self.model_configs:
                config = self.model_configs[model_name]

                # Update pretrained backbone dropdown options
                parameters[3].filter.list = config["backbone_options"]

                # Set default backbone if current value not in new list
                if parameters[3].value not in config["backbone_options"]:
                    parameters[3].value = config["default_backbone"]

                # Update image size
                parameters[4].value = config["recommended_tile_size"]
        return

    def execute(self, parameters, messages):
        try:
            import torch
            
            model_file = parameters[0].valueAsText
            num_classes = parameters[1].value
            model_architecture = parameters[2].valueAsText
            backbone = parameters[3].valueAsText
            image_size = parameters[4].value or 512

            arcpy.AddMessage("=" * 80)
            arcpy.AddMessage("Model Validation")
            arcpy.AddMessage("=" * 80)
            
            arcpy.AddMessage("\n[1/3] Loading model checkpoint...")
            _ = torch.load(model_file, map_location='cpu')
            arcpy.AddMessage("✓ Checkpoint loaded")
            
            
            class DummyDataset:
                def __init__(self, img_size):
                    self.dummy_img = torch.zeros(3, img_size, img_size)
                def __getitem__(self, idx):
                    return (self.dummy_img, None)
            class DummyData:
                def __init__(self, n_classes, img_size=512):
                    self.classes = [f"Class{i}" for i in range(n_classes)]
                    self.train_ds = DummyDataset(img_size)


            arcpy.AddMessage("\n[2/3] Initializing model architecture...")
            dummy_data = DummyData(num_classes, img_size=image_size)
            model_wrapper : ModelClass = getattr(models, self.model_configs[model_architecture]["class_name"])()
            
            arcpy.AddMessage(f'Backbone: {backbone, backbone == "resnet18"}, model_wrapper: {model_wrapper}')
            
            model = model_wrapper.get_model(dummy_data, backbone=backbone, state_dict=model_file)
        
            
            arcpy.AddMessage("✓ Model architecture created")
            
            arcpy.AddMessage("\n[3/3] Testing inference...")
            model.eval()
            test_input = torch.randn(1, 3, image_size, image_size)
            with torch.no_grad():
                output = model(test_input)
            
            arcpy.AddMessage(f"✓ Inference test successful")
            arcpy.AddMessage(f"  Input shape: {test_input.shape}")
            arcpy.AddMessage(f"  Output shape: {output.shape}")
            arcpy.AddMessage("\n" + "=" * 80)
            arcpy.AddMessage("✓ Model validation passed!")
            arcpy.AddMessage("=" * 80)
            
        except Exception as e:
            arcpy.AddError("\n" + "=" * 80)
            arcpy.AddError("✗ Model validation failed!")
            arcpy.AddError("=" * 80)
            arcpy.AddError(f"\nError: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            raise