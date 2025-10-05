import arcpy
import sys
import os

class Toolbox(object):
    def __init__(self):
        self.label = "SegFormer Semantic Segmentation"
        self.alias = "segformer"
        self.tools = [
            SegFormerClassify,
            SegFormerBatchProcess,
            SegFormerModelInfo,
            SegFormerValidateModel
        ]

class SegFormerClassify(object):
    """Main classification tool for single raster processing"""
    
    def __init__(self):
        self.label = "Classify Raster with SegFormer"
        self.description = "Perform semantic segmentation on a raster using a trained SegFormer model"
        self.canRunInBackground = False  # Run in foreground to avoid new ArcGIS instance

    def getParameterInfo(self):
        params = []
        
        # Input Raster
        params.append(arcpy.Parameter(
            displayName="Input Raster",
            name="input_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"))
        
        # Model File
        params.append(arcpy.Parameter(
            displayName="Trained Model File (.pth)",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"))
        params[1].filter.list = ['pth', 'pt']
        
        # Output Raster
        params.append(arcpy.Parameter(
            displayName="Output Classified Raster",
            name="output_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"))
        
        # Processing Options Group
        params.append(arcpy.Parameter(
            displayName="Tile Size (pixels)",
            name="tile_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"))
        params[3].value = 512
        
        params.append(arcpy.Parameter(
            displayName="Tile Overlap (pixels)",
            name="tile_overlap",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"))
        params[4].value = 64
        
        params.append(arcpy.Parameter(
            displayName="Batch Size",
            name="batch_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"))
        params[5].value = 4
        params[5].filter.type = "Range"
        params[5].filter.list = [1, 32]
        
        # Use GPU
        params.append(arcpy.Parameter(
            displayName="Use GPU (if available) - May cause crashes!",
            name="use_gpu",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"))
        params[6].value = False  # Default to CPU for stability
        
        # Class Configuration
        params.append(arcpy.Parameter(
            displayName="Class Names (comma-separated)",
            name="class_names",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"))
        params[7].value = "Background,Class1"
        
        # Model Weights
        params.append(arcpy.Parameter(
            displayName="Pretrained Weights",
            name="pretrained_weights",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"))
        params[8].value = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        # NoData Value
        params.append(arcpy.Parameter(
            displayName="NoData Value",
            name="nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"))
        params[9].value = 255
        
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        if parameters[3].value and parameters[3].value < 128:
            parameters[3].setWarningMessage("Tile size < 128 may produce poor results")
        
        if parameters[4].value and parameters[3].value:
            if parameters[4].value >= parameters[3].value / 2:
                parameters[4].setErrorMessage("Overlap must be less than half the tile size")
        
        return

    def execute(self, parameters, messages):
        model = None
        try:
            # Import here to avoid issues during toolbox load
            import torch
            import gc
            
            arcpy.AddMessage("Importing SegFormer module...")
            try:
                from SegFormer import SegFormer
            except ImportError as e:
                arcpy.AddError(f"Cannot import SegFormer module: {e}")
                return
            
            # Get parameters
            input_raster = parameters[0].valueAsText
            model_file = parameters[1].valueAsText
            output_raster = parameters[2].valueAsText
            tile_size = parameters[3].value or 512
            tile_overlap = parameters[4].value or 64
            batch_size = parameters[5].value or 4
            use_gpu = parameters[6].value
            class_names = [c.strip() for c in parameters[7].valueAsText.split(',')]
            pretrained_weights = parameters[8].valueAsText
            nodata_value = parameters[9].value or 255
            
            arcpy.AddMessage("=" * 80)
            arcpy.AddMessage("SegFormer Semantic Segmentation Tool")
            arcpy.AddMessage("=" * 80)
            
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
            arcpy.AddMessage("\n[2/5] Loading SegFormer model...")
            arcpy.AddMessage(f"Model file: {model_file}")
            model = self._load_model(model_file, class_names, pretrained_weights, device, messages)
            
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
                input_raster, output_raster, model, device,
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
        
        # Force CPU mode for ArcGIS Pro stability
        # GPU can cause crashes in ArcGIS Pro's Python environment
        if use_gpu and torch.cuda.is_available():
            try:
                # Test if GPU actually works
                test_tensor = torch.randn(10, 10).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                
                device = torch.device('cuda')
                arcpy.AddMessage(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
                arcpy.AddMessage(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                arcpy.AddWarning("  Note: GPU mode may be unstable in ArcGIS Pro")
                arcpy.AddWarning("  If you experience crashes, disable 'Use GPU' option")
                
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
                arcpy.AddMessage("✓ Using CPU mode (recommended for stability)")
        
        return device

    def _load_model(self, model_file, class_names, pretrained_weights, device, messages):
        """Load the SegFormer model"""
        import torch
        import numpy as np
        import gc
        
        try:
            arcpy.AddMessage("  Loading SegFormer module...")
            from SegFormer import SegFormer
            
            # Create dummy data object that matches what SegFormer.get_model expects
            class DummyDataset:
                def __init__(self, img_size):
                    # Create a simple numpy array to avoid early tensor creation
                    self.dummy_img = np.zeros((3, img_size, img_size), dtype=np.float32)
                    
                def __getitem__(self, idx):
                    return (torch.from_numpy(self.dummy_img), None)
            
            class DummyData:
                def __init__(self, classes, img_size=512):
                    self.classes = classes
                    self.train_ds = DummyDataset(img_size)
            
            arcpy.AddMessage("  Creating model architecture...")
            dummy_data = DummyData(class_names, img_size=512)
            
            # Don't pass state_dict in constructor to avoid loading twice
            model_wrapper = SegFormer(weights=pretrained_weights, state_dict=None)
            
            # Get model architecture first
            arcpy.AddMessage("  Building model structure...")
            model = model_wrapper.get_model(dummy_data, state_dict=model_file)
            
            # Clear dummy data
            del dummy_data
            gc.collect()
            
            arcpy.AddMessage("  Moving model to device...")
            model = model.to(device)
            model.eval()
            
            # Disable gradient computation to save memory
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
            
        except Exception as e:
            arcpy.AddError(f"Failed to load model: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            raise

    def _process_raster(self, input_path, output_path, model, device, 
                       tile_size, overlap, batch_size, nodata_value, messages):
        """Process raster with tiling"""
        import torch
        from osgeo import gdal, gdalconst
        import numpy as np
        import gc
        import tempfile
        import os
        
        # Check if output is geodatabase
        is_gdb = '.gdb' in output_path.lower() or '.sde' in output_path.lower()
        
        # If geodatabase, create temporary GeoTIFF first, then copy to GDB
        if is_gdb:
            temp_dir = tempfile.gettempdir()
            temp_output = os.path.join(temp_dir, "segformer_temp_output.tif")
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
        
        # Create output (temporary GeoTIFF or final output)
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
        
        # Process tiles one at a time to minimize memory usage
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
                
                # Process single tile
                tile_tensor = torch.from_numpy(tile_data).unsqueeze(0).float() / 255.0
                tile_tensor = tile_tensor.to(device)
                
                with torch.no_grad():
                    output = model(tile_tensor)
                    prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
                
                # Extract valid region
                pred = prediction[:y_size, :x_size]
                
                # Write result
                dst_band.WriteArray(pred, x_off, y_off)
                
                # Cleanup
                del tile_tensor, output, prediction, pred
                processed += 1
                
                if processed % 10 == 0:
                    arcpy.SetProgressorPosition(processed)
                    arcpy.AddMessage(f"  Progress: {processed}/{total_tiles} tiles ({100*processed/total_tiles:.1f}%)")
                    # Force memory cleanup every 10 tiles
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                arcpy.AddWarning(f"  Failed to process tile {tile_idx}: {e}")
                # Write nodata for failed tile
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
        
        # If geodatabase output, copy temp file to GDB
        if is_gdb:
            arcpy.AddMessage(f"  Copying to geodatabase...")
            try:
                # Use arcpy to copy to geodatabase
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



    def _blend_overlap(self, tile, overlap):
        """Simple center-weighted blending for overlapping tiles"""
        # For simplicity, just return center region
        # Full implementation would blend with neighboring tiles
        return tile

    def _finalize_output(self, output_path, class_names, messages):
        """Build pyramids and statistics"""
        try:
            # Check if output is in a geodatabase
            is_gdb = '.gdb' in output_path.lower() or '.sde' in output_path.lower()
            
            if is_gdb:
                arcpy.AddMessage("✓ Output saved to geodatabase")
                # Geodatabases handle pyramids automatically
                try:
                    arcpy.management.CalculateStatistics(output_path)
                    arcpy.AddMessage("✓ Calculated statistics")
                except:
                    arcpy.AddMessage("  (Statistics will be calculated on first display)")
            else:
                # For file-based rasters (GeoTIFF, etc.)
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
            
            # Try to build attribute table with class names
            try:
                arcpy.management.BuildRasterAttributeTable(output_path, "Overwrite")
                arcpy.AddMessage("✓ Built raster attribute table")
            except Exception as e:
                arcpy.AddMessage(f"  Could not build attribute table: {e}")
                
        except Exception as e:
            arcpy.AddWarning(f"Post-processing warning: {e}")
            arcpy.AddMessage("  Classification completed successfully despite post-processing issues")


class SegFormerBatchProcess(object):
    """Batch processing tool for multiple rasters"""
    
    def __init__(self):
        self.label = "Batch Classify Rasters"
        self.description = "Process multiple rasters with SegFormer"
        self.canRunInBackground = False  # Run in foreground to see progress

    def getParameterInfo(self):
        params = []
        
        # Input Folder
        params.append(arcpy.Parameter(
            displayName="Input Raster Folder",
            name="input_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"))
        
        # File Pattern
        params.append(arcpy.Parameter(
            displayName="File Pattern (e.g., *.tif)",
            name="file_pattern",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"))
        params[1].value = "*.tif"
        
        # Model File
        params.append(arcpy.Parameter(
            displayName="Trained Model File (.pth)",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"))
        params[2].filter.list = ['pth', 'pt']
        
        # Output Folder
        params.append(arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"))
        
        # Output Suffix
        params.append(arcpy.Parameter(
            displayName="Output Suffix",
            name="output_suffix",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"))
        params[4].value = "_classified"
        
        # Tile Size
        params.append(arcpy.Parameter(
            displayName="Tile Size",
            name="tile_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"))
        params[5].value = 512
        
        # Class Names
        params.append(arcpy.Parameter(
            displayName="Class Names (comma-separated)",
            name="class_names",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"))
        params[6].value = "Background,Class1"
        
        return params

    def execute(self, parameters, messages):
        try:
            import glob
            
            input_folder = parameters[0].valueAsText
            file_pattern = parameters[1].valueAsText
            model_file = parameters[2].valueAsText
            output_folder = parameters[3].valueAsText
            output_suffix = parameters[4].valueAsText or "_classified"
            tile_size = parameters[5].value or 512
            class_names = parameters[6].valueAsText
            
            # Find input files
            search_path = os.path.join(input_folder, file_pattern)
            input_files = glob.glob(search_path)
            
            if not input_files:
                arcpy.AddError(f"No files found matching: {search_path}")
                return
            
            arcpy.AddMessage(f"Found {len(input_files)} raster(s) to process")
            arcpy.SetProgressor("step", "Processing rasters...", 0, len(input_files), 1)
            
            # Process each file using SegFormerClassify
            classify_tool = SegFormerClassify()
            
            for i, input_file in enumerate(input_files, 1):
                arcpy.AddMessage(f"\n{'='*80}")
                arcpy.AddMessage(f"Processing {i}/{len(input_files)}: {os.path.basename(input_file)}")
                arcpy.AddMessage(f"{'='*80}")
                
                # Generate output path
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(output_folder, f"{base_name}{output_suffix}.tif")
                
                # Create parameter objects for classify tool
                class Param:
                    def __init__(self, value):
                        self.value = value
                        self.valueAsText = str(value) if value is not None else None
                
                params = [
                    Param(input_file),
                    Param(model_file),
                    Param(output_file),
                    Param(tile_size),
                    Param(64),  # overlap
                    Param(4),   # batch_size
                    Param(True),  # use_gpu
                    Param(class_names),
                    Param("nvidia/segformer-b0-finetuned-ade-512-512"),
                    Param(255)  # nodata
                ]
                
                try:
                    classify_tool.execute(params, messages)
                    arcpy.AddMessage(f"✓ Completed: {output_file}")
                except Exception as e:
                    arcpy.AddWarning(f"✗ Failed: {input_file} - {str(e)}")
                
                arcpy.SetProgressorPosition(i)
            
            arcpy.ResetProgressor()
            arcpy.AddMessage(f"\n{'='*80}")
            arcpy.AddMessage(f"✓ Batch processing complete! Processed {len(input_files)} raster(s)")
            arcpy.AddMessage(f"{'='*80}")
            
        except Exception as e:
            arcpy.AddError(f"Batch processing error: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())


class SegFormerModelInfo(object):
    """Tool to inspect model information"""
    
    def __init__(self):
        self.label = "Inspect Model Information"
        self.description = "Display information about a trained SegFormer model"
        self.canRunInBackground = False

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
            arcpy.AddMessage("SegFormer Model Information")
            arcpy.AddMessage("=" * 80)
            arcpy.AddMessage(f"\nModel File: {model_file}")
            arcpy.AddMessage(f"File Size: {os.path.getsize(model_file) / 1e6:.2f} MB")
            
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')
            
            arcpy.AddMessage("\nCheckpoint Contents:")
            if isinstance(checkpoint, dict):
                for key in checkpoint.keys():
                    if key != 'state_dict':
                        arcpy.AddMessage(f"  - {key}")
                
                # Get state dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Count parameters
                total_params = sum(p.numel() for p in state_dict.values())
                arcpy.AddMessage(f"\nTotal Parameters: {total_params:,}")
                
                # Detect number of classes
                for key in state_dict.keys():
                    if 'classifier' in key and 'weight' in key:
                        num_classes = state_dict[key].shape[0]
                        arcpy.AddMessage(f"Number of Classes: {num_classes}")
                        break
                
                # Show layer info
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


class SegFormerValidateModel(object):
    """Tool to validate model can be loaded"""
    
    def __init__(self):
        self.label = "Validate Model"
        self.description = "Test if a model can be loaded successfully"
        self.canRunInBackground = False

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
        params[1].value = 2
        
        params.append(arcpy.Parameter(
            displayName="Pretrained Weights",
            name="pretrained_weights",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"))
        params[2].value = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        return params

    def execute(self, parameters, messages):
        try:
            import torch
            from SegFormer import SegFormer
            
            model_file = parameters[0].valueAsText
            num_classes = parameters[1].value
            pretrained_weights = parameters[2].valueAsText
            
            arcpy.AddMessage("=" * 80)
            arcpy.AddMessage("Model Validation")
            arcpy.AddMessage("=" * 80)
            
            arcpy.AddMessage("\n[1/3] Loading model checkpoint...")
            checkpoint = torch.load(model_file, map_location='cpu')
            arcpy.AddMessage("✓ Checkpoint loaded")
            
            arcpy.AddMessage("\n[2/3] Initializing model architecture...")
            
            # Create proper dummy dataset
            class DummyDataset:
                def __init__(self, img_size=512):
                    self.dummy_img = torch.zeros(3, img_size, img_size)
                    
                def __getitem__(self, idx):
                    return (self.dummy_img, None)
            
            class DummyData:
                def __init__(self, n_classes):
                    self.classes = [f"Class{i}" for i in range(n_classes)]
                    self.train_ds = DummyDataset()
            
            dummy_data = DummyData(num_classes)
            model_wrapper = SegFormer(weights=pretrained_weights, state_dict=model_file)
            model = model_wrapper.get_model(dummy_data)
            arcpy.AddMessage("✓ Model architecture created")
            
            arcpy.AddMessage("\n[3/3] Testing inference...")
            model.eval()
            test_input = torch.randn(1, 3, 256, 256)
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