"""
SegFormer Toolbox Installation & Validation Script
Run this from ArcGIS Pro's Python Command Prompt to verify installation
"""
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_step(step, total, text):
    """Print step information"""
    print(f"\n[{step}/{total}] {text}")

def check_mark(success):
    """Return check or X mark"""
    return "✓" if success else "✗"

def main():
    print_header("SegFormer Toolbox Installation Validator")
    
    total_steps = 8
    results = {}
    
    # Step 1: Check Python version
    print_step(1, total_steps, "Checking Python version...")
    python_version = sys.version_info
    py_ok = python_version.major == 3 and python_version.minor >= 7
    results['python'] = py_ok
    print(f"  {check_mark(py_ok)} Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    if not py_ok:
        print("  ⚠ Warning: Python 3.7+ required")
    
    # Step 2: Check ArcPy
    print_step(2, total_steps, "Checking ArcPy...")
    try:
        import arcpy # type: ignore
        arcpy_version = arcpy.GetInstallInfo()['Version']
        results['arcpy'] = True
        print(f"  ✓ ArcGIS Pro {arcpy_version}")
    except ImportError:
        results['arcpy'] = False
        print("  ✗ ArcPy not found - Are you running from ArcGIS Pro Python?")
    
    # Step 3: Check PyTorch
    print_step(3, total_steps, "Checking PyTorch...")
    try:
        import torch
        torch_version = torch.__version__
        results['torch'] = True
        print(f"  ✓ PyTorch {torch_version}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU Available: {gpu_name}")
            print(f"    VRAM: {gpu_memory:.2f} GB")
            results['gpu'] = True
        else:
            print("  ⚠ GPU not available - will use CPU (slower)")
            results['gpu'] = False
    except ImportError:
        results['torch'] = False
        print("  ✗ PyTorch not installed")
        print("    Install with: conda install pytorch torchvision cpuonly -c pytorch")
    
    # Step 4: Check Transformers
    print_step(4, total_steps, "Checking Transformers library...")
    try:
        import transformers
        trans_version = transformers.__version__
        results['transformers'] = True
        print(f"  ✓ Transformers {trans_version}")
    except ImportError:
        results['transformers'] = False
        print("  ✗ Transformers not installed")
        print("    Install with: pip install transformers")
    
    # Step 5: Check GDAL
    print_step(5, total_steps, "Checking GDAL...")
    try:
        from osgeo import gdal # type: ignore
        gdal_version = gdal.__version__
        results['gdal'] = True
        print(f"  ✓ GDAL {gdal_version}")
    except ImportError:
        results['gdal'] = False
        print("  ✗ GDAL not found")
        print("    Install with: conda install gdal")
    
    # Step 6: Check NumPy
    print_step(6, total_steps, "Checking NumPy...")
    try:
        import numpy as np
        numpy_version = np.__version__
        results['numpy'] = True
        print(f"  ✓ NumPy {numpy_version}")
    except ImportError:
        results['numpy'] = False
        print("  ✗ NumPy not found")
    
    # Step 7: Check SegFormer module
    print_step(7, total_steps, "Checking SegFormer module...")
    segformer_paths = [
        r"C:\Users\gwrye\Documents\ArcGIS\Projects\MyProject1\SegFormer",
        os.path.join(os.getcwd(), "SegFormer"),
        os.path.join(Path.home(), "Documents", "ArcGIS", "Projects", "MyProject1", "SegFormer")
    ]
    
    segformer_found = False
    for path in segformer_paths:
        segformer_file = os.path.join(path, "SegFormer.py")
        if os.path.exists(segformer_file):
            print(f"  ✓ Found: {segformer_file}")
            segformer_found = True
            
            # Try to import
            sys.path.insert(0, path)
            try:
                from models import SegFormer
                print(f"  ✓ Successfully imported SegFormer class")
                results['segformer'] = True
                break
            except Exception as e:
                print(f"  ⚠ Found file but import failed: {e}")
                results['segformer'] = False
                break
    
    if not segformer_found:
        results['segformer'] = False
        print(f"  ✗ SegFormer.py not found in common locations")
        print(f"    Searched:")
        for path in segformer_paths:
            print(f"      - {path}")
        print(f"    Please update SEGFORMER_PATH in SegFormerToolbox.pyt")
    
    # Step 8: Test model initialization
    print_step(8, total_steps, "Testing model initialization...")
    if results.get('torch') and results.get('transformers') and results.get('segformer'):
        try:
            # Create dummy data with proper structure
            class DummyDataset:
                def __init__(self, img_size=512):
                    import torch
                    self.dummy_img = torch.zeros(3, img_size, img_size)
                    
                def __getitem__(self, idx):
                    return (self.dummy_img, None)
            
            class DummyData:
                def __init__(self):
                    self.classes = ['Background', 'Class1']
                    self.train_ds = DummyDataset()
            
            from models import SegFormer
            dummy_data = DummyData()
            model_wrapper = SegFormer()
            model = model_wrapper.get_model(dummy_data)
            
            # Test forward pass
            import torch
            test_input = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                output = model(test_input)
            
            print(f"  ✓ Model initialization successful")
            print(f"    Input shape: {test_input.shape}")
            print(f"    Output shape: {output.shape}")
            results['model_test'] = True
        except Exception as e:
            print(f"  ✗ Model initialization failed: {e}")
            results['model_test'] = False
    else:
        print(f"  ⊗ Skipped - dependencies not met")
        results['model_test'] = False
    
    # Summary
    print_header("Installation Summary")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTests Passed: {passed}/{total}")
    print(f"\nComponent Status:")
    print(f"  Python 3.7+:        {check_mark(results.get('python', False))}")
    print(f"  ArcPy:              {check_mark(results.get('arcpy', False))}")
    print(f"  PyTorch:            {check_mark(results.get('torch', False))}")
    print(f"  Transformers:       {check_mark(results.get('transformers', False))}")
    print(f"  GDAL:               {check_mark(results.get('gdal', False))}")
    print(f"  NumPy:              {check_mark(results.get('numpy', False))}")
    print(f"  SegFormer Module:   {check_mark(results.get('segformer', False))}")
    print(f"  Model Test:         {check_mark(results.get('model_test', False))}")
    if results.get('gpu'):
        print(f"  GPU Acceleration:   ✓ Available")
    else:
        print(f"  GPU Acceleration:   ⚠ Not available (CPU mode)")
    
    # Recommendations
    print("\nRecommendations:")
    
    if not results.get('torch'):
        print("  • Install PyTorch:")
        print("    conda install pytorch torchvision cpuonly -c pytorch")
        print("    Or for GPU: conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
    
    if not results.get('transformers'):
        print("  • Install Transformers:")
        print("    pip install transformers")
    
    if not results.get('gdal'):
        print("  • Install GDAL:")
        print("    conda install gdal")
    
    if not results.get('segformer'):
        print("  • Ensure SegFormer.py is in the correct location")
        print("  • Update SEGFORMER_PATH in SegFormerToolbox.pyt")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to use the SegFormer Toolbox!")
    elif passed >= total - 1:
        print("\n⚠ Almost ready! Fix the issues above and re-run this script.")
    else:
        print("\n❌ Several components missing. Please install required packages.")
    
    print("\n" + "="*80)
    
    # Installation commands summary
    if not all([results.get('torch'), results.get('transformers'), results.get('gdal')]):
        print_header("Quick Installation Commands")
        print("\nRun these commands in ArcGIS Pro Python Command Prompt:\n")
        
        if not results.get('torch'):
            print("# Install PyTorch (CPU)")
            print("conda install pytorch torchvision cpuonly -c pytorch\n")
            print("# OR for GPU support:")
            print("conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia\n")
        
        if not results.get('transformers'):
            print("# Install Transformers")
            print("pip install transformers\n")
        
        if not results.get('gdal'):
            print("# Install GDAL")
            print("conda install gdal\n")
        
        print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nValidation cancelled by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")