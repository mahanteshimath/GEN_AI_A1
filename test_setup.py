"""
Test script to verify the VAE Streamlit app setup
Run this to check if all dependencies are installed correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    
    print("=" * 60)
    print("Testing VAE Streamlit App Setup")
    print("=" * 60)
    print()
    
    # Test basic imports
    print("Testing basic imports...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
        
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        from PIL import Image
        print(f"✅ Pillow (PIL)")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    print()
    
    # Test custom modules
    print("Testing custom modules...")
    try:
        from vae_model import VAE, Encoder, Decoder
        print("✅ vae_model.py imported successfully")
        
        from utils import load_cifar10_data, vae_loss, generate_image_grid
        print("✅ utils.py imported successfully")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    
    print()
    
    # Test PyTorch CUDA availability
    print("Checking device availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  GPU not available, using CPU (training will be slower)")
    
    print()
    
    # Test VAE model creation
    print("Testing VAE model creation...")
    try:
        model = VAE(latent_dim=128).to(device)
        print("✅ VAE model created successfully")
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32).to(device)
        x_recon, mu, logvar = model(x)
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {x_recon.shape}")
        print(f"   Latent dim: {mu.shape[1]}")
        
    except Exception as e:
        print(f"❌ Model Error: {e}")
        return False
    
    print()
    
    return True


def check_files():
    """Check if all required files exist"""
    
    print("Checking required files...")
    
    import os
    
    required_files = [
        'streamlit_app.py',
        'vae_model.py',
        'utils.py',
        'requirements.txt',
        'README.md',
        'STREAMLIT_GUIDE.md'
    ]
    
    all_exist = True
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} not found")
            all_exist = False
    
    print()
    return all_exist


def main():
    """Run all tests"""
    
    # Check files
    files_ok = check_files()
    
    # Test imports and model
    imports_ok = test_imports()
    
    # Final result
    print("=" * 60)
    if files_ok and imports_ok:
        print("✅ All tests passed!")
        print()
        print("You're ready to run the Streamlit app:")
        print("  streamlit run streamlit_app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
