# Variational Autoencoder (VAE) for CIFAR-10

## Overview
This project implements a Variational Autoencoder (VAE) for generating realistic color images using the CIFAR-10 dataset. It includes latent space exploration and β-VAE experiments.

**🎨 NEW: Interactive Streamlit App Available!** - Learn VAE concepts through an interactive web interface.

## Assignment Details
- **Course:** Foundational Models and Generative AI
- **Assignment:** Assignment 1
- **Student ID:** M25AI2134

## Implementation Features
1. **Encoder Network** - Compresses 32×32×3 images into latent distributions (μ, σ)
2. **Decoder Network** - Reconstructs images from latent codes
3. **Reparameterization Trick** - Enables backpropagation through sampling
4. **Training Pipeline** - Complete training loop with loss visualization
5. **Image Generation** - Generate new images from random noise
6. **Latent Space Interpolation** - Smooth morphing between two images
7. **β-VAE Experiment** - Comparison between β=1 and β=5

## Dataset
- **CIFAR-10**: 60,000 32×32 color images in 10 classes
- **Classes**: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
- **Training samples**: 50,000
- **Test samples**: 10,000

## Model Architecture
- **Latent Dimension**: 128
- **Encoder**: Convolutional layers with progressive downsampling (Conv2D with stride=2)
- **Decoder**: Transposed convolutional layers for upsampling
- **Output**: Tanh activation for normalized [-1, 1] range

## Hyperparameters
- Batch Size: 128
- Learning Rate: 1e-3
- Epochs: 50
- Image Size: 32×32×3

## Requirements
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm

## Files
- `M25AI2134_assign_1.ipynb` - Main Jupyter notebook with complete implementation
- `M25AI2134_assign_1.html` - HTML export of the notebook
- `streamlit_app.py` - Interactive Streamlit web application
- `vae_model.py` - VAE model architecture (Encoder, Decoder, VAE)
- `utils.py` - Utility functions for training and visualization
- `requirements.txt` - Python dependencies

## Usage

### Option 1: Jupyter Notebook
Open and run the Jupyter notebook:
```bash
jupyter notebook M25AI2134_assign_1.ipynb
```

The notebook will:
1. Download and prepare the CIFAR-10 dataset
2. Build and train the VAE model
3. Generate sample images
4. Perform latent space interpolation
5. Compare different β-VAE configurations

### Option 2: Interactive Streamlit App (Recommended)

#### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

#### Running the App
```bash
streamlit run streamlit_app.py
```

#### Features of the Streamlit App
The interactive app provides:

1. **🏠 Home Page**
   - Introduction to VAE concepts
   - Mathematical formulations
   - Quick tutorial

2. **🏗️ Architecture Page**
   - Visual explanation of encoder-decoder structure
   - Detailed architecture diagrams
   - Reparameterization trick explanation

3. **📊 Dataset Page**
   - Load and explore CIFAR-10 dataset
   - View sample images from all 10 classes

4. **🎓 Training Page**
   - Interactive training interface
   - Real-time loss visualization
   - Train both β=1 and β=5 models
   - Adjustable hyperparameters

5. **🎨 Generate Images**
   - Generate new images from random noise
   - Compare outputs from different β values
   - View reconstruction quality

6. **🌈 Interpolation Page**
   - Explore latent space interpolation
   - Adjustable number of steps
   - Multiple interpolation rows
   - Smooth morphing between images

7. **⚖️ β-VAE Comparison**
   - Side-by-side comparison of β=1 vs β=5
   - Loss curves comparison
   - Generation quality comparison
   - Reconstruction quality analysis
   - Educational insights

#### Why Use the Streamlit App?
- **Interactive Learning**: Adjust parameters and see results instantly
- **Visual Understanding**: Better visualization of VAE concepts
- **Step-by-Step**: Learn VAE components one at a time
- **Experimentation**: Try different settings without writing code
- **Comparison Tools**: Easy side-by-side model comparison

📖 **For detailed instructions, see [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md)**

## Device Support
The implementation automatically detects and uses GPU if available, otherwise falls back to CPU.

## Project Structure
```
GEN_AI_A1/
├── M25AI2134_assign_1.ipynb    # Original Jupyter notebook
├── M25AI2134_assign_1.html     # HTML export
├── streamlit_app.py             # Main Streamlit application
├── vae_model.py                 # VAE architecture
├── utils.py                     # Utility functions
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── STREAMLIT_GUIDE.md          # Detailed Streamlit guide
└── .gitignore                  # Git ignore file
```
