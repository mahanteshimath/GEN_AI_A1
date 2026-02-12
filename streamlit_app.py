import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os
from vae_model import VAE, Encoder, Decoder
from utils import (
    vae_loss, generate_image_grid, show_reconstructions, 
    latent_space_interpolation, plot_training_curves,
    load_cifar10_data, train_vae_epoch
)

# Page config
st.set_page_config(
    page_title="Interactive VAE Explorer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        font-weight: bold;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'model_beta1' not in st.session_state:
    st.session_state.model_beta1 = None

if 'model_beta5' not in st.session_state:
    st.session_state.model_beta5 = None

if 'train_loader' not in st.session_state:
    st.session_state.train_loader = None
    st.session_state.test_loader = None

if 'history_beta1' not in st.session_state:
    st.session_state.history_beta1 = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

if 'history_beta5' not in st.session_state:
    st.session_state.history_beta5 = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

# Sidebar navigation
st.sidebar.title("🎨 VAE Explorer")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🏗️ Architecture", "📊 Dataset", "🎓 Training", 
     "🎨 Generate Images", "🌈 Interpolation", "⚖️ β-VAE Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
LATENT_DIM = st.sidebar.slider("Latent Dimension", 32, 256, 128, 32)
BATCH_SIZE = st.sidebar.slider("Batch Size", 32, 256, 128, 32)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Device:** {st.session_state.device}")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown("<h1 class='main-header'>🎨 Interactive VAE Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Learn Variational Autoencoders Through Interactive Experimentation</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧠 What is VAE?")
        st.markdown("""
        <div class='info-box'>
        A <b>Variational Autoencoder (VAE)</b> is a generative model that learns to:
        <ul>
            <li>Compress images into a latent space (Encoder)</li>
            <li>Reconstruct images from latent codes (Decoder)</li>
            <li>Generate new images from random noise</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Key Components")
        st.markdown("""
        <div class='info-box'>
        <ul>
            <li><b>Encoder:</b> Image → (μ, σ)</li>
            <li><b>Reparameterization:</b> z = μ + σ × ε</li>
            <li><b>Decoder:</b> z → Reconstructed Image</li>
            <li><b>Loss:</b> Reconstruction + KL Divergence</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 📚 What You'll Learn")
        st.markdown("""
        <div class='info-box'>
        <ul>
            <li>How VAE architecture works</li>
            <li>Training process and loss function</li>
            <li>Generate new images</li>
            <li>Explore latent space</li>
            <li>β-VAE experiments</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔑 The VAE Formula")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Reparameterization Trick")
        st.latex(r"z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)")
        st.info("This trick allows gradients to flow through the sampling operation!")
    
    with col2:
        st.markdown("#### Loss Function (ELBO)")
        st.latex(r"\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL}")
        st.info("Balance between reconstruction quality and latent regularization")
    
    st.markdown("---")
    
    st.markdown("### 🚀 Getting Started")
    st.success("👈 Use the sidebar to navigate through different sections and start exploring!")
    
    with st.expander("📖 Quick Tutorial"):
        st.markdown("""
        1. **Dataset**: Load and explore CIFAR-10 images
        2. **Architecture**: Understand the encoder-decoder structure
        3. **Training**: Train your own VAE model
        4. **Generate**: Create new images from random noise
        5. **Interpolation**: Explore smooth transitions in latent space
        6. **β-VAE**: Experiment with different β values
        """)

# ============================================================================
# ARCHITECTURE PAGE
# ============================================================================
elif page == "🏗️ Architecture":
    st.markdown("<h1 class='main-header'>🏗️ VAE Architecture</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📐 Overview", "🔽 Encoder", "🔼 Decoder"])
    
    with tab1:
        st.markdown("### Complete VAE Pipeline")
        
        st.markdown("""
        ```
        Input Image (32×32×3)
              ↓
         ┌─────────┐
         │ ENCODER │  → Compresses to latent space
         └─────────┘
              ↓
         (μ, log σ²)  → Mean and log-variance
              ↓
        Reparameterization: z = μ + σ × ε
              ↓
         ┌─────────┐
         │ DECODER │  → Reconstructs image
         └─────────┘
              ↓
        Output Image (32×32×3)
        ```
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Encoder Path")
            st.code("""
32×32×3 (Input)
  ↓ Conv(3→32, stride=2)
16×16×32
  ↓ Conv(32→64, stride=2)
8×8×64
  ↓ Conv(64→128, stride=2)
4×4×128
  ↓ Conv(128→256, stride=2)
2×2×256 = 1024
  ↓ Linear
μ (128) & log σ² (128)
            """, language="text")
        
        with col2:
            st.markdown("#### 📊 Decoder Path")
            st.code("""
z (128)
  ↓ Linear
1024 → 2×2×256
  ↓ ConvT(256→128, stride=2)
4×4×128
  ↓ ConvT(128→64, stride=2)
8×8×64
  ↓ ConvT(64→32, stride=2)
16×16×32
  ↓ ConvT(32→3, stride=2)
32×32×3 (Output)
            """, language="text")
    
    with tab2:
        st.markdown("### 🔽 Encoder Network")
        st.markdown("The encoder compresses a 32×32×3 image into two vectors:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**μ (mu)**: Mean of latent distribution")
        with col2:
            st.info("**log σ²**: Log-variance (for numerical stability)")
        
        st.code("""
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
        """, language="python")
    
    with tab3:
        st.markdown("### 🔼 Decoder Network")
        st.markdown("The decoder reconstructs images from latent vectors using transposed convolutions:")
        
        st.code("""
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 2 * 2),
            nn.ReLU()
        )
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        """, language="python")
    
    st.markdown("---")
    st.markdown("### 🎲 The Reparameterization Trick")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### ❌ The Problem")
        st.warning("Direct sampling z ~ N(μ, σ²) is **not differentiable** - gradients can't flow back!")
        
    with col2:
        st.markdown("#### ✅ The Solution")
        st.success("Sample ε ~ N(0, I), then compute z = μ + σ × ε")
    
    st.latex(r"z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)")
    
    st.code("""
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)  # Convert log(σ²) to σ
    epsilon = torch.randn_like(std)  # Sample from N(0,1)
    z = mu + std * epsilon  # Reparameterization
    return z
    """, language="python")

# ============================================================================
# DATASET PAGE
# ============================================================================
elif page == "📊 Dataset":
    st.markdown("<h1 class='main-header'>📊 CIFAR-10 Dataset</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>CIFAR-10</b> contains 60,000 32×32 color images in 10 classes:
    <code>plane, car, bird, cat, deer, dog, frog, horse, ship, truck</code>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📥 Load CIFAR-10 Dataset"):
        with st.spinner("Downloading and preparing CIFAR-10..."):
            train_loader, test_loader = load_cifar10_data(BATCH_SIZE)
            st.session_state.train_loader = train_loader
            st.session_state.test_loader = test_loader
        st.success("✅ Dataset loaded successfully!")
    
    if st.session_state.train_loader is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", "50,000")
        with col2:
            st.metric("Test Samples", "10,000")
        with col3:
            st.metric("Image Size", "32×32×3")
        
        st.markdown("### 🖼️ Sample Images")
        
        # Show sample images
        images, labels = next(iter(st.session_state.test_loader))
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for idx, ax in enumerate(axes.flat):
            if idx < 16:
                img = images[idx].numpy().transpose(1, 2, 0)
                img = img * 0.5 + 0.5  # Denormalize
                ax.imshow(np.clip(img, 0, 1))
                ax.set_title(classes[labels[idx]], fontsize=8)
            ax.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================================================
# TRAINING PAGE
# ============================================================================
elif page == "🎓 Training":
    st.markdown("<h1 class='main-header'>🎓 Train Your VAE</h1>", unsafe_allow_html=True)
    
    if st.session_state.train_loader is None:
        st.warning("⚠️ Please load the dataset first from the '📊 Dataset' page!")
    else:
        tab1, tab2 = st.tabs(["⚙️ Training Settings", "📈 Training Progress"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Standard VAE (β=1)")
                epochs_beta1 = st.number_input("Number of Epochs (β=1)", 1, 100, 10, key="epochs_beta1")
                lr_beta1 = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="lr1")
                
                if st.button("🚀 Train VAE (β=1)", key="train1"):
                    st.session_state.model_beta1 = VAE(LATENT_DIM).to(st.session_state.device)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    loss_chart = st.empty()
                    
                    for epoch in range(epochs_beta1):
                        avg_loss, avg_recon, avg_kl = train_vae_epoch(
                            st.session_state.model_beta1,
                            st.session_state.train_loader,
                            st.session_state.device,
                            beta=1.0,
                            lr=lr_beta1
                        )
                        
                        st.session_state.history_beta1['total_loss'].append(avg_loss)
                        st.session_state.history_beta1['recon_loss'].append(avg_recon)
                        st.session_state.history_beta1['kl_loss'].append(avg_kl)
                        
                        progress_bar.progress((epoch + 1) / epochs_beta1)
                        status_text.text(f"Epoch {epoch+1}/{epochs_beta1} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")
                        
                        # Update chart
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(st.session_state.history_beta1['total_loss'], 'b-', label='Total')
                        ax.plot(st.session_state.history_beta1['recon_loss'], 'g-', label='Recon')
                        ax.plot(st.session_state.history_beta1['kl_loss'], 'r-', label='KL')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        loss_chart.pyplot(fig)
                        plt.close()
                    
                    st.success("✅ Training complete!")
            
            with col2:
                st.markdown("### β-VAE (β=5)")
                epochs_beta5 = st.number_input("Number of Epochs (β=5)", 1, 100, 10, key="epochs_beta5")
                lr_beta5 = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="lr5")
                
                if st.button("🚀 Train VAE (β=5)", key="train5"):
                    st.session_state.model_beta5 = VAE(LATENT_DIM).to(st.session_state.device)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    loss_chart = st.empty()
                    
                    for epoch in range(epochs_beta5):
                        avg_loss, avg_recon, avg_kl = train_vae_epoch(
                            st.session_state.model_beta5,
                            st.session_state.train_loader,
                            st.session_state.device,
                            beta=5.0,
                            lr=lr_beta5
                        )
                        
                        st.session_state.history_beta5['total_loss'].append(avg_loss)
                        st.session_state.history_beta5['recon_loss'].append(avg_recon)
                        st.session_state.history_beta5['kl_loss'].append(avg_kl)
                        
                        progress_bar.progress((epoch + 1) / epochs_beta5)
                        status_text.text(f"Epoch {epoch+1}/{epochs_beta5} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")
                        
                        # Update chart
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(st.session_state.history_beta5['total_loss'], 'b-', label='Total')
                        ax.plot(st.session_state.history_beta5['recon_loss'], 'g-', label='Recon')
                        ax.plot(st.session_state.history_beta5['kl_loss'], 'r-', label='KL')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        loss_chart.pyplot(fig)
                        plt.close()
                    
                    st.success("✅ Training complete!")
        
        with tab2:
            st.markdown("### 📈 Training Curves")
            
            if len(st.session_state.history_beta1['total_loss']) > 0:
                fig = plot_training_curves(st.session_state.history_beta1, "Standard VAE (β=1)")
                st.pyplot(fig)
                plt.close()
            
            if len(st.session_state.history_beta5['total_loss']) > 0:
                fig = plot_training_curves(st.session_state.history_beta5, "β-VAE (β=5)")
                st.pyplot(fig)
                plt.close()

# ============================================================================
# GENERATE PAGE
# ============================================================================
elif page == "🎨 Generate Images":
    st.markdown("<h1 class='main-header'>🎨 Generate New Images</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Generate completely new images by sampling random latent vectors from a standard normal distribution N(0, I) 
    and passing them through the decoder.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Standard VAE (β=1)")
        if st.session_state.model_beta1 is None:
            st.warning("⚠️ Train the model first!")
        else:
            num_images1 = st.slider("Number of images", 4, 16, 16, 4, key="num1")
            seed1 = st.number_input("Random Seed", 0, 1000, 42, key="seed1")
            
            if st.button("🎲 Generate (β=1)"):
                torch.manual_seed(seed1)
                fig = generate_image_grid(st.session_state.model_beta1, st.session_state.device, 
                                         num_images1, 4, "Generated Images (β=1)")
                st.pyplot(fig)
                plt.close()
    
    with col2:
        st.markdown("### β-VAE (β=5)")
        if st.session_state.model_beta5 is None:
            st.warning("⚠️ Train the model first!")
        else:
            num_images5 = st.slider("Number of images", 4, 16, 16, 4, key="num5")
            seed5 = st.number_input("Random Seed", 0, 1000, 42, key="seed5")
            
            if st.button("🎲 Generate (β=5)"):
                torch.manual_seed(seed5)
                fig = generate_image_grid(st.session_state.model_beta5, st.session_state.device,
                                         num_images5, 4, "Generated Images (β=5)")
                st.pyplot(fig)
                plt.close()
    
    st.markdown("---")
    
    if st.session_state.model_beta1 is not None and st.session_state.test_loader is not None:
        st.markdown("### 🔄 Reconstruction Quality")
        if st.button("Show Reconstructions"):
            fig = show_reconstructions(st.session_state.model_beta1, st.session_state.test_loader, 
                                      st.session_state.device, num_images=8)
            st.pyplot(fig)
            plt.close()

# ============================================================================
# INTERPOLATION PAGE
# ============================================================================
elif page == "🌈 Interpolation":
    st.markdown("<h1 class='main-header'>🌈 Latent Space Interpolation</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Latent space interpolation</b> reveals the structure of the learned representation. 
    We smoothly transition between two random latent vectors using linear interpolation:
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"z_{\alpha} = (1-\alpha) \cdot z_1 + \alpha \cdot z_2, \quad \alpha \in [0, 1]")
    
    if st.session_state.model_beta1 is None:
        st.warning("⚠️ Train a model first!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            num_steps = st.slider("Number of interpolation steps", 5, 20, 10)
        
        with col2:
            num_rows = st.slider("Number of interpolation rows", 1, 5, 3)
        
        seed = st.number_input("Random Seed", 0, 1000, 42, key="interp_seed")
        
        if st.button("🌈 Generate Interpolation"):
            torch.manual_seed(seed)
            fig = latent_space_interpolation(st.session_state.model_beta1, st.session_state.device,
                                            num_steps, num_rows)
            st.pyplot(fig)
            plt.close()
            
            st.success("✅ Notice how the images smoothly transition from one to another!")
        
        with st.expander("💡 Understanding Interpolation"):
            st.markdown("""
            - **Smooth transitions** indicate a well-structured latent space
            - Each intermediate image should look realistic
            - VAE's prior regularization (KL divergence) encourages this smoothness
            - Compare with β-VAE to see how β affects the latent space organization
            """)

# ============================================================================
# BETA-VAE COMPARISON PAGE
# ============================================================================
elif page == "⚖️ β-VAE Comparison":
    st.markdown("<h1 class='main-header'>⚖️ β-VAE Comparison</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    The <b>β parameter</b> controls the trade-off between reconstruction quality and latent space regularization:
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"\mathcal{L}_{\beta\text{-VAE}} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**β = 1**: Standard VAE with balanced loss")
    
    with col2:
        st.info("**β > 1**: Stronger latent regularization → more disentangled features")
    
    st.markdown("---")
    
    if st.session_state.model_beta1 is None or st.session_state.model_beta5 is None:
        st.warning("⚠️ Train both models (β=1 and β=5) first!")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Loss Comparison", "🎨 Generation Comparison", "🔄 Reconstruction Comparison"])
        
        with tab1:
            st.markdown("### Training Loss Comparison")
            
            if len(st.session_state.history_beta1['total_loss']) > 0 and len(st.session_state.history_beta5['total_loss']) > 0:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                epochs = range(1, len(st.session_state.history_beta1['total_loss']) + 1)
                
                axes[0].plot(epochs, st.session_state.history_beta1['total_loss'], 'b-', label='β=1', linewidth=2)
                axes[0].plot(epochs, st.session_state.history_beta5['total_loss'], 'r-', label='β=5', linewidth=2)
                axes[0].set_title('Total Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(epochs, st.session_state.history_beta1['recon_loss'], 'b-', label='β=1', linewidth=2)
                axes[1].plot(epochs, st.session_state.history_beta5['recon_loss'], 'r-', label='β=5', linewidth=2)
                axes[1].set_title('Reconstruction Loss')
                axes[1].set_xlabel('Epoch')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                axes[2].plot(epochs, st.session_state.history_beta1['kl_loss'], 'b-', label='β=1', linewidth=2)
                axes[2].plot(epochs, st.session_state.history_beta5['kl_loss'], 'r-', label='β=5', linewidth=2)
                axes[2].set_title('KL Divergence')
                axes[2].set_xlabel('Epoch')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with tab2:
            st.markdown("### Generated Images Comparison")
            st.info("Using the **same random latent vectors** for fair comparison")
            
            seed = st.number_input("Random Seed", 0, 1000, 42, key="comp_seed")
            
            if st.button("🎲 Compare Generations"):
                torch.manual_seed(seed)
                z = torch.randn(8, LATENT_DIM).to(st.session_state.device)
                
                st.session_state.model_beta1.eval()
                st.session_state.model_beta5.eval()
                
                with torch.no_grad():
                    gen1 = st.session_state.model_beta1.decoder(z)
                    gen5 = st.session_state.model_beta5.decoder(z)
                
                gen1 = (gen1 * 0.5 + 0.5).cpu()
                gen5 = (gen5 * 0.5 + 0.5).cpu()
                
                fig, axes = plt.subplots(2, 8, figsize=(16, 4))
                
                for i in range(8):
                    img1 = gen1[i].numpy().transpose(1, 2, 0)
                    img5 = gen5[i].numpy().transpose(1, 2, 0)
                    
                    axes[0, i].imshow(np.clip(img1, 0, 1))
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_ylabel('β=1', fontsize=12, rotation=0, labelpad=30)
                    
                    axes[1, i].imshow(np.clip(img5, 0, 1))
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_ylabel('β=5', fontsize=12, rotation=0, labelpad=30)
                
                plt.suptitle('Generation Comparison (Same Latent Vectors)', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with tab3:
            st.markdown("### Reconstruction Quality Comparison")
            
            if st.session_state.test_loader is not None:
                if st.button("🔍 Compare Reconstructions"):
                    images, _ = next(iter(st.session_state.test_loader))
                    images = images[:6].to(st.session_state.device)
                    
                    st.session_state.model_beta1.eval()
                    st.session_state.model_beta5.eval()
                    
                    with torch.no_grad():
                        recon1, _, _ = st.session_state.model_beta1(images)
                        recon5, _, _ = st.session_state.model_beta5(images)
                    
                    images = (images * 0.5 + 0.5).cpu()
                    recon1 = (recon1 * 0.5 + 0.5).cpu()
                    recon5 = (recon5 * 0.5 + 0.5).cpu()
                    
                    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
                    
                    for i in range(6):
                        orig = images[i].numpy().transpose(1, 2, 0)
                        r1 = recon1[i].numpy().transpose(1, 2, 0)
                        r5 = recon5[i].numpy().transpose(1, 2, 0)
                        
                        axes[0, i].imshow(np.clip(orig, 0, 1))
                        axes[0, i].axis('off')
                        if i == 0:
                            axes[0, i].set_ylabel('Original', fontsize=10, rotation=0, labelpad=40)
                        
                        axes[1, i].imshow(np.clip(r1, 0, 1))
                        axes[1, i].axis('off')
                        if i == 0:
                            axes[1, i].set_ylabel('β=1', fontsize=10, rotation=0, labelpad=40)
                        
                        axes[2, i].imshow(np.clip(r5, 0, 1))
                        axes[2, i].axis('off')
                        if i == 0:
                            axes[2, i].set_ylabel('β=5', fontsize=10, rotation=0, labelpad=40)
                    
                    plt.suptitle('Reconstruction Quality Comparison', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
    
    st.markdown("---")
    
    with st.expander("📊 Key Observations"):
        st.markdown("""
        ### Expected Differences:
        
        1. **Reconstruction Quality**:
           - β=1: Sharper, more detailed images
           - β=5: Slightly blurrier images (stronger compression)
        
        2. **Latent Space Organization**:
           - β=1: Standard smoothness
           - β=5: More disentangled features, better interpolation
        
        3. **Loss Behavior**:
           - β=5 has higher total loss (due to β weight)
           - β=5 has lower KL divergence (stronger regularization)
           - β=1 has lower reconstruction loss (less compression)
        
        ### The Trade-off:
        Higher β → Better latent structure but lower visual quality
        Lower β → Better visual quality but less organized latent space
        """)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 About")
st.sidebar.info("Interactive VAE Explorer - Learn Variational Autoencoders through hands-on experimentation!")
