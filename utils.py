"""
Utility Functions for VAE Training and Visualization
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def load_cifar10_data(batch_size=128, num_workers=2):
    """
    Load and prepare CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    # Normalization to [-1, 1] range (for Tanh output)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Calculate VAE loss (ELBO).
    
    Loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        recon_x: Reconstructed images [batch_size, 3, 32, 32]
        x: Original images [batch_size, 3, 32, 32]
        mu: Mean of latent distribution [batch_size, latent_dim]
        logvar: Log variance of latent distribution [batch_size, latent_dim]
        beta: Weight for KL divergence term (default 1.0)
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss
    """
    batch_size = x.size(0)
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    
    # KL divergence loss
    # KL(N(μ, σ²) || N(0, 1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_vae_epoch(model, train_loader, device, beta=1.0, lr=1e-3):
    """
    Train VAE for one epoch.
    
    Args:
        model: VAE model
        train_loader: DataLoader for training data
        device: Device to train on
        beta: Beta parameter for β-VAE
        lr: Learning rate
        
    Returns:
        avg_total_loss: Average total loss for the epoch
        avg_recon_loss: Average reconstruction loss
        avg_kl_loss: Average KL divergence loss
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epoch_total = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0
    num_batches = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # Calculate loss
        loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        epoch_total += loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()
        num_batches += 1
    
    # Calculate averages
    avg_total_loss = epoch_total / num_batches
    avg_recon_loss = epoch_recon / num_batches
    avg_kl_loss = epoch_kl / num_batches
    
    return avg_total_loss, avg_recon_loss, avg_kl_loss


def plot_training_curves(history, title="Training Progress"):
    """
    Plot training loss curves.
    
    Args:
        history: Dictionary with 'total_loss', 'recon_loss', 'kl_loss' lists
        title: Title for the plot
        
    Returns:
        fig: Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['total_loss']) + 1)
    
    # Total Loss
    axes[0].plot(epochs, history['total_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss (ELBO)')
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[1].plot(epochs, history['recon_loss'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss (MSE)')
    axes[1].grid(True, alpha=0.3)
    
    # KL Divergence
    axes[2].plot(epochs, history['kl_loss'], 'r-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('KL Divergence')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def generate_image_grid(model, device, num_images=16, nrow=4, title="Generated Images"):
    """
    Generate a grid of images from random latent vectors.
    
    Args:
        model: VAE model
        device: Device to generate on
        num_images: Number of images to generate
        nrow: Number of images per row
        title: Title for the plot
        
    Returns:
        fig: Matplotlib figure object
    """
    model.eval()
    
    with torch.no_grad():
        # Sample random latent vectors
        z = torch.randn(num_images, model.latent_dim).to(device)
        
        # Generate images
        generated = model.decoder(z)
        
        # Denormalize from [-1, 1] to [0, 1]
        generated = generated * 0.5 + 0.5
        generated = generated.cpu()
    
    # Create grid
    fig, axes = plt.subplots(nrow, nrow, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = generated[i].numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def show_reconstructions(model, data_loader, device, num_images=8):
    """
    Show original images vs their reconstructions.
    
    Args:
        model: VAE model
        data_loader: DataLoader for images
        device: Device to run on
        num_images: Number of images to show
        
    Returns:
        fig: Matplotlib figure object
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        recon, _, _ = model(images)
    
    # Denormalize
    images = images * 0.5 + 0.5
    recon = recon * 0.5 + 0.5
    
    # Plot
    fig, axes = plt.subplots(2, num_images, figsize=(14, 4))
    
    for i in range(num_images):
        # Original
        orig = images[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(np.clip(orig, 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        
        # Reconstruction
        rec = recon[i].cpu().numpy().transpose(1, 2, 0)
        axes[1, i].imshow(np.clip(rec, 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12)
    
    plt.suptitle('Original vs Reconstructed Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def latent_space_interpolation(model, device, num_steps=10, num_rows=3):
    """
    Perform linear interpolation between random latent vectors.
    
    Args:
        model: VAE model
        device: Device to run on
        num_steps: Number of interpolation steps
        num_rows: Number of interpolation rows
        
    Returns:
        fig: Matplotlib figure object
    """
    model.eval()
    
    fig, axes = plt.subplots(num_rows, num_steps, figsize=(15, 5))
    
    for row in range(num_rows):
        # Sample two random latent vectors
        z1 = torch.randn(1, model.latent_dim).to(device)
        z2 = torch.randn(1, model.latent_dim).to(device)
        
        # Create interpolation steps
        alphas = np.linspace(0, 1, num_steps)
        
        for i, alpha in enumerate(alphas):
            # Linear interpolation: z = (1-α)*z1 + α*z2
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Generate image
            with torch.no_grad():
                img = model.decoder(z_interp)
            
            # Denormalize and display
            img = img * 0.5 + 0.5
            img = img.squeeze().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            
            if num_rows == 1:
                ax = axes[i]
            else:
                ax = axes[row, i]
            
            ax.imshow(img)
            ax.axis('off')
            
            # Label first and last
            if row == 0:
                if i == 0:
                    ax.set_title('z₁', fontsize=10)
                elif i == num_steps - 1:
                    ax.set_title('z₂', fontsize=10)
                else:
                    ax.set_title(f'α={alpha:.1f}', fontsize=8)
    
    plt.suptitle(f'Latent Space Interpolation ({num_steps} Steps)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def save_model(model, filepath):
    """
    Save model weights to file.
    
    Args:
        model: VAE model
        filepath: Path to save model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """
    Load model weights from file.
    
    Args:
        model: VAE model (initialized)
        filepath: Path to load model from
        device: Device to load model to
        
    Returns:
        model: Model with loaded weights
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {filepath}")
    return model


def denormalize_image(img_tensor):
    """
    Denormalize image tensor from [-1, 1] to [0, 1].
    
    Args:
        img_tensor: Image tensor in [-1, 1] range
        
    Returns:
        Denormalized image tensor in [0, 1] range
    """
    return img_tensor * 0.5 + 0.5


def tensor_to_numpy(img_tensor):
    """
    Convert image tensor to numpy array for visualization.
    
    Args:
        img_tensor: Image tensor [C, H, W]
        
    Returns:
        Numpy array [H, W, C]
    """
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img, 0, 1)


if __name__ == "__main__":
    print("Utility functions module loaded successfully!")
    print("\nAvailable functions:")
    print("  - load_cifar10_data()")
    print("  - vae_loss()")
    print("  - train_vae_epoch()")
    print("  - plot_training_curves()")
    print("  - generate_image_grid()")
    print("  - show_reconstructions()")
    print("  - latent_space_interpolation()")
    print("  - save_model() / load_model()")
