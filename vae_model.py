"""
VAE Model Definition
Contains Encoder, Decoder, and complete VAE architecture
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder network that compresses images into latent distributions.
    
    Architecture:
    - Input: 32×32×3 image
    - Output: μ (mean) and log σ² (log variance) of latent distribution
    """
    
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 32×32×3 → 16×16×32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16×16×32 → 8×8×64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8×8×64 → 4×4×128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4×4×128 → 2×2×256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Flatten size: 256 * 2 * 2 = 1024
        self.flatten_size = 256 * 2 * 2
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: Input image tensor [batch_size, 3, 32, 32]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network that reconstructs images from latent codes.
    
    Architecture:
    - Input: Latent vector z
    - Output: Reconstructed 32×32×3 image
    """
    
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Project latent vector to initial feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 2 * 2),
            nn.ReLU(inplace=True)
        )
        
        # Transposed convolutions for upsampling
        self.deconv_layers = nn.Sequential(
            # 2×2×256 → 4×4×128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 4×4×128 → 8×8×64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 8×8×64 → 16×16×32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 16×16×32 → 32×32×3
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, z):
        """
        Forward pass through decoder.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            Reconstructed image [batch_size, 3, 32, 32]
        """
        x = self.fc(z)
        x = x.view(x.size(0), 256, 2, 2)  # Reshape to 2×2×256
        x = self.deconv_layers(x)
        return x


class VAE(nn.Module):
    """
    Complete Variational Autoencoder.
    
    Combines encoder and decoder with reparameterization trick.
    """
    
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ × ε
        
        This allows gradients to flow through the sampling operation.
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
            
        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)  # Convert log(σ²) to σ
        epsilon = torch.randn_like(std)  # Sample ε ~ N(0, 1)
        z = mu + std * epsilon  # Reparameterization
        return z
    
    def forward(self, x):
        """
        Forward pass through complete VAE.
        
        Args:
            x: Input image [batch_size, 3, 32, 32]
            
        Returns:
            x_reconstructed: Reconstructed image [batch_size, 3, 32, 32]
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_reconstructed = self.decoder(z)
        
        return x_reconstructed, mu, logvar
    
    def generate(self, num_samples, device):
        """
        Generate new images from random latent vectors.
        
        Args:
            num_samples: Number of images to generate
            device: Device to run generation on
            
        Returns:
            Generated images [num_samples, 3, 32, 32]
        """
        # Sample random latent vectors from standard normal
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Decode to images
        with torch.no_grad():
            samples = self.decoder(z)
        
        return samples
    
    def encode(self, x):
        """
        Encode images to latent space.
        
        Args:
            x: Input images [batch_size, 3, 32, 32]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent vectors to images.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
            
        Returns:
            Decoded images [batch_size, 3, 32, 32]
        """
        return self.decoder(z)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total: Total number of parameters
        trainable: Number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = VAE(latent_dim=128).to(device)
    print(f"VAE model created on {device}")
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    print(f"\nInput shape: {x.shape}")
    
    x_recon, mu, logvar = model(x)
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test generation
    samples = model.generate(8, device)
    print(f"\nGenerated {samples.shape[0]} images with shape {samples.shape}")
    
    print("\n✅ Model test passed!")
