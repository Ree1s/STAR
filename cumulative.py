#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

##############################################################################
# 1. Generate or load sample latent data
##############################################################################
def get_sample_latent_data(b=4, c=8, t=5, h=16, w=16):
    """
    Create a random (b, c, t, h, w) tensor for demonstration.
    Replace this with actual VAE latents in practice.
    """
    return torch.randn(b, c, t, h, w)

##############################################################################
# 2. PCA analysis for Temporal, Spatial, and Channel dimensions
##############################################################################

def analyze_temporal_pca(latent):
    """
    Perform PCA on the time dimension of a (b, c, t, h, w) tensor.
    * Treat 'time' as features (columns)
    * Flatten (b, c, h, w) as samples (rows).
    Returns the PCA object and explained_variance_ratio_.
    """
    b, c, t, h, w = latent.shape
    # Move 't' to last -> shape (b, c, h, w, t), then flatten -> (b*c*h*w, t)
    data_2d = latent.permute(0, 1, 3, 4, 2).reshape(-1, t)
    data_np = data_2d.detach().cpu().numpy()

    pca = PCA(n_components=t)
    pca.fit(data_np)
    evr = pca.explained_variance_ratio_
    return pca, evr

def analyze_spatial_pca(latent):
    """
    Perform PCA on the spatial dimensions of a (b, c, t, h, w) tensor.
    * Flatten (b, c, t) as samples (rows)
    * Flatten (h, w) as features (columns).
    Returns the PCA object and explained_variance_ratio_.
    """
    b, c, t, h, w = latent.shape
    # shape -> (b*c*t, h*w)
    data_2d = latent.reshape(b * c * t, h * w)
    data_np = data_2d.detach().cpu().numpy()

    # If h*w is huge, limit n_components to something smaller like 50 or 100
    pca = PCA(n_components=min(h*w, 50))
    pca.fit(data_np)
    evr = pca.explained_variance_ratio_
    return pca, evr

def analyze_channel_pca(latent):
    """
    Perform PCA on the channel dimension of a (b, c, t, h, w) tensor.
    * Flatten (b, t, h, w) as samples (rows)
    * 'c' as features (columns).
    Returns the PCA object and explained_variance_ratio_.
    """
    b, c, t, h, w = latent.shape
    # permute -> shape (b, t, h, w, c), then flatten -> (b*t*h*w, c)
    data_2d = latent.permute(0, 2, 3, 4, 1).reshape(-1, c)
    data_np = data_2d.detach().cpu().numpy()

    # If c is large, limit n_components
    pca = PCA(n_components=min(c, 16))
    pca.fit(data_np)
    evr = pca.explained_variance_ratio_
    return pca, evr

##############################################################################
# 3. Plotting function for Cumulative Explained Variance
##############################################################################

def plot_cumulative_evr(evr, title="Cumulative Explained Variance"):
    """
    Plots a bar chart of individual EVR and a line of cumulative EVR.
    evr: 1D array-like of explained_variance_ratio_ from PCA.
    """
    cum_evr = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)

    plt.figure(figsize=(6, 4))
    # Individual EVR as bars
    plt.bar(x, evr, alpha=0.5, label="Individual EVR")
    # Cumulative EVR as a line
    plt.plot(x, cum_evr, marker='o', color='red', label="Cumulative EVR")
    # Horizontal line for reference (e.g., 90% coverage)
    plt.axhline(y=0.9, color='green', linestyle='--', label="0.9 Threshold")

    plt.xlabel("Principal Component Index")
    plt.ylabel("Explained Variance Ratio")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}.png")

##############################################################################
# 4. Main script to tie everything together
##############################################################################

def main():
    # Load or generate sample data
    latent = torch.load("latents.pt", map_location="cpu")
    print("Latent shape:", latent.shape)

    # 1) Temporal PCA
    pca_time, evr_time = analyze_temporal_pca(latent)
    plot_cumulative_evr(evr_time, title="Temporal-Cumulative EVR")

    # 2) Spatial PCA
    pca_space, evr_space = analyze_spatial_pca(latent)
    plot_cumulative_evr(evr_space, title="Spatial-Cumulative EVR")

    # 3) Channel PCA
    pca_channel, evr_channel = analyze_channel_pca(latent)
    plot_cumulative_evr(evr_channel, title="Channel-Cumulative EVR")

if __name__ == "__main__":
    main()
