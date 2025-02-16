import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def cluster_latent_neurons(latent_reps, n_clusters=5):
    """
    Cluster latent neurons based on their activation patterns across samples.
    
    This function interprets each latent dimension (neuron) as a vector of activations 
    across all samples. It then applies PCA for noise reduction and KMeans to group 
    neurons into candidate circuits.
    
    Args:
        latent_reps (torch.Tensor): Latent representations of shape (num_samples, hidden_dim).
        n_clusters (int): Number of clusters to form.
        
    Returns:
        clusters (np.ndarray): An array of cluster labels for each latent neuron (shape: (hidden_dim,)).
        neuron_embeddings (np.ndarray): PCA-reduced embeddings of neurons used for clustering.
    """
    # Convert latent representations from torch.Tensor to numpy array (num_samples, hidden_dim)
    latent_np = latent_reps.cpu().numpy()
    # Transpose so that each row corresponds to a latent neuron and contains activations over samples
    neuron_activations = latent_np.T  # shape becomes (hidden_dim, num_samples)
    
    # Optionally reduce dimensionality (noise reduction) with PCA
    pca = PCA(n_components=min(50, neuron_activations.shape[1]))
    neuron_embeddings = pca.fit_transform(neuron_activations)
    
    # Cluster neurons using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(neuron_embeddings)
    
    return clusters, neuron_embeddings

def ablate_non_circuit_neurons(latent_reps, clusters, target_cluster):
    """
    Retain only the neurons that belong to the target cluster (circuit) and ablate (zero out) the others.
    
    Args:
        latent_reps (torch.Tensor): Latent representations of shape (num_samples, hidden_dim).
        clusters (np.ndarray): Cluster labels for each neuron, with shape (hidden_dim,).
        target_cluster (int): The label of the circuit (cluster) to retain.
        
    Returns:
        modified_latent (torch.Tensor): Modified latent representations where neurons not in the target
                                        circuit are set to zero.
    """
    # Create a boolean mask for which neurons to keep
    mask = (clusters == target_cluster)
    # Convert mask to a torch tensor and reshape so it can broadcast over samples
    mask_tensor = torch.tensor(mask, dtype=latent_reps.dtype, device=latent_reps.device).unsqueeze(0)
    modified_latent = latent_reps * mask_tensor  # Zero out neurons not in target circuit
    return modified_latent

if __name__ == "__main__":
    # Example usage with random data for demonstration purposes.
    # Imagine you have latent representations of shape (num_samples, hidden_dim)
    latent_reps = torch.randn(100, 64)  # e.g., 100 samples and 64 latent features
    clusters, neuron_embeddings = cluster_latent_neurons(latent_reps, n_clusters=5)
    print("Cluster assignments for neurons:", clusters)
    
    # Suppose we want to keep neurons from circuit labeled as '0' only.
    modified_latent = ablate_non_circuit_neurons(latent_reps, clusters, target_cluster=0)
    print("Modified latent representations shape:", modified_latent.shape) 