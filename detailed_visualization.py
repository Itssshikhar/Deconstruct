import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def visualize_detailed_circuits(latent_reps, clusters, save_path='detailed_circuit_visualization.png'):
    """
    Visualize circuit clusters in the latent space with detailed plots.
    
    This function performs PCA to reduce the latent representations to 2D, creates a
    scatter plot colored by circuit clusters, overlays KDE contours for each cluster,
    and also produces separate histograms displaying the distribution of the principal components.
    
    Args:
        latent_reps (torch.Tensor or np.ndarray): The latent representations 
            of shape (num_samples, hidden_dim).
        clusters (np.ndarray): Cluster labels for each latent neuron used to color the plot.
        save_path (str): Path to save the detailed circuit visualization.
    """
    # Convert latent representations to numpy if needed
    if not isinstance(latent_reps, np.ndarray):
        latent_np = latent_reps.cpu().numpy()
    else:
        latent_np = latent_reps

    # Perform PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_np)

    # Create a DataFrame for plotting with seaborn
    df = pd.DataFrame({
        'PC1': latent_2d[:, 0],
        'PC2': latent_2d[:, 1],
        'Cluster': clusters.astype(str)
    })

    # Set up the matplotlib figure
    plt.figure(figsize=(14, 10))

    # Scatter plot with hue defined by clusters
    ax = sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette="viridis",
                         data=df, s=50, alpha=0.7, edgecolor="none")

    # Overlay KDE contours for each cluster
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        subset = df[df['Cluster'] == str(cluster)]
        sns.kdeplot(x=subset['PC1'], y=subset['PC2'], ax=ax, 
                    levels=5, color=None, fill=False, alpha=0.3)

    ax.set_title('Detailed Circuit Visualization (PCA with KDE Contours)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True)
    plt.legend(title='Circuit Cluster')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Create additional plots for component distributions.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df['PC1'], kde=True, color="skyblue", ax=axes[0])
    axes[0].set_title("Distribution of PC1")
    sns.histplot(df['PC2'], kde=True, color="salmon", ax=axes[1])
    axes[1].set_title("Distribution of PC2")
    plt.tight_layout()
    plt.savefig('detailed_component_distributions.png')
    plt.close() 