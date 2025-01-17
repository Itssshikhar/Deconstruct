# Deconstruct

This repository contains code and resources for analyzing and interpreting neuron activations in deep language models, such as GPT-2, using Sparse Autoencoders (SAEs). The project focuses on decomposing neuron activations, probing their latent structure, and clustering activations for interpretability.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

---

## Overview

This project aims to explore mechanistic interpretability of transformer-based language models by:
- Extracting hidden state activations from specific layers.
- Using Sparse Autoencoders to uncover sparse latent structures in activations.
- Probing latent representations with supervised classifiers.
- Performing unsupervised clustering for interpretability.

Key focus: The method provides insights into the roles of individual neurons and their grouped behavior, contributing to understanding model decision-making.

---

## Features

1. **Activation Extraction**: Extract hidden state activations from any layer of GPT-2.
2. **Sparse Autoencoder Training**:
   - Reconstructs activations while enforcing sparsity in latent representations.
3. **Probing Task**: Evaluates the quality of latent representations using linear classifiers.
4. **Clustering**: Groups activations into clusters and analyzes patterns across them.
5. **Visualization and Analysis**: Provides a framework for qualitative inspection of clusters.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers
- scikit-learn
- Datasets

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neuron-decomposition-sae.git
   cd neuron-decomposition-sae
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Extracting Activations
The code extracts activations from a specified layer of GPT-2:
```python
layer_index = 5  # Choose a layer
activations_dataset = preprocess_data(dataset, tokenizer, model, layer_index)
```

### 2. Training Sparse Autoencoder
Define and train the Sparse Autoencoder:
```python
model = SparseAutoencoder(input_dim, hidden_dim)
train_autoencoder_with_sparsity(data_loader, model, criterion, optimizer)
```

### 3. Probing Latent Representations
Train a logistic regression classifier on latent representations:
```python
latent_representations = get_latent_representations(data_loader, model)
classifier.fit(latent_representations.numpy(), labels.numpy())
accuracy = accuracy_score(labels.numpy(), predictions)
print(f'Probing Accuracy: {accuracy:.4f}')
```

### 4. Clustering
Cluster latent representations and analyze results:
```python
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(latent_representations.numpy())
```

---

## Results

- **Probing Accuracy**: A measure of how well the latent representations capture task-specific information.
- **Cluster Analysis**: Insights into common patterns in neuron activations and the sentences associated with different clusters.

---

## Contributing

Contributions are welcome! If you'd like to improve the code, add features, or suggest ideas:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of changes.

---

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

