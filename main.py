import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import os
import argparse
import numpy as np
from circuits import cluster_latent_neurons, ablate_non_circuit_neurons
from detailed_visualization import visualize_detailed_circuits

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Memory optimization settings
torch.backends.cuda.max_split_size_mb = 512
batch_size = 16
accumulation_steps = 4


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ActivationsDataset(torch.utils.data.Dataset):
    def __init__(self, temp_dir, num_chunks):
        self.temp_dir = temp_dir
        self.num_chunks = num_chunks
        self.chunk_cache = None
        self.current_chunk_idx = None

        # Load first chunk to get dimensions
        first_chunk = torch.load(f'{temp_dir}/chunk_0.pt')
        self.chunk_sizes = [first_chunk.size(0)]
        self.dim = first_chunk.size(1)
        del first_chunk

        # Get sizes of all chunks
        for i in range(1, num_chunks):
            chunk = torch.load(f'{temp_dir}/chunk_{i}.pt')
            self.chunk_sizes.append(chunk.size(0))
            del chunk

        self.total_size = sum(self.chunk_sizes)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        chunk_idx = 0
        local_idx = idx
        while local_idx >= self.chunk_sizes[chunk_idx]:
            local_idx -= self.chunk_sizes[chunk_idx]
            chunk_idx += 1

        if self.current_chunk_idx != chunk_idx:
            self.chunk_cache = torch.load(
                f'{self.temp_dir}/chunk_{chunk_idx}.pt')
            self.current_chunk_idx = chunk_idx

        return self.chunk_cache[local_idx]

# GPT-2 Functions


def extract_activations(text, layer_index, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    return hidden_states[layer_index]


def preprocess_data(dataset, tokenizer, model, layer_index, batch_size=32, chunk_size=1000):
    """Process data in chunks and save to disk to prevent memory overflow"""
    os.makedirs('temp_activations', exist_ok=True)

    num_samples = len(dataset)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    logging.info("Extracting activations from the dataset...")

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_samples)

        chunk_activations = []
        for i in tqdm(range(start_idx, end_idx, batch_size),
                      desc=f"Processing chunk {chunk_idx+1}/{num_chunks}"):
            batch_end = min(i + batch_size, end_idx)
            batch = dataset[i:batch_end]

            batch_activations = []
            for example in batch['sentence']:
                with torch.no_grad():
                    activations = extract_activations(
                        example, layer_index, model, tokenizer)
                    mean_activations = torch.mean(
                        activations.squeeze(0), dim=0)
                    batch_activations.append(mean_activations.cpu())

            batch_tensor = torch.stack(batch_activations)
            chunk_activations.append(batch_tensor)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        chunk_tensor = torch.cat(chunk_activations, dim=0)
        torch.save(chunk_tensor, f'temp_activations/chunk_{chunk_idx}.pt')
        del chunk_activations, chunk_tensor
        torch.cuda.empty_cache()

    return ActivationsDataset('temp_activations', num_chunks)

# Training Functions


def train_autoencoder_with_sparsity(data_loader, model, criterion, optimizer,
                                    l1_lambda=0.001, start_epoch=0, epochs=50, accumulation_steps=4,
                                    checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    start_epoch = 0
    losses = []
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        logging.info(f"Resuming from epoch {start_epoch}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)
    best_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for i, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            data = (data - data.mean()) / (data.std() + 1e-6)

            outputs = model(data)
            mse_loss = criterion(outputs, data)
            l1_loss = l1_lambda * torch.norm(model.encoder[0].weight, 1)
            loss = (mse_loss + l1_loss) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps

        avg_loss = epoch_loss/len(data_loader)
        losses.append(avg_loss)
        logging.info(
            f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'losses': losses
        }, checkpoint_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(
                checkpoint_dir, 'best_model.pt'))

        if (epoch + 1) % 5 == 0:
            visualize_training_progress(losses,
                                        save_path=os.path.join(checkpoint_dir, f'training_progress_epoch_{epoch+1}.png'))

        scheduler.step(avg_loss)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return losses

# Visualization Functions


def visualize_training_progress(losses, save_path='training_progress.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Autoencoder Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def visualize_latent_space(latent_representations, labels=None, save_path='latent_space.png'):
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_representations.numpy())

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                          c=labels if labels is not None else None,
                          cmap='viridis', alpha=0.6)

    if labels is not None:
        plt.colorbar(scatter)

    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(save_path)
    plt.close()

# Utility Functions


def get_latent_representations(data_loader, model):
    logging.info("Extracting latent representations...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_latent = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Extracting latent representations"):
            data = data.to(device)
            data = (data - data.mean()) / (data.std() + 1e-6)
            encoded = model.encoder(data)
            all_latent.append(encoded.cpu())

    return torch.cat(all_latent, dim=0)


def load_or_create_model(checkpoint_dir='checkpoints', input_dim=768, hidden_dim=1024):
    model = SparseAutoencoder(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-5)
    start_epoch = 0
    losses = []

    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        logging.info("Loading existing checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        logging.info(f"Resumed from epoch {start_epoch}")
    else:
        logging.info("No checkpoint found. Starting from scratch.")

    return model, optimizer, start_epoch, losses


def load_or_process_activations(dataset, tokenizer, model, layer_index):
    if os.path.exists('temp_activations') and len(os.listdir('temp_activations')) > 0:
        logging.info("Loading existing activations...")
        num_chunks = len([f for f in os.listdir('temp_activations')
                         if f.startswith('chunk_') and f.endswith('.pt')])
        return ActivationsDataset('temp_activations', num_chunks)
    else:
        logging.info("Processing new activations...")
        return preprocess_data(dataset, tokenizer, model, layer_index)


def main(resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GPT-2 model and tokenizer
    model_name = 'gpt2'
    gpt2_model = GPT2Model.from_pretrained(
        model_name, output_hidden_states=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset('glue', 'sst2', split='train')

    if resume:
        model, optimizer, start_epoch, losses = load_or_create_model()
        activations_dataset = load_or_process_activations(
            dataset, tokenizer, gpt2_model, layer_index=5)

        data_loader = torch.utils.data.DataLoader(
            activations_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=2
        )

        if start_epoch < 50:  # num_epochs = 50
            criterion = nn.MSELoss(reduction='mean')
            new_losses = train_autoencoder_with_sparsity(
                data_loader,
                model,
                criterion,
                optimizer,
                start_epoch=start_epoch,
                epochs=50
            )
            losses.extend(new_losses)

        visualize_training_progress(losses)
        latent_representations = get_latent_representations(data_loader, model)

        # Cluster the latent neurons
        clusters, neuron_embeddings = cluster_latent_neurons(
            latent_representations, n_clusters=5)
        print("Identified circuits (clusters):", clusters)

        # Ablate non-circuit neurons
        target_circuit = 0  # For example, circuit #0
        modified_latents = ablate_non_circuit_neurons(
            latent_representations, clusters, target_circuit)

        # You can then probe the performance of your downstream classifier with modified_latents
        visualize_detailed_circuits(
            modified_latents, clusters, save_path='detailed_circuit_visualization.png')

    else:
        # Run everything from scratch
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    args = parser.parse_args()

    main(resume=args.resume)
