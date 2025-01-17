import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# Load GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to extract activations
def extract_activations(text, layer_index):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    return hidden_states[layer_index]

# Load the SST-2 dataset
dataset = load_dataset('glue', 'sst2', split='train')

# Preprocess the dataset to extract activations
def preprocess_data(dataset, tokenizer, model, layer_index):
    activations_list = []
    for example in dataset:
        text = example['sentence']
        activations = extract_activations(text, layer_index)
        activations_list.append(activations.squeeze(0))  # Remove batch dimension
    return torch.stack(activations_list)

# Extract activations for the entire dataset
layer_index = 5  # Choose a specific layer
activations_dataset = preprocess_data(dataset, tokenizer, model, layer_index)

# Define the Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
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

# Hyperparameters
input_dim = activations_dataset.size(-1)  # e.g., 768 for GPT-2 small
hidden_dim = 128  # Smaller dimension for sparsity

# Initialize model, loss function, and optimizer
model = SparseAutoencoder(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Modified training loop with L1 regularization
def train_autoencoder_with_sparsity(data_loader, model, criterion, optimizer, l1_lambda=0.001, epochs=10):
    for epoch in range(epochs):
        for data in data_loader:
            optimizer.zero_grad()
            outputs = model(data)
            mse_loss = criterion(outputs, data)
            l1_loss = l1_lambda * torch.norm(model.encoder[0].weight, 1)
            loss = mse_loss + l1_loss
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}')

# Create a DataLoader for the activations dataset
data_loader = torch.utils.data.DataLoader(activations_dataset, batch_size=32, shuffle=True)

# Train the model with sparsity
train_autoencoder_with_sparsity(data_loader, model, criterion, optimizer)

# Extract latent representations
def get_latent_representations(data_loader, model):
    latent_representations = []
    for data in data_loader:
        with torch.no_grad():
            encoded = model.encoder(data)
            latent_representations.append(encoded)
    return torch.cat(latent_representations)

# Prepare data for probing
latent_representations = get_latent_representations(data_loader, model)
labels = torch.tensor([example['label'] for example in dataset])

# Train a linear classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(latent_representations.numpy(), labels.numpy())

# Evaluate the classifier
predictions = classifier.predict(latent_representations.numpy())
accuracy = accuracy_score(labels.numpy(), predictions)
print(f'Probing Accuracy: {accuracy:.4f}')

# Perform clustering
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(latent_representations.numpy())

# Analyze clusters
for cluster_id in range(5):
    print(f"Cluster {cluster_id}:")
    cluster_indices = (clusters == cluster_id).nonzero()[0]
    for idx in cluster_indices[:5]:  # Show first 5 examples
        print(dataset[idx]['sentence'])
