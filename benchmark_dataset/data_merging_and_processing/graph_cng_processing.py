import torch
import json
import pandas as pd
import networkx as nx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch.optim import Adam
from torch_geometric.utils import to_dense_adj

class EmbeddingGNN(torch.nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(EmbeddingGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, embedding_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

# Load the JSON data for connections
with open('connections_weights.json', 'r') as file:
    connections_data = json.load(file)

# Initialize a directed graph and add edges
G = nx.DiGraph()
for user, connections in connections_data.items():
    for target, weight in zip(connections['outgoing_connections']['usernames'],
                              connections['outgoing_connections']['weights']):
        G.add_edge(user, target, weight=weight)

# Load demographics CSV and ensure it's indexed by 'twitter'
df_demo = pd.read_csv('/Users/bellavg/PycharmProjects/DIG_FACT/benchmark_dataset/encoded_data.csv')

# Filter the DataFrame to include only the users present in the graph
# Ensure 'df_demo' is indexed by 'twitter'
df_demo.set_index('twitter', inplace=True)

# Explicitly convert G.nodes() to a list
node_list = list(G.nodes())

# Use the list of nodes to index into the DataFrame
df_ordered = df_demo.loc[node_list]
# Convert features to tensor
features_tensor = torch.tensor(df_ordered.values, dtype=torch.float)

# Prepare PyTorch Geometric data
edge_weights = torch.tensor([G[u][v]['weight'] for u, v in G.edges()], dtype=torch.float)
data = Data(x=features_tensor, edge_index=torch.tensor(list(G.edges)).t().contiguous().long(), edge_attr=edge_weights)

# Initialize the model
model = EmbeddingGNN(num_features=features_tensor.size(1), embedding_dim=64)
optimizer = Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    adj_dense = to_dense_adj(data.edge_index, edge_attr=data.edge_attr).squeeze(0)
    out_adj_dense = torch.matmul(out, out.t())
    loss = F.mse_loss(out_adj_dense, adj_dense)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate and save embeddings
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index, data.edge_attr)
torch.save(embeddings, 'cng_embeddings.pt')