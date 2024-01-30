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

from node2vec import Node2Vec
import networkx as nx

# Load the JSON data for connections
with open('connections_weights.json', 'r') as file:
    connections_data = json.load(file)

# Initialize a directed graph
G = nx.DiGraph()

# Add edges with weights
for user, connections in connections_data.items():
    for target, weight in zip(connections['outgoing_connections']['usernames'],
                              connections['outgoing_connections']['weights']):
        # Ensure that both the user and the target are nodes in the graph
        G.add_edge(user, target, weight=weight)

# Configure Node2Vec parameters
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4, weight_key='weight')

# Train the Node2Vec model (this may take some time depending on the size of your graph and the parameters)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Save the embeddings to a file
with open('node_embeddings.txt', 'w') as f:
    for node in model.wv.key_to_index.keys():
        embeddings = model.wv.get_vector(node)
        f.write(f"{node} {' '.join(map(str, embeddings))}\n")