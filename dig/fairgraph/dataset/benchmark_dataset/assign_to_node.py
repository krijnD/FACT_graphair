import networkx as nx
from networkx.algorithms import community as nx_community
from collections import Counter
from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd
from itertools import cycle


# Load the edge list from the provided file
file_path = './congress.edgelist'
G = nx.read_edgelist(file_path)

# Check the basic information of the graph to understand its structure



# Since the Girvan-Newman algorithm can be computationally expensive for large graphs,
# and it produces a dendrogram of communities, we'll start its computation but might need to
# adjust based on the graph's size and computational constraints.
# We will find just the top level of communities for demonstration.

# Find communities using the Girvan-Newman algorithm
communities_generator = nx_community.girvan_newman(G)
top_level_communities = next(communities_generator)
sorted_communities = sorted(map(sorted, top_level_communities))

# Number of communities found
num_communities = len(sorted_communities)
print(num_communities)

#Check labeled data
# Load the GEXF file to construct the graph
gexf_file_path = './115th Senate retweets.gexf'
G_senate = nx.read_gexf(gexf_file_path)

# Check the basic information of the Senate interactions graph



# For a directed graph like the Senate interactions graph, we'll convert it to an undirected graph for community detection
G_senate_undirected = G_senate.to_undirected()

# Apply the Louvain method for community detection
# As the 'community' module is not available, we'll use an alternative approach that's compatible with networkx
# Attempting a method suitable for undirected graphs within networkx's capabilities
communities_senate = nx_community.louvain_communities(G_senate_undirected)

# Number of communities found in the Senate interactions graph
num_communities_senate = len(communities_senate)

# Use the Label Propagation algorithm for community detection in the undirected Senate graph
communities_label_propagation = list(nx_community.label_propagation_communities(G_senate_undirected))

# Number of communities found using Label Propagation
num_communities_label_propagation = len(communities_label_propagation)

# Extract node attributes, specifically looking for party affiliation
node_attributes = nx.get_node_attributes(G_senate, 'party')

# Analyze the composition of each community in terms of party affiliation
cluster_1_party_affiliation = [node_attributes[node] for node in communities_label_propagation[0]]
cluster_2_party_affiliation = [node_attributes[node] for node in communities_label_propagation[1]]

# Count the number of Republicans and Democrats in each cluster

cluster_1_party_count = Counter(cluster_1_party_affiliation)
cluster_2_party_count = Counter(cluster_2_party_affiliation)

print(cluster_1_party_count, cluster_2_party_count)

# Cluster 1: Comprises entirely of 37 Republican (R) members.
# Cluster 2: Includes 47 Democrat (D) members, 15 Republican (R) members, and 2 Independent (I) members.

# Convert the graph into an adjacency matrix
adjacency_matrix = nx.to_numpy_array(G)

# Apply Spectral Clustering to partition the graph into 2 clusters
spectral_model = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize', random_state=42)
labels = spectral_model.fit_predict(adjacency_matrix)

# Check the balance of the partition
partition_balance = Counter(labels)

# Assign nodes to clusters based on the labels
cluster_spectral_1 = [str(node) for node, label in zip(G.nodes, labels) if label == 0]
cluster_spectral_2 = [str(node) for node, label in zip(G.nodes, labels) if label == 1]

print(partition_balance, len(cluster_spectral_1), len(cluster_spectral_2))


# Load the uploaded CSV file containing the fake Congress data
fake_congress_data_path = './encoded_congress.csv'
fake_congress_df = pd.read_csv(fake_congress_data_path)
#fake_congress_df = fake_congress_df.drop("Unnamed: 0", axis=1)
# Preview the first few rows of the dataframe to understand its structure
fake_congress_df.head()

# Partition the dataframe into Democrats and Republicans
democrats_df = fake_congress_df[fake_congress_df['Party_Democrat.1'] == 1]
republicans_df = fake_congress_df[fake_congress_df['Party_Republican.1'] == 1]

# Since the number of entries in each partition may not exactly match the number of nodes in each cluster,
# we'll cycle through the node IDs for assignment to ensure each entry gets a node ID.

# Create cycles of node IDs for each cluster
np.random.shuffle(np.array(cluster_spectral_1))
np.random.shuffle(np.array(cluster_spectral_2))

ids = np.concatenate([cluster_spectral_1, cluster_spectral_2])

# Combine the partitions back into a single dataframe
assigned_node_df = pd.concat([democrats_df, republicans_df])
assert len(set(ids)) == len(assigned_node_df)
assigned_node_df["numeric_id"] = ids


assigned_node_df.to_csv('./encoded_congress.csv', index=False)