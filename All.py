import pandas as pd
import spacy
from pathlib import Path


# Load the CSV file (adjust the path dynamically in the final version)
file_path = Path(__file__).parent / "impo.csv"
df = pd.read_csv(file_path, skiprows=4)

# Rename columns explicitly
df.columns = ["text", "text_en", "cmp_code", "eu_code"]

# Filter valid triplets and labels
df = df.dropna(subset=['cmp_code'])
df = df[df['cmp_code'].apply(lambda x: x.isdigit() and len(x) == 3)]

# Filter rows where cmp_code starts with '4' (Economy domain)
df_economy = df[df['cmp_code'].str.startswith('4')]

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess the text (removes stop words and punctuation)
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.text for token in doc if not token.is_stop and not token.is_punct])

# Apply the preprocessing function to the 'text' column
df_economy['processed_text'] = df_economy['text'].apply(preprocess_text)

# Function to extract relationships (triplets) using dependency parsing
def extract_relationships(text):
    doc = nlp(text)
    relationships = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("nsubj", "dobj"):  # Subject and object detection
                subject = token.text
                verb = token.head.text
                obj = [child for child in token.head.children if child.dep_ == "dobj"]
                obj = obj[0].text if obj else ""
                if subject and verb and obj:  # Ensure valid triplet
                    relationships.append((subject, verb, obj))
    return relationships

# Apply the triplet extraction function
df_economy['relationships'] = df_economy['text'].apply(extract_relationships)

# Filter out any rows with empty relationships
df_economy = df_economy[df_economy['relationships'].apply(lambda x: len(x) > 0)]

# Print a sample of the extracted economy domain triplets
print(df_economy[['text', 'relationships']].head())


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Initialize the Sentence Transformer model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Extract unique entities (head + tail) and relations from the economy triplets
entities = set()
relations = set()
for relationships in df_economy['relationships']:
    for triplet in relationships:
        head, relation, tail = triplet
        entities.update([head, tail])  # Add head and tail to entities
        relations.add(relation)  # Add relation to relations

# Convert sets to lists for further processing
entity_list = list(entities)
relation_list = list(relations)

# Generate embeddings for unique entities and relations
entity_embeddings = model.encode(entity_list)
relation_embeddings = model.encode(relation_list)

# Calculate cosine similarity matrices for entities and relations
entity_similarity_matrix = cosine_similarity(entity_embeddings)
relation_similarity_matrix = cosine_similarity(relation_embeddings)

# Print the entity similarity matrix
print("Entity Similarity Matrix:")
print(entity_similarity_matrix)

# Print the relation similarity matrix
print("\nRelation Similarity Matrix:")
print(relation_similarity_matrix)

# Optionally, print the entity names with their similarity scores (first 10)
print("\nEntity Similarity (first 10):")
for i in range(min(10, len(entity_list))):
    print(f"{entity_list[i]}: {entity_similarity_matrix[i]}")

# Optionally, print the relation names with their similarity scores (first 10)
print("\nRelation Similarity (first 10):")
for i in range(min(10, len(relation_list))):
    print(f"{relation_list[i]}: {relation_similarity_matrix[i]}")


import networkx as nx
import numpy as np

# Initialize a directed graph
G = nx.DiGraph()

# Add nodes for entities
for entity in entity_list:
    G.add_node(entity, label='entity')

# Add nodes for relations
for relation in relation_list:
    G.add_node(relation, label='relation')

# Add edges for triplets with ideology labels and similarity values
for index, row in df_economy.iterrows():
    ideology_label = row['cmp_code']  # Use cmp_code as ideology label
    triplets = row['relationships']
    for head, relation, tail in triplets:
        if head in G.nodes and tail in G.nodes and relation in G.nodes:
            # Add edge from head to tail with the relation as label
            G.add_edge(head, tail, label=relation, ideology=ideology_label)

# Augment the graph with entity similarity values
for i in range(len(entity_list)):
    for j in range(i + 1, len(entity_list)):
        entity_i = entity_list[i]
        entity_j = entity_list[j]
        similarity_value = entity_similarity_matrix[i][j]
        # Add or update undirected similarity edge between entities
        if similarity_value > 0.65:  # Optional: filter by threshold
            G.add_edge(entity_i, entity_j, label='similarity', weight=similarity_value)

# Augment the graph with relation similarity values
for i in range(len(relation_list)):
    for j in range(i + 1, len(relation_list)):
        relation_i = relation_list[i]
        relation_j = relation_list[j]
        similarity_value = relation_similarity_matrix[i][j]
        # Add or update undirected similarity edge between relations
        if similarity_value > 0.65:  # Optional: filter by threshold
            G.add_edge(relation_i, relation_j, label='similarity', weight=similarity_value)

# Print the number of nodes and edges in the knowledge graph
print(f"Knowledge Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Save the graph for further processing (e.g., visualization, GNN application)
nx.write_gexf(G, str(Path(__file__).parent / 'knowledge_graph_with_similarity.gexf'))


import torch
from torch_geometric.utils import from_networkx
import networkx as nx

# Ensure all edges have the same attributes
for u, v, data in G.edges(data=True):
    # Set default values for missing attributes
    if 'label' not in data:
        data['label'] = 'none'  # or some default value
    if 'ideology' not in data:
        data['ideology'] = 'unknown'  # or some default value
    if 'weight' not in data:
        data['weight'] = 1.0  # Set a default similarity weight for edges that lack it

# Convert the NetworkX graph to a PyTorch Geometric Data object
pyg_graph = from_networkx(G)

# Now the graph can be processed using GNN models in PyTorch Geometric
print(pyg_graph)

import torch
import torch_geometric
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Assuming you have entity_embeddings and relation_embeddings as in previous examples
num_entities = len(entity_list)
num_relations = len(relation_list)

# Create node features for entities and relations
node_features = torch.zeros((G.number_of_nodes(), entity_embeddings.shape[1]))

# Assign embeddings to nodes based on whether they are entities or relations
for i, entity in enumerate(entity_list):
    node_idx = list(G.nodes).index(entity)  # Get node index in the graph
    node_features[node_idx] = torch.tensor(entity_embeddings[i])

for i, relation in enumerate(relation_list):
    node_idx = list(G.nodes).index(relation)  # Get node index in the graph
    node_features[node_idx] = torch.tensor(relation_embeddings[i])

# Assign node features to the PyTorch Geometric Data object
pyg_graph.x = node_features

# For node labels (ideology labels for entities, dummy labels for relations)
node_labels = torch.zeros(G.number_of_nodes(), dtype=torch.long)
for entity in entity_list:
    node_idx = list(G.nodes).index(entity)
    ideology_label = df_economy[df_economy['relationships'].apply(lambda x: entity in [t[0] for t in x])]['cmp_code'].iloc[0]
    node_labels[node_idx] = int(ideology_label)  # Convert ideology label to integer

pyg_graph.y = node_labels  # Set labels to PyG graph

# Edge attributes (for similarity)
edge_weights = []
for edge in G.edges(data=True):
    edge_weights.append(edge[2].get('weight', 1.0))  # Use weight if available, else 1.0

pyg_graph.edge_attr = torch.tensor(edge_weights, dtype=torch.float)

import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import torch.nn.functional as F

# Define the GAT model architecture
class GATModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_heads=4):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_node_features, 128, heads=num_heads)  # First GAT layer
        self.conv2 = GATConv(128 * num_heads, 64, heads=num_heads)  # Second GAT layer
        self.fc = torch.nn.Linear(64 * num_heads, num_classes)  # Fully connected layer for output

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # Exponential Linear Unit
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)  # Exponential Linear Unit
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return x

# Create a mapping from ideology code to class index
unique_labels = sorted(df_economy['cmp_code'].unique())
ideology_to_class = {label: idx for idx, label in enumerate(unique_labels)}

# Prepare the target labels for the pyg_graph
pyg_graph.y = torch.tensor(
    [ideology_to_class.get(label, ideology_to_class[unique_labels[-1]]) for label in pyg_graph.y],
    dtype=torch.long
)

# Ensure that the maximum index in pyg_graph.y is less than the number of classes
num_classes = len(ideology_to_class)
assert (pyg_graph.y.max().item() < num_classes), f"Target {pyg_graph.y.max().item()} is out of bounds for number of classes {num_classes}."

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Move the model and data to the correct device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATModel(num_node_features=pyg_graph.x.size(1), num_classes=num_classes).to(device)
pyg_graph = pyg_graph.to(device)

# Training loop
model.train()
for epoch in range(200):  # Train for 200 epochs
    optimizer.zero_grad()
    # Forward pass
    out = model(pyg_graph.x, pyg_graph.edge_index)
    # Compute loss (only on nodes with ideology labels)
    if hasattr(pyg_graph, 'train_mask'):
        loss = criterion(out[pyg_graph.train_mask], pyg_graph.y[pyg_graph.train_mask])
    else:
        loss = criterion(out, pyg_graph.y)
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    # Print training loss
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.optim as optim

# Initialize a directed graph
G = nx.DiGraph()

# Add nodes for entities
for entity in entity_list:
    G.add_node(entity, label='entity')

# Add nodes for relations
for relation in relation_list:
    G.add_node(relation, label='relation')

# Add edges for triplets with ideology labels and similarity values
for index, row in df_economy.iterrows():
    ideology_label = row['cmp_code']  # Use cmp_code as ideology label
    triplets = row['relationships']
    for head, relation, tail in triplets:
        if head in G.nodes and tail in G.nodes and relation in G.nodes:
            G.add_edge(head, tail, label=relation, ideology=ideology_label)

# Augment the graph with entity similarity values
for i in range(len(entity_list)):
    for j in range(i + 1, len(entity_list)):
        entity_i = entity_list[i]
        entity_j = entity_list[j]
        similarity_value = entity_similarity_matrix[i][j]
        if similarity_value > 0.7:
            G.add_edge(entity_i, entity_j, label='similarity', weight=similarity_value)

# Augment the graph with relation similarity values
for i in range(len(relation_list)):
    for j in range(i + 1, len(relation_list)):
        relation_i = relation_list[i]
        relation_j = relation_list[j]
        similarity_value = relation_similarity_matrix[i][j]
        if similarity_value > 0.7:
            G.add_edge(relation_i, relation_j, label='similarity', weight=similarity_value)

print(f"Knowledge Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Ensure all edges have the same attributes
for u, v, data in G.edges(data=True):
    if 'label' not in data:
        data['label'] = 'none'
    if 'ideology' not in data:
        data['ideology'] = 'unknown'
    if 'weight' not in data:
        data['weight'] = 1.0

pyg_graph = from_networkx(G)

# Create node features
node_features = torch.zeros((G.number_of_nodes(), entity_embeddings.shape[1]))
for i, entity in enumerate(entity_list):
    node_idx = list(G.nodes).index(entity)
    node_features[node_idx] = torch.tensor(entity_embeddings[i])

for i, relation in enumerate(relation_list):
    node_idx = list(G.nodes).index(relation)
    node_features[node_idx] = torch.tensor(relation_embeddings[i])

pyg_graph.x = node_features

# Assign labels
node_labels = torch.zeros(G.number_of_nodes(), dtype=torch.long)
for entity in entity_list:
    node_idx = list(G.nodes).index(entity)
    ideology_label = df_economy[df_economy['relationships'].apply(lambda x: entity in [t[0] for t in x])]['cmp_code'].iloc[0]
    node_labels[node_idx] = int(ideology_label)

pyg_graph.y = node_labels

# Edge attributes
edge_weights = [edge[2].get('weight', 1.0) for edge in G.edges(data=True)]
pyg_graph.edge_attr = torch.tensor(edge_weights, dtype=torch.float)

# Define GAT Model
class GATModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_heads=4):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_node_features, 128, heads=num_heads)
        self.conv2 = GATConv(128 * num_heads, 64, heads=num_heads)
        self.fc = torch.nn.Linear(64 * num_heads, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return x

# Prepare model and optimizer
unique_labels = sorted(df_economy['cmp_code'].unique())
ideology_to_class = {label: idx for idx, label in enumerate(unique_labels)}
pyg_graph.y = torch.tensor(
    [ideology_to_class.get(label, ideology_to_class[unique_labels[-1]]) for label in pyg_graph.y],
    dtype=torch.long
)

num_classes = len(ideology_to_class)
model = GATModel(num_node_features=pyg_graph.x.size(1), num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

pyg_graph = pyg_graph.to(device)

# Train/Test split
train_mask = torch.zeros(pyg_graph.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(pyg_graph.num_nodes, dtype=torch.bool)
num_train = int(0.8 * pyg_graph.num_nodes)
train_mask[:num_train] = 1
test_mask[num_train:] = 1
pyg_graph.train_mask = train_mask
pyg_graph.test_mask = test_mask

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(pyg_graph.x, pyg_graph.edge_index)
    loss = criterion(out[pyg_graph.train_mask], pyg_graph.y[pyg_graph.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluation
def evaluate_model(model, pyg_graph):
    model.eval()
    with torch.no_grad():
        out = model(pyg_graph.x, pyg_graph.edge_index)
    preds = out.argmax(dim=1)
    correct = (preds[pyg_graph.test_mask] == pyg_graph.y[pyg_graph.test_mask]).sum().item()
    return correct / pyg_graph.test_mask.sum().item()

test_accuracy = evaluate_model(model, pyg_graph)
print(f"Test Accuracy: {test_accuracy:.4f}")


import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.optim as optim

# Initialize a directed graph
G = nx.DiGraph()

# Add nodes for entities
for entity in entity_list:
    G.add_node(entity, label='entity')

# Add nodes for relations
for relation in relation_list:
    G.add_node(relation, label='relation')

# Add edges for triplets with ideology labels
for index, row in df_economy.iterrows():
    ideology_label = row['cmp_code']  # Use cmp_code as ideology label
    triplets = row['relationships']
    for head, relation, tail in triplets:
        if head in G.nodes and tail in G.nodes and relation in G.nodes:
            G.add_edge(head, tail, label=relation, ideology=ideology_label)

# Print the number of nodes and edges in the knowledge graph
print(f"Knowledge Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Ensure all edges have the same attributes
for u, v, data in G.edges(data=True):
    if 'label' not in data:
        data['label'] = 'none'
    if 'ideology' not in data:
        data['ideology'] = 'unknown'

pyg_graph_no_edges = from_networkx(G)

# Create node features for entities and relations
node_features = torch.zeros((G.number_of_nodes(), entity_embeddings.shape[1]))
for i, entity in enumerate(entity_list):
    node_idx = list(G.nodes).index(entity)
    node_features[node_idx] = torch.tensor(entity_embeddings[i])

for i, relation in enumerate(relation_list):
    node_idx = list(G.nodes).index(relation)
    node_features[node_idx] = torch.tensor(relation_embeddings[i])

pyg_graph_no_edges.x = node_features

# Assign labels for ideology codes
node_labels = torch.zeros(G.number_of_nodes(), dtype=torch.long)
for entity in entity_list:
    node_idx = list(G.nodes).index(entity)
    ideology_label = df_economy[df_economy['relationships'].apply(lambda x: entity in [t[0] for t in x])]['cmp_code'].iloc[0]
    node_labels[node_idx] = int(ideology_label)

pyg_graph_no_edges.y = node_labels

# Define the GAT model
class GATModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_heads=4):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_node_features, 128, heads=num_heads)
        self.conv2 = GATConv(128 * num_heads, 64, heads=num_heads)
        self.fc = torch.nn.Linear(64 * num_heads, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return x

# Training and testing functions
def train_model(model, pyg_graph, device, epochs=200):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    pyg_graph = pyg_graph.to(device)
    model = model.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(pyg_graph.x, pyg_graph.edge_index)
        loss = criterion(out, pyg_graph.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def test_model(model, pyg_graph, device):
    model.eval()
    pyg_graph = pyg_graph.to(device)
    with torch.no_grad():
        out = model(pyg_graph.x, pyg_graph.edge_index)
    preds = out.argmax(dim=1)
    correct = (preds == pyg_graph.y).sum().item()
    accuracy = correct / pyg_graph.y.size(0)
    return accuracy

# Example split and dual graph training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unique_labels = sorted(df_economy['cmp_code'].unique())
ideology_to_class = {label: idx for idx, label in enumerate(unique_labels)}

pyg_graph_no_edges.y = torch.tensor(
    [ideology_to_class.get(label, ideology_to_class[unique_labels[-1]]) for label in pyg_graph_no_edges.y],
    dtype=torch.long
)

num_classes = len(ideology_to_class)

# Train and test models for graph with edges and without edges
model_no_edges = GATModel(num_node_features=pyg_graph_no_edges.x.size(1), num_classes=num_classes).to(device)
train_model(model_no_edges, pyg_graph_no_edges, device)
test_accuracy_no_edges = test_model(model_no_edges, pyg_graph_no_edges, device)
print(f"Test Accuracy (without edges): {test_accuracy_no_edges:.4f}")


from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.parse
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch

# Function to generate DBpedia URI from entity name
def generate_dbpedia_uri(entity_name):
    base_uri = "http://dbpedia.org/resource/"
    entity_uri = base_uri + urllib.parse.quote(str(entity_name).replace(" ", "_"))
    return entity_uri

# Function to query DBpedia for additional relationships
def query_dbpedia(entity_uri):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    query = f"""
    SELECT ?property ?value
    WHERE {{
        <{entity_uri}> ?property ?value
    }} LIMIT 100
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print(f"Error querying DBpedia for {entity_uri}: {e}")
        return []

# Enrich PyG graph using DBpedia
def enrich_pyg_graph_no_edges(pyg_graph):
    # Convert PyG graph to NetworkX for easier manipulation
    nx_graph = to_networkx(pyg_graph, to_undirected=True)
    # Create a static list of nodes to iterate over
    nodes = list(nx_graph.nodes)
    # Add edges based on DBpedia
    for node in nodes:
        entity_uri = generate_dbpedia_uri(node)
        dbpedia_data = query_dbpedia(entity_uri)
        for result in dbpedia_data:
            property_uri = result["property"]["value"]
            value = result["value"]["value"]
            if result["value"]["type"] == "uri":
                # Add new edge with property as label
                nx_graph.add_edge(node, value, label=property_uri)
            else:
                # Add new node attribute if the value is not a URI
                if node in nx_graph.nodes:
                    nx_graph.nodes[node][property_uri] = value
    # Convert back to PyTorch Geometric
    enriched_pyg_graph = from_networkx(nx_graph)
    # Add original node features and labels
    enriched_pyg_graph.x = pyg_graph.x  # Preserve original features
    enriched_pyg_graph.y = pyg_graph.y  # Preserve original labels
    return enriched_pyg_graph

# Assuming pyg_graph_no_edges is already defined
# Enrich the graph
pyg_graph_enriched_no_edges = enrich_pyg_graph_no_edges(pyg_graph_no_edges)

# Print details of the enriched graph
print(f"Enriched PyG graph: {pyg_graph_enriched_no_edges.num_nodes} nodes, {pyg_graph_enriched_no_edges.num_edges} edges.")

# Save the enriched graph for future use
torch.save(pyg_graph_enriched_no_edges, str(Path(__file__).parent / 'pyg_graph_enriched_no_edges.pt'))
print("Enriched PyG graph saved as 'pyg_graph_enriched_no_edges.pt'.")
