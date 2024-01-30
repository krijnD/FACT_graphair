import json

# Read the JSON file
with open('/benchmark_dataset/data_merging_and_processing/connections_weights.json', 'r') as infile:
    data = json.load(infile)

with open('/benchmark_dataset/cng_relationship.txt', 'w') as outfile:
    for user, connections in data.items():
        # Extracting outgoing connections as an example
        for target in connections['outgoing_connections']['usernames']:
            outfile.write(f"{user}\t{target}\n")