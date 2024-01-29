import json
import pandas as pd

# Load the JSON data
with open('/Users/bellavg/PycharmProjects/DIG_FACT/benchmark_dataset/data_merging_and_processing/original_datafiles/CongressionalTwitterNetwork/congress_network_data.json', 'r') as file:
    network_data = json.load(file)[0]

usernames_of_interest = pd.read_csv("/Users/bellavg/PycharmProjects/DIG_FACT/benchmark_dataset/encoded_data.csv")["twitter"]


# Prepare a dictionary to hold all the data
all_data = {}

# Find the indices of these usernames
indices_of_interest = [i for i, username in enumerate(network_data['usernameList']) if username in usernames_of_interest.values]

for index in indices_of_interest:
    in_connections = network_data['inList'][index]
    in_weights = network_data['inWeight'][index]
    out_connections = network_data['outList'][index]
    out_weights = network_data['outWeight'][index]

    # Translate connection indices back to usernames for readability
    in_usernames = [network_data['usernameList'][i] for i in in_connections]
    out_usernames = [network_data['usernameList'][i] for i in out_connections]

    # Populate the dictionary with structured data
    all_data[network_data['usernameList'][index]] = {
        'incoming_connections': {
            'usernames': in_usernames,
            'weights': in_weights
        },
        'outgoing_connections': {
            'usernames': out_usernames,
            'weights': out_weights
        }
    }

# Write the dictionary to a file as JSON
with open('/Users/bellavg/PycharmProjects/DIG_FACT/benchmark_dataset/data_merging_and_processing/connections_weights.json', 'w') as outfile:
    json.dump(all_data, outfile, indent=4)