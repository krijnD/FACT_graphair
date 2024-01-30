import json
import pandas as pd

with open(
        './data_merging_and_processing/original_datafiles/CongressionalTwitterNetwork/congress_network_data.json', 'r') as file:
    network_data = json.load(file)[0]


df_of_interest = pd.read_csv("./full_with_ed.csv")
usernames_of_interest = df_of_interest["twitter"].values


username_to_id_mapping = {username: i for i, username in enumerate(network_data['usernameList'])}
# Find the indices of these usernames

indices_of_interest = [i for i, username in enumerate(network_data['usernameList']) if username in usernames_of_interest]

all_data = {}

for index in indices_of_interest:
    in_connections = network_data['inList'][index]
    in_weights = network_data['inWeight'][index]
    out_connections = network_data['outList'][index]
    out_weights = network_data['outWeight'][index]

    # Translate connection indices back to usernames for readability
    in_usernames = [network_data['usernameList'][i] for i in in_connections]
    out_usernames = [network_data['usernameList'][i] for i in out_connections]

    # Filter in_usernames and out_usernames to include only those in your list of interest
    filtered_in_usernames = [username for username in in_usernames if username in usernames_of_interest]
    filtered_out_usernames = [username for username in out_usernames if username in usernames_of_interest]

    # Then use these filtered lists instead of the original in_usernames and out_usernames
    all_data[network_data['usernameList'][index]] = {
        'incoming_connections': {
            'usernames': filtered_in_usernames,
            'weights': in_weights
            # Assuming the weights correspond directly; you may need to adjust this if filtering affects alignment
        },
        'outgoing_connections': {
            'usernames': filtered_out_usernames,
            'weights': out_weights  # Same assumption as above
        }
    }


with open('./cng_relationship.txt', 'w') as outfile:
    for user, connections in all_data.items():
        # Extracting outgoing connections as an example
        for target in connections['outgoing_connections']['usernames']:
            outfile.write(f"{username_to_id_mapping[user]}\t{username_to_id_mapping[target]}\n")