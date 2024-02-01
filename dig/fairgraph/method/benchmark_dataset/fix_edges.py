import json
import pandas as pd

# Read the file content
# with open('congress.edgelist', 'r') as file:
#     file_content = file.read()
#
# # Split the file content into lines
# lines = file_content.split('\n')
#
# # Process each line to remove the weight part
# processed_lines = []
# for line in lines:
#     if line:  # Make sure the line is not empty
#         parts = line.split(' ')
#         # Keep only the node identifiers (which are the first two parts)
#         processed_line = ' '.join(parts[:2])
#         processed_lines.append(processed_line)
#
# # Combine the processed lines back into a single string
# processed_content = '\n'.join(processed_lines)
#
# # Now 'processed_content' is the file content without the weight attributes
# print(processed_content)
#
# # If you want to write the processed content back to a file
# with open('congress.edgelist', 'w') as file:
#     file.write(processed_content)
real_cng_data = pd.read_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/method/benchmark_dataset/encoded_data.csv")
# real_cng_data = real_cng_data.drop(["first_name_vector", "last_name_vector"], axis=1)
# real_cng_data = real_cng_data.rename(columns={"class_net_worth":"NET_WORTH" , "gender_feat":"Gender_Female"})
# real_cng_data.to_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/method/benchmark_dataset/encoded_data.csv", index=False)
