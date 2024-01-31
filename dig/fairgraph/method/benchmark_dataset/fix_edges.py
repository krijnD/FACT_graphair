import json
import pandas as pd

# Read the file content
with open('congress.edgelist', 'r') as file:
    file_content = file.read()

# Split the file content into lines
lines = file_content.split('\n')

# Process each line to remove the weight part
processed_lines = []
for line in lines:
    if line:  # Make sure the line is not empty
        parts = line.split(' ')
        # Keep only the node identifiers (which are the first two parts)
        processed_line = ' '.join(parts[:2])
        processed_lines.append(processed_line)

# Combine the processed lines back into a single string
processed_content = '\n'.join(processed_lines)

# Now 'processed_content' is the file content without the weight attributes
print(processed_content)

# If you want to write the processed content back to a file
with open('congress.edgelist', 'w') as file:
    file.write(processed_content)