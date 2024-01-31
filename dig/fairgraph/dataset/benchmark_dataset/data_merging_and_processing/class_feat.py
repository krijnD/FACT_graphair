import pandas as pd

# df = pd.read_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/dataset/benchmark_dataset/full_with_ed.csv")
#
# # Assuming df is your DataFrame with the 'Average Net Worth' column already cleaned
# df['Average Net Worth'] = df['Average Net Worth'].str.replace('$', '').str.replace(',', '')
# df['Average Net Worth'] = pd.to_numeric(df['Average Net Worth'])
#
# # Calculate the median net worth
# median_net_worth = df['Average Net Worth'].median()

# Load your datasets
#to_change1 = pd.read_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/dataset/benchmark_dataset/encoded_data.csv")
to_change2 = pd.read_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/dataset/benchmark_dataset/encoded_data_wo_religions.csv")
#
# # Define a buffer range (20% of the median)
# buffer_range = 0.25 * median_net_worth
#
# # Create the classification based on the median net worth
# for dataset in [to_change1, to_change2]:
#     dataset['class_net_worth'] = pd.cut(df['Average Net Worth'],
#                                          bins=[float('-inf'), median_net_worth - buffer_range, median_net_worth + buffer_range, float('inf')],
#                                          labels=[-1, 0, 1],
#                                          include_lowest=True)
#
#
# # Convert the 'value_assignment' column to integer type
# to_change1['class_net_worth'] = to_change1['class_net_worth'].astype(int)
#
# # Convert the 'value_assignment' column to integer type
# to_change2['class_net_worth'] = to_change2['class_net_worth'].astype(int)
#
# # Calculate and print the distribution of 'class_net_worth' in to_change1
# distribution_to_change1 = to_change1['class_net_worth'].value_counts()
# print("Distribution in to_change1:")
# print(distribution_to_change1)
#
# # Calculate and print the distribution of 'class_net_worth' in to_change2
# distribution_to_change2 = to_change2['class_net_worth'].value_counts()
# print("\nDistribution in to_change2:")
# print(distribution_to_change2)
#
# to_change1.to_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/dataset/benchmark_dataset/encoded_data.csv", index=False)
# to_change1.to_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/method/benchmark_dataset/encoded_data.csv", index=False)

print(to_)

to_change2.to_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/dataset/benchmark_dataset/encoded_data_wo_religions.csv", index=False)
to_change2.to_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/method/benchmark_dataset/encoded_data_wo_religions.csv", index=False)
