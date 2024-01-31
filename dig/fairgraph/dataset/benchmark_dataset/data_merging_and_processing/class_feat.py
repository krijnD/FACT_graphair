import pandas as pd

df = pd.read_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/dataset/benchmark_dataset/full_with_ed.csv")

# Remove the '$' symbol from the 'money_column'
df['Average Net Worth'] = df['Average Net Worth'].str.replace('$', '').str.replace(',', '')

# Convert the column to a numeric type (e.g., float) if needed
df['Average Net Worth'] = pd.to_numeric(df['Average Net Worth'])

average_net_worth = df['Average Net Worth'].mean()

to_change1 = pd.read_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/dataset/benchmark_dataset/encoded_data.csv")

to_change2 = pd.read_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/dataset/benchmark_dataset/encoded_data_wo_religions.csv")
# Define a buffer range (10% of the average)
buffer_range = 0.2 * average_net_worth
# 1 if above the average with a 10 pecent buffer -1 if below net wroth average of congress and 0 if around the same within 10 percent
# Create a new column 'value_assignment' based on conditions
# average of average net worth of these congress members is 6007207.75984252
to_change1['class_net_worth'] = pd.cut(df['Average Net Worth'],
                                bins=[float('-inf'), average_net_worth - buffer_range, average_net_worth + buffer_range, float('inf')],
                                labels=[-1, 0, 1],
                                include_lowest=True)

to_change2['class_net_worth'] = pd.cut(df['Average Net Worth'],
                                bins=[float('-inf'), average_net_worth - buffer_range, average_net_worth + buffer_range, float('inf')],
                                labels=[-1, 0, 1],
                                include_lowest=True)


# Convert the 'value_assignment' column to integer type
to_change1['class_net_worth'] = to_change1['class_net_worth'].astype(int)

# Convert the 'value_assignment' column to integer type
to_change2['class_net_worth'] = to_change2['class_net_worth'].astype(int)

to_change1.to_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/dataset/benchmark_dataset/encoded_data.csv", index=False)
to_change1.to_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/method/benchmark_dataset/encoded_data.csv", index=False)

to_change2.to_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/dataset/benchmark_dataset/encoded_data_wo_religions.csv", index=False)
to_change2.to_csv("/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/method/benchmark_dataset/encoded_data_wo_religions.csv", index=False)
