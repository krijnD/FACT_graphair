import numpy as np
import pandas as pd
# Change to numerical features
import scipy.stats as stats

# Seed for reproducibility
np.random.seed(42)

# Total members
total_members = 475

# Age Distribution (using given averages and approximating a range)
rep_age_mean, rep_age_std = 57.9, 10
sen_age_mean, sen_age_std = 64, 10

# Education Levels (Approximations based on given percentages)
education_levels = ['Bachelor', 'Master', 'Law', 'Doctorate', 'No College']
# Adjust education distribution to sum to 1
education_distribution = [0.6, 0.2, 0.15, 0.04, 0.01]  # Adjusted for sum to 1

# Occupation Categories (Based on the document's info)
occupations = ['Public Service/Politics', 'Business', 'Law', 'Education', 'Healthcare', 'Military', 'Other']
occupation_distribution = [0.3, 0.25, 0.2, 0.1, 0.05, 0.05, 0.05]

# Race/Ethnicity
races = ['White', 'African American', 'Hispanic or Latino', 'Asian American/Pacific Islander', 'Native American',
         'Other']
race_distribution = [0.6, 0.115, 0.115, 0.04, 0.009, 0.121]  # Adjusted based on given numbers

# Military Service
military_service = [True, False]
military_service_distribution = [0.181, 0.819]  # 18.1% with military service

current_year = 2024
gen_z_start = current_year - 1996
millennial_start = current_year - 1981
gen_x_start = current_year - 1965
boomer_start = current_year - 1946
silent_start = current_year - 1928


def determine_generation(age):
    birth_year = current_year - age
    if birth_year >= gen_z_start:
        return 'Gen Z'
    elif birth_year >= millennial_start:
        return 'Millennial'
    elif birth_year >= gen_x_start:
        return 'Gen X'
    elif birth_year >= boomer_start:
        return 'Boomer'
    else:
        return 'Silent'


# Religion distribution (approximated based on general knowledge)
religions = ['Christianity', 'Judaism', 'Islam', 'Buddhism', 'Hinduism', 'Non-Affiliated/Other']
religion_distribution = [0.7, 0.06, 0.02, 0.01, 0.01,
                         0.2]  # Adjusted to reflect general religious affiliations in Congress

# Generating synthetic dataset
data = {
    'LGBTQ+ Status': np.random.choice(['LGBTQ+', 'Non-LGBTQ+'], total_members, p=[0.02, .98]),
    "Party": np.random.choice(["Democrat", "Republican"], total_members, p=[216 / 438, 222 / 438]),
    "Gender": np.random.choice(["Male", "Female"], total_members, p=[1 - 0.2865, .2865]),
    "Age": np.concatenate((np.random.normal(rep_age_mean, rep_age_std, total_members - 100),
                           np.random.normal(sen_age_mean, sen_age_std, 100))).astype(int),
    "Education": np.random.choice(education_levels, total_members, p=education_distribution),
    "Occupation": np.random.choice(occupations, total_members, p=occupation_distribution),
    "Race/Ethnicity": np.random.choice(races, total_members, p=race_distribution),
    "Religion": np.random.choice(religions, total_members, p=religion_distribution),
    "Military Service": np.random.choice(military_service, total_members, p=military_service_distribution)
}

# Create DataFrame
congress_df = pd.DataFrame(data)

# Define the median net worth based on age from the image provided
age_net_worth = {
    (0, 34): 39000,
    (35, 44): 135600,
    (45, 54): 247200,
    (55, 64): 364500,
    (65, 74): 409900,
    (75, float('inf')): 335600,
}


# Helper function to get the median net worth based on age
def get_age_based_net_worth(age):
    for age_range, median in age_net_worth.items():
        if age_range[0] <= age <= age_range[1]:
            return median
    return 0


# Define the adjustment multipliers for race and gender
race_gender_multiplier = {
    'White Male': 78200 / 1000000,
    'White Female': 81200 / 1000000,
    'African American Male': 10100 / 1000000,
    'African American Female': 1700 / 1000000,
    'Hispanic Male': 4200 / 1000000,
    'Hispanic Female': 1000 / 1000000,
    'Other Male': 43800 / 1000000,
    'Other Female': 36000 / 1000000,
}

# Define the additional wealth adjustment for LGBTQ+ status
lgbtq_wealth_adjustment = 0.82  # 82% of the median wealth compared to non-LGBTQ+


# Function to estimate net worth based on member characteristics
def estimate_net_worth(row):
    # Get the base median net worth based on age
    base_net_worth = get_age_based_net_worth(row['Age'])

    # Determine the race and gender key
    race_gender_key = f"{row['Race/Ethnicity']} {'Male' if row['Gender'] == 'Male' else 'Female'}"

    # Adjust the base net worth for race and gender
    adjusted_net_worth = base_net_worth * race_gender_multiplier.get(race_gender_key, 1)

    # Further adjust for LGBTQ+ status if applicable
    if row.get('LGBTQ+ Status', 'Non-LGBTQ+') == 'LGBTQ+':
        adjusted_net_worth *= lgbtq_wealth_adjustment

    # Congress members are generally wealthier, set minimum at $500,000
    adjusted_net_worth = max(adjusted_net_worth, 500000)

    # Assuming a skewed distribution, we use lognormal to represent high wealth inequality
    # We scale the median net worth and use a small sigma to have a tight distribution around the median
    distribution = stats.lognorm(s=0.5, scale=adjusted_net_worth)
    return distribution.rvs()


# Apply the function to estimate net worth for each row in the dataframe
congress_df['Estimated Net Worth'] = congress_df.apply(estimate_net_worth, axis=1)

# Display sample of the dataset
congress_df.to_csv("./generated_congress.csv")

# One-hot encoding of categorical columns
categorical_columns = ['LGBTQ+ Status', 'Party', 'Gender', 'Education', 'Occupation', 'Race/Ethnicity', 'Religion', 'Military Service']
congress_df_encoded = pd.get_dummies(congress_df, columns=categorical_columns)
congress_df_encoded = congress_df_encoded.astype(float)
# Function to categorize net worth

# Calculate median net worth from the generated data
median_net_worth = congress_df['Estimated Net Worth'].median()
# Define the buffer as 25% of the median net worth
net_worth_buffer = median_net_worth * 0.25
def categorize_net_worth(net_worth):
    if net_worth >= median_net_worth + net_worth_buffer:
        return 1
    elif median_net_worth - net_worth_buffer <= net_worth <= median_net_worth + net_worth_buffer:
        return 0
    else:
        return -1

# Apply the function to create the NET_WORTH category
congress_df_encoded['NET_WORTH'] = congress_df['Estimated Net Worth'].apply(categorize_net_worth)
congress_df_encoded = congress_df_encoded.drop('Estimated Net Worth', axis=1)

congress_df_encoded.to_csv("./encoded_congress.csv", index_label="numeric_id")
