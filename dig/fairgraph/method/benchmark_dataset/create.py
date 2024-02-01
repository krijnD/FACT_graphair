import numpy as np
import pandas as pd
# Change to numerical features
import scipy.stats as stats
from calcl_networth import estimate_net_worth

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

# Number of Senators and Representatives
num_senators = 100
num_representatives = 375  #

# Generate the 'Chamber' column
chamber = ['Senate'] * num_senators + ['House'] * num_representatives

# If the total_members is greater than the sum of senators and representatives,
# we can randomly assign the remaining members to either chamber based on proportion
extra_members = total_members - (num_senators + num_representatives)
if extra_members > 0:
    extra_chambers = np.random.choice(['Senate', 'House'], extra_members, p=[num_senators/475, num_representatives/475])
    chamber.extend(extra_chambers)

# Shuffle the chamber list to randomize the assignment before adding to DataFrame
np.random.shuffle(chamber)

# Add the 'Chamber' column to the DataFrame
congress_df['Chamber'] = chamber





# Example function to generate "Experience in Congress Years"
# Assuming an average of 8.5 years for representatives and 11.2 years for senators

def generate_terms_served(total_members):
    return np.random.normal(2, 1, total_members).astype(int)
def generate_congress_experience(total_members, num_senators=100):
    reps_experience = np.random.normal(8.5, 3, total_members - num_senators)
    senators_experience = np.random.normal(11.2, 3, num_senators)
    return np.concatenate((reps_experience, senators_experience)).astype(int)



# Function to generate approval ratings for a synthetic dataset
def generate_approval_ratings(total_members, mean_approval=23, std_dev=15):
    # Generate approval ratings from a normal distribution centered around the mean approval rate
    # Clipped to ensure ratings are between 0% and 100%
    ratings = np.random.normal(mean_approval, std_dev, total_members)
    ratings = np.clip(ratings, 0, 100).astype(int)
    return ratings

# Total members
total_members = 475

# Party expenditure statistics
republican_expenditure_total = 4.2e9
democrat_expenditure_total = 4e9

# Average expenditures
average_senate_expenditure = 13.5e6  # Average for Senate
average_house_expenditure = 1.8e6    # Average for House

def generate_campaign_expenditure(member_party, chamber):
    if chamber == 'Senate':
        base_expenditure = average_senate_expenditure
    else:  # 'House'
        base_expenditure = average_house_expenditure

    # Republicans spend more than Democrats, reflect this in the expenditure
    if member_party == 'Republican':
        expenditure_multiplier = republican_expenditure_total / (
                    republican_expenditure_total + democrat_expenditure_total)
    else:
        expenditure_multiplier = democrat_expenditure_total / (
                    republican_expenditure_total + democrat_expenditure_total)

    # Sample from a normal distribution around the adjusted base expenditure
    expenditure = np.random.normal(base_expenditure * expenditure_multiplier,
                                   base_expenditure * 0.1)  # 10% std deviation
    return max(0, expenditure)  # Ensure non-negative

# Generating synthetic dataset with approval ratings
# Apply the function to generate campaign expenditure for each member
congress_df['Campaign_Expenditure'] = congress_df.apply(
    lambda row: generate_campaign_expenditure(row['Party'], row['Chamber']), axis=1
)
# Apply the function to estimate net worth for each row in the dataframe
congress_df['Estimated Net Worth'] = congress_df.apply(estimate_net_worth, axis=1)

# LES values for a subset of Democrats and Republicans from the provided data
les_values_democrats = [
    1.472, 2.500, 1.438, 2.917, 1.438, 2.432, 2.520, 1.048, 1.533, 2.500,
    # ... add all other LES values for Democrats
]

les_values_republicans = [
    0.572, 0.722, 0.586, 0.579, 0.579, 0.613, 0.606, 0.586, 0.572, 0.586,
    # ... add all other LES values for Republicans
]

# Calculate the mean LES for Democrats
mean_les_democrats = np.mean(les_values_democrats)

# Calculate the standard deviation LES for Democrats
std_les_democrats = np.std(les_values_democrats)

# Calculate the mean LES for Republicans
mean_les_republicans = np.mean(les_values_republicans)

# Calculate the standard deviation LES for Republicans
std_les_republicans = np.std(les_values_republicans)

les_stats = {
    'Democrat': {'mean': mean_les_democrats, 'std': std_les_democrats},
    'Republican': {'mean': mean_les_republicans, 'std': std_les_republicans}
}


def sample_legislative_effectiveness_score(party):
    # Get the statistics for the party
    stats = les_stats.get(party, {'mean': 1.0, 'std': 0.5})

    # Generate a score from a normal distribution
    score = np.random.normal(stats['mean'], stats['std'])

    # Ensure the score is non-negative
    score = max(score, 0)

    return score


# Example usage:
# Assuming 'members_df' is your DataFrame with a 'Party' column
congress_df['Legislative_Effectiveness_Score'] = congress_df['Party'].apply(sample_legislative_effectiveness_score)

congress_df["Approval_Rating"] = generate_approval_ratings(total_members)

congress_df['Terms_Served'] = generate_terms_served(total_members)
congress_df['Experience_in_Congress_Years'] = generate_congress_experience(total_members)


def generate_twitter_followers(party):
    # Placeholder average and standard deviation values based on image data
    # Replace with actual values derived from the image data
    twitter_stats = {
        'Democrat': {'mean': 500000, 'std': 100000},
        'Republican': {'mean': 250000, 'std': 50000}
    }

    # Get the mean and standard deviation for the given party
    mean_followers = twitter_stats[party]['mean']
    std_followers = twitter_stats[party]['std']

    # Generate a follower count from a normal distribution
    followers = np.random.normal(mean_followers, std_followers)

    # Ensure the follower count is non-negative and round to the nearest whole number
    followers = max(int(round(followers)), 0)

    return followers


# Example usage
# Assuming 'congress_df' is your DataFrame with a 'Party' column
np.random.seed(42)  # For reproducible results
congress_df['Party'] = np.random.choice(['Democrat', 'Republican'], size=475)
congress_df['Twitter_Followers'] = congress_df['Party'].apply(generate_twitter_followers)


# Function to generate the number of bills introduced
def generate_bills_introduced(party):
    # Placeholder values for the average and standard deviation of bills introduced
    # You should replace these with your actual calculated values
    party_averages = {
        'Republican': {'mean': 75, 'std': 25},
        'Democrat': {'mean': 85, 'std': 20}
    }

    # Get the mean and standard deviation for the party
    mean = party_averages.get(party, {'mean': 80})['mean']  # Default mean if party not found
    std = party_averages.get(party, {'std': 22.5})['std']  # Default standard deviation if party not found

    # Generate a random number of bills introduced from a normal distribution
    num_bills_introduced = np.random.normal(mean, std)

    # Return the number of bills, making sure it's a non-negative integer
    return max(0, int(num_bills_introduced))


# Assume we have a DataFrame with a 'Party' column
np.random.seed(42)  # For consistent random results
# Apply the function to generate the number of bills introduced for each member
congress_df['Bills_Introduced'] = congress_df['Party'].apply(generate_bills_introduced)


def generate_approval_rating(party, party_approval_stats):
    # Define the approval ratings based on party from the provided statistics
    # This is an example based on your description; you should adjust it using the actual data from the image
    approval_ratings = {
        'Republican': {
            'mean': 45,  # Mean approval for Republicans
            'std': 10  # Standard deviation for approval ratings
        },
        'Democrat': {
            'mean': 26,  # Mean approval for Democrats
            'std': 10  # Standard deviation for approval ratings
        }
    }

    # Check if the party is in the approval ratings dictionary
    if party in approval_ratings:
        # Get the mean and standard deviation for the party
        mean = approval_ratings[party]['mean']
        std = approval_ratings[party]['std']

        # Generate a random approval rating from a normal distribution
        # Clipped to the range 0-100
        approval_rating = np.clip(np.random.normal(mean, std), 0, 100)
    else:
        # Default to a neutral approval rating if party is not recognized
        approval_rating = 50

    return int(approval_rating)



# Party approval statistics, which you would extract from the image provided
party_approval_stats = {
    # These values should be derived from your image data
    'Republican': {'mean': 45, 'std': 10},
    'Democrat': {'mean': 26, 'std': 10}
}

# Apply the function to generate approval ratings for each member
congress_df['Approval_Rating'] =congress_df['Party'].apply(
    lambda party: generate_approval_rating(party, party_approval_stats))

# Display sample of the dataset
congress_df.to_csv("./generated_congress.csv")

# One-hot encoding of categorical columns
categorical_columns = ['LGBTQ+ Status', 'Party', 'Gender', 'Education', 'Occupation', 'Race/Ethnicity', 'Religion', 'Military Service', 'Chamber', 'Party']
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
