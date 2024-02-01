import numpy as np
from scipy.stats import lognorm

# Define the median net worth based on age
age_net_worth = {
    (0, 34): 39000,
    (35, 44): 135600,
    (45, 54): 247200,
    (55, 64): 364500,
    (65, 74): 409900,
    (75, float('inf')): 335600,
}


def get_age_based_net_worth(age):
    for age_range, median in age_net_worth.items():
        if age_range[0] <= age <= age_range[1]:
            return median
    return 0


# Update race and gender multipliers based on ratio to White Male
race_gender_multiplier = {
    'White Male': 1,
    'White Female': 36000 / 78200,
    'African American Male': 10100 / 78200,
    'African American Female': 1700 / 78200,
    'Hispanic Male': 4200 / 78200,
    'Hispanic Female': 1000 / 78200,
    'Other Male': 1,
    'Other Female': 1,
}

# Define the additional wealth adjustment for LGBTQ+ status
lgbtq_wealth_adjustment = 0.82

# Updated Education Net Worth Adjustments
education_net_worth = {
    'No College': 20500,
    'High School': 74000,
    'Some College': 88800,
    'Bachelor': 302200,
    'Master': 350000,
    'Law': 400000,
    'Doctorate': 450000,
}

# Updated Occupation Multipliers
occupation_multipliers = {
    'Public Service/Politics': 1.079,
    'Education': 1.05,
    'Business': 1.163,
    'Law': 1.226,
    'Healthcare': 1.049,
    'Other': 1.0,
    'Military': 1.0,
}



# Function to estimate net worth with min, max constraints, and party factor
def estimate_net_worth(row):
    # Base calculation as before
    base_net_worth = get_age_based_net_worth(row['Age'])
    race_gender_key = f"{row['Race/Ethnicity']} {'Male' if row['Gender'] == 'Male' else 'Female'}"
    adjusted_net_worth = base_net_worth * race_gender_multiplier.get(race_gender_key, 1)

    if row['LGBTQ+ Status'] == 'LGBTQ+':
        adjusted_net_worth *= lgbtq_wealth_adjustment

    education_level = row['Education']
    adjusted_net_worth *= education_net_worth.get(education_level, 20500) / education_net_worth['No College']

    occupation_key = row['Occupation']
    adjusted_net_worth *= occupation_multipliers.get(occupation_key, 1.0)

    adjusted_net_worth *= np.log1p(row['Campaign_Expenditure']) / 100000

    # Apply party-based adjustment (Republicans have more money)
    if row['Party'] == 'Republican':
        adjusted_net_worth *= 1.2  # Adjust this multiplier as needed

    # Apply min and max constraints
    adjusted_net_worth = max(100000, adjusted_net_worth)  # Minimum net worth constraint
    adjusted_net_worth = min(adjusted_net_worth, 5000000000)  # Maximum net worth constraint ($5bil)

    # Generate a skewed distribution around the adjusted net worth
    distribution = lognorm(s=0.5, scale=adjusted_net_worth)
    return distribution.rvs()

