from scipy.stats import ttest_ind

import numpy as np
from scipy.stats import ttest_ind_from_stats

# Your results (mean, standard deviation)
nba_results = [
    (69.36, 0.45),  # ACC R
    (60.99, 2.87),  # ACC A
    (60.99, 1.35),  # ACC C
    (72.77, 0.59),  # ACC default
    (2.56, 0.41),   # ΔDP R
    (9.55, 1.62),   # ΔDP A
    (4.91, 3.62),   # ΔDP C
    (6.59, 0.59),   # ΔDP default
    (4.64, 0.17),   # ΔEO R
    (21.47, 2.38),  # ΔEO A
    (6.56, 3.00),   # ΔEO C
    (18.09, 1.44)   # ΔEO default
]

pokecz_results = [
    (68.17, 0.08),  # ACC R
    (62.47, 0.74),  # ACC A, C
    (60.35, 1.26),  # ACC default
    (2.10, 0.17),   # ΔDP R
    (1.02, 0.48),   # ΔDP A, C
    (4.68, 1.12),   # ΔDP default
    (2.76, 0.19),   # ΔEO R
    (0.70, 0.51),   # ΔEO A, C
    (4.15, 0.73)    # ΔEO default
]

pokecn_results = [
    (67.43, 0.25),  # ACC R
    (64.91, 0.29),  # ACC A, C
    (62.33, 1.36),  # ACC default
    (2.02, 0.40),   # ΔDP R
    (0.26, 0.22),   # ΔDP A, C
    (9.74, 1.84),   # ΔDP default
    (1.62, 0.47),   # ΔEO R
    (0.58, 0.34),   # ΔEO A, C
    (10.01, 2.51)   # ΔEO default
]
# Assuming a sample size of 5 for each experiment
sample_size = 5

# Function to perform the t-tests
def t_tests( results):
    t_stats = []
    p_values = []

    # Perform t-tests for each metric within the dataset
    for i in range(0, len(results), 4):  # Loop through each metric for all sources
        for j in range(1,4):  # Loop through each source
            mean1, std1 = results[i]
            mean2, std2 = results[i + j]
            se1 = std1 / np.sqrt(sample_size)
            se2 = std2 / np.sqrt(sample_size)
            t_stat, p_val = ttest_ind_from_stats(mean1, se1, sample_size, mean2, se2, sample_size, equal_var=False)
            t_stats.append(t_stat)
            p_values.append(p_val)

    return t_stats, p_values

# Run t-tests for NBA
nba_t_stats, nba_p_values = t_tests(nba_results)

# Print the results for NBA
print("NBA t-stats:", nba_t_stats)
print("NBA p-values:", nba_p_values)

def t_tests(results):
    t_stats = []
    p_values = []

    # Perform t-tests for each metric within the dataset
    for i in range(0, len(results), 3):  # Loop through each metric for all sources
        for j in range(1,3):  # Loop through each source
            mean1, std1 = results[i]
            mean2, std2 = results[i + j]
            se1 = std1 / np.sqrt(sample_size)
            se2 = std2 / np.sqrt(sample_size)
            t_stat, p_val = ttest_ind_from_stats(mean1, se1, sample_size, mean2, se2, sample_size, equal_var=False)
            t_stats.append(t_stat)
            p_values.append(p_val)

    return t_stats, p_values
# Run t-tests for NBA
pz_t_stats, pz_p_values = t_tests(pokecz_results)

# Print the results for NBA
print("PZ t-stats:",pz_t_stats)
print("PZ p-values:", pz_p_values)

# Run t-tests for NBA
pn_t_stats, pn_p_values = t_tests( pokecn_results)

# Print the results for NBA
print("PN t-stats:",pn_t_stats)
print("PN p-values:", pn_p_values)

