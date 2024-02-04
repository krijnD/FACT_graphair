import optuna
import pickle




# Path to your .pkl file
pkl_file_path = '/Users/bellavg/PycharmProjects/DIG_FACT/dig/fairgraph/method/cng_hpo_study.pkl'

# Load the study
with open(pkl_file_path, 'rb') as f:
    study = pickle.load(f)

from optuna.visualization import plot_optimization_history, plot_rank, plot_parallel_coordinate

fig = plot_optimization_history(study)
fig.show()
fig = plot_rank(study)
fig.show()
fig = plot_parallel_coordinate(study)
fig.show()