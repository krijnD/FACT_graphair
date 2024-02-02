import optuna
import pickle
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour,plot_parallel_coordinate




# Load the study
with open("/Users/bellavg/PycharmProjects/DIG_FACT/nba_hpo_study.pkl", 'rb') as f:
    study = pickle.load(f)




# Best trial
best_trial = study.best_trial
print(f"Best trial value: {best_trial.value}")
print(f"Best trial parameters: {best_trial.params}")

# Number of trials
print(f"Total trials: {len(study.trials)}")

# Assuming `study` is your Optuna study object


# Parameter Importance
plt = plot_param_importances(study)
plt.show()

