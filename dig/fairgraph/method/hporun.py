from dig.fairgraph.method.Graphair import graphair, aug_module, GCN, GCN_Body, Classifier
from dig.fairgraph.dataset import POKEC, NBA, Congress
import torch
import time
import numpy as np
import optuna
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Graphair model with HPO")
    parser.add_argument('--dataset', type=str, default='NBA', choices=['NBA', 'POKEC', 'Congress'],
                        help='Dataset to use for training and evaluation.')
    args = parser.parse_args()
    return args


def hpo(trial, dataset_name, run_fair):
    # Load the dataset based on command line argument
    if dataset_name == 'NBA':
        dataset = NBA()
    elif dataset_name == 'POKEC':
        dataset = POKEC()
    elif dataset_name == 'Congress':
        dataset = Congress()
    else:
        raise ValueError("Unsupported dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = trial.suggest_float('alpha', 0.1, 10, step=0.5)
    gamma = trial.suggest_float('gamma', 0.1, 10, step=0.5)
    lam = trial.suggest_float('lambda', 0.1, 10, step=0.5)
    acc = run_fair.run(alpha, gamma, lam, device, dataset=dataset, epochs=100, test_epochs=100,
              lr=1e-4, weight_decay=1e-5)
    return acc


class run():
    r"""
    This class instantiates Graphair model and implements method to train and evaluate.
    """

    def __init__(self):
        pass

    def run(alpha, gamma, lam, device, dataset, epochs=10_000, test_epochs=1_000,
            lr=1e-4, weight_decay=1e-5):
        r""" This method runs training and evaluation for a fairgraph model on the given dataset.
            Check :obj:`examples.fairgraph.Graphair.run_graphair_nba.py` for examples on how to run the Graphair model.


            :param device: Device for computation.
            :type device: :obj:`torch.device`

            :param model: Defaults to `Graphair`. (Note that at this moment, only `Graphair` is supported)
            :type model: str, optional

            :param dataset: The dataset to train on. Should be one of :obj:`dig.fairgraph.dataset.fairgraph_dataset.POKEC` or :obj:`dig.fairgraph.dataset.fairgraph_dataset.NBA`.
            :type dataset: :obj:`object`

            :param epochs: Number of epochs to train on. Defaults to 10_000.
            :type epochs: int, optional

            :param test_epochs: Number of epochs to train the classifier while running evaluation. Defaults to 1_000.
            :type test_epochs: int,optional

            :param lr: Learning rate. Defaults to 1e-4.
            :type lr: float,optional

            :param weight_decay: Weight decay factor for regularization. Defaults to 1e-5.
            :type weight_decay: float, optional

            :raise:
                :obj:`Exception` when model is not Graphair. At this moment, only Graphair is supported.
            """

        # Train script

        dataset_name = dataset.name

        features = dataset.features
        sens = dataset.sens
        adj = dataset.adj
        idx_sens = dataset.idx_sens_train

        print("Doing Hyperparameter Search")

        print("Test for alpha, gamma, lam as", alpha, gamma, lam)
        aug_model = aug_module(features, n_hidden=64, temperature=1).to(device)
        f_encoder = GCN_Body(in_feats=features.shape[1], n_hidden=64, out_feats=64, dropout=0.1,
                             nlayer=3).to(
            device)
        sens_model = GCN(in_feats=features.shape[1], n_hidden=64, out_feats=64, nclass=1).to(device)
        classifier_model = Classifier(input_dim=64, hidden_dim=128)
        model = graphair(aug_model=aug_model, f_encoder=f_encoder, sens_model=sens_model,
                         classifier_model=classifier_model, lr=lr, weight_decay=weight_decay,
                         alpha=alpha, gamma=gamma, lam=lam,
                         dataset=dataset_name).to(device)

        # call fit_whole
        st_time = time.time()
        model.fit_whole(epochs=epochs, adj=adj, x=features, sens=sens, idx_sens=idx_sens, warmup=0,
                        adv_epoches=1)
        print("Training time: ", time.time() - st_time)

        # Test script
        acc = model.test(adj=adj, features=features, labels=dataset.labels, epochs=test_epochs,
                         idx_train=dataset.idx_train,
                         idx_val=dataset.idx_val, idx_test=dataset.idx_test, sens=sens)
        print(f'alpha = {alpha}, gamma = {gamma}, lambda = {lam}')
        return acc


if __name__ == '__main__':
    args = parse_args()


    def objective(trial):
        run_fair = run()
        return hpo(trial, args.dataset, run_fair)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=250)

    # After optimization, save the study object
    with open(f'{args.dataset.lower()}_hpo_study.pkl', 'wb') as f:
        pickle.dump(study, f)
