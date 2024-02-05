from dig.fairgraph.method.Graphair.aug_module import *
from dig.fairgraph.method.Graphair.classifier import Classifier
from dig.fairgraph.dataset import POKEC, NBA, CNG,RCNG
from dig.fairgraph.method.Graphair.graphair import graphair
from dig.fairgraph.method.Graphair.GCN import GCN, GCN_Body
import torch
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Graphair model with HPO")
    parser.add_argument('--dataset', type=str, default='NBA', choices=['NBA', 'POKEC', 'CNG', 'RCNG'],
                        help='Dataset to use for training and evaluation.')
    parser.add_argument('--sens_att', type=str, default=None,
                        help='For Congress which attribute to focus on')
    parser.add_argument('--fm', type=bool, default=True,
                        help='Do feature masking')
    parser.add_argument('--ep', type=bool, default=True,
                        help='Do edge perturbation')
    parser.add_argument('--with_fair', type=bool, default=True,
                        help='Optimize with fair metrics')
    args = parser.parse_args()
    return args


def log_gpu_usage():
    if torch.cuda.is_available():
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory}")
        print(f"Used GPU Memory: {torch.cuda.memory_allocated()}")
        print(f"Free GPU Memory: {torch.cuda.memory_reserved()}")
    else:
        print("CUDA not available")


class run():
    r"""
    This class instantiates Graphair model and implements method to train and evaluate.
    """

    def __init__(self):
        pass

    def run(self, device, dataset, model='Graphair', epochs=10_000, test_epochs=1_000,
            lr=1e-4, weight_decay=1e-5, fm = True, ep=True, with_fair=True):
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

        if dataset.name == "CNG":
            # : {'alpha': 2.6, 'gamma': 3.1, 'lambda': 1.1}
            alpha = 2.6
            lama = 1.1 #change with hpo results
            gamma = 3.1
        elif dataset.name == "NBA":
            #alpha': 10.1, 'gamma': 5.1, 'lambda': 4.6
            alpha = 10
            lama = 5
            gamma = 5
        else:
            alpha = 10
            lama = 0.5
            gamma = 0.5

        # generate model
        if model == 'Graphair':
            aug_model = aug_module(features, n_hidden=64, temperature=1,  FM=fm, EP=ep).to(device)
            f_encoder = GCN_Body(in_feats=features.shape[1], n_hidden=64, out_feats=64, dropout=0.1, nlayer=2).to(
                device)
            sens_model = GCN(in_feats=features.shape[1], n_hidden=64, out_feats=64, nclass=1).to(device)
            classifier_model = Classifier(input_dim=64, hidden_dim=128)
            model = graphair(aug_model=aug_model, f_encoder=f_encoder, sens_model=sens_model,
                             classifier_model=classifier_model, lr=lr, weight_decay=weight_decay,
                             dataset=dataset_name, alpha=alpha, beta=1, gamma=gamma, lam=lama).to(device)
        else:
            raise Exception('At this moment, only Graphair is supported!')


        # call fit_whole
        st_time = time.time()
        model.fit_whole(epochs=epochs, adj=adj, x=features, sens=sens, idx_sens=idx_sens, warmup=50, adv_epoches=1, with_fair=with_fair)
        print("Training time: ", time.time() - st_time)
        st_time = time.time()


        # Test script
        model.test(adj=adj, features=features, labels=dataset.labels, epochs=test_epochs, idx_train=dataset.idx_train,
                   idx_val=dataset.idx_val, idx_test=dataset.idx_test, sens=sens)
        log_gpu_usage()


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'NBA':
        dataset = NBA()
    elif args.dataset == 'RCNG':
        dataset = RCNG()
    elif args.dataset == 'POKEC':
        dataset = POKEC()
    elif args.dataset == 'CNG':
        if args.sens_att:
            dataset = CNG(sens_attr=args.sens_att)
        else:
            dataset = CNG()
    else:
        raise ValueError("Unsupported dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_fairgraph = run()
    run_fairgraph.run(device, dataset=dataset, model='Graphair', epochs=1000, test_epochs=500,
                      lr=1e-3, weight_decay=1e-5, fm = args.fm, ep=args.ep, with_fair=args.with_fair)
