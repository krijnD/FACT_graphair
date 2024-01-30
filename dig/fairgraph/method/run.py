from dig.fairgraph.method.Graphair.aug_module import *
from dig.fairgraph.method.Graphair.classifier import Classifier
from dig.fairgraph.dataset import POKEC, NBA, Congress
from dig.fairgraph.method.Graphair.graphair import graphair
from dig.fairgraph.method.Graphair.GCN import GCN, GCN_Body
import torch
import time

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
        # Log GPU usage before training
        log_gpu_usage()

        dataset_name = dataset.name

        features = dataset.features
        sens = dataset.sens
        adj = dataset.adj
        idx_sens = dataset.idx_sens_train

        # generate model
        if model == 'Graphair':
            aug_model = aug_module(features, n_hidden=64, temperature=1).to(device)
            f_encoder = GCN_Body(in_feats=features.shape[1], n_hidden=64, out_feats=64, dropout=0.1, nlayer=3).to(
                device)
            sens_model = GCN(in_feats=features.shape[1], n_hidden=64, out_feats=64, nclass=1).to(device)
            print("with hidden dim 128")
            classifier_model = Classifier(input_dim=64, hidden_dim=128)
            model = graphair(aug_model=aug_model, f_encoder=f_encoder, sens_model=sens_model,
                             classifier_model=classifier_model, lr=lr, weight_decay=weight_decay,
                             dataset=dataset_name).to(device)
        else:
            raise Exception('At this moment, only Graphair is supported!')

        # call fit_whole
        st_time = time.time()
        model.fit_whole(epochs=epochs, adj=adj, x=features, sens=sens, idx_sens=idx_sens, warmup=0, adv_epoches=1)
        print("Training time: ", time.time() - st_time)
        # Log GPU usage after training
        log_gpu_usage()

        # Test script
        print("Testing Beginning")
        log_gpu_usage()
        model.test(adj=adj, features=features, labels=dataset.labels, epochs=test_epochs, idx_train=dataset.idx_train,
                   idx_val=dataset.idx_val, idx_test=dataset.idx_test, sens=sens)
        log_gpu_usage()



# Load the dataset
nba = NBA()

# Train and evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_fairgraph = run()
run_fairgraph.run(device,dataset=nba,model='Graphair',epochs=500,test_epochs=500,
            lr=1e-3,weight_decay=1e-5)
