DS = 'Pokec-n' #or 'NBA' or 'Pokec-z'

from dig.fairgraph.method.Graphair.aug_module import *
from dig.fairgraph.method.Graphair.classifier import Classifier
from dig.fairgraph.dataset import POKEC, NBA
from dig.fairgraph.method.Graphair.graphair import graphair
from dig.fairgraph.method.Graphair.GCN import GCN, GCN_Body
from dig.fairgraph.utils.utils import scipysp_to_pytorchsp
import torch
import time
import scipy.sparse as sp
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy

def calculate_epsilon(adj_np, sens):
    # useful to vectorize the indicator function:
    indicator_matrix = sens[:, np.newaxis] == sens

    numerator = np.sum(adj_np * indicator_matrix, axis=1)
    denominator = np.sum(adj_np, axis=1)
    epsilon = numerator / denominator

    # handling division by zero:
    epsilon[denominator == 0] = 0

    return epsilon

def plot_eps(eps, eps_aug, lab, key):
    plt.figure(figsize=(6, 4))

    # Compute KDEs without plotting
    density_orig = sns.kdeplot(eps).get_lines()[0].get_data()
    density_fair = sns.kdeplot(eps_aug).get_lines()[1].get_data()

    # Clear the figure
    plt.clf()

    sns.kdeplot(eps, label='Original', color='blue', linestyle='-')
    sns.kdeplot(eps_aug, label='Fair view', color='orange', linestyle='-')

    # Find modes of the distributions
    mode_orig = density_orig[0][np.argmax(density_orig[1])]
    mode_fair = density_fair[0][np.argmax(density_fair[1])]

    # Draw vertical lines at the mode of the distributions
    plt.axvline(x=mode_orig, color='blue', linestyle='--')
    plt.axvline(x=mode_fair, color='orange', linestyle='--')

    plt.xlabel('Node sensitive homophily', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'{lab}, {key}', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14)

    #take care of the ticks
    plt.xlim(left=0)
    standard_ticks = np.linspace(0, plt.xlim()[1], 5)
    combined_ticks = np.unique(np.concatenate((standard_ticks, [mode_orig, mode_fair])))
    rounded_ticks = [round(tick, 2) for tick in combined_ticks]
    plt.xticks(rounded_ticks, [f"{tick:.2f}" for tick in rounded_ticks], rotation=45)

    plt.tight_layout()
    plt.savefig(f'{lab}-{key}.pdf', bbox_inches='tight')
    plt.show()

def plot_corr(top_10_cors, x_aug_old_top_10, x_aug_new_top_10, lab, key):

    labels = [str(i) for i in range(10)]  # Just as placeholder labels for the top-10 features

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, top_10_cors, width, label='Original')
    rects2 = ax.bar(x, x_aug_old_top_10, width, label='Fair view (old top-10)')
    rects3 = ax.bar(x + width, x_aug_new_top_10, width, label='Fair view (new top-10)')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Feature rank or index', fontsize=14)
    ax.set_ylabel('Absolute Spearman correlation', fontsize=14)
    ax.set_title(f'{lab}, {key}', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=14)

    fig.tight_layout()

    plt.savefig(f'{lab}-{key}-corr.pdf', bbox_inches='tight')
    plt.show()


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
            lr=1e-4, weight_decay=1e-5, alpha = 1, beta = 1, gamma = 1, lam = 1, lab = 'None', key='None'):
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
        #log_gpu_usage()

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
            classifier_model = Classifier(input_dim=64, hidden_dim=128)
            model = graphair(aug_model=aug_model, f_encoder=f_encoder, sens_model=sens_model,
                             classifier_model=classifier_model, lr=lr, weight_decay=weight_decay,
                             dataset=dataset_name, alpha=alpha, beta=beta, gamma=gamma, lam = lam).to(device)
        else:
            raise Exception('At this moment, only Graphair is supported!')

        # call fit_whole
        st_time = time.time()
        model.fit_whole(epochs=epochs, adj=adj, x=features, sens=sens, idx_sens=idx_sens, warmup=0, adv_epoches=1)
        print("Training time: ", time.time() - st_time)
        # Log GPU usage after training
        log_gpu_usage()
        adj_old = deepcopy(adj)


        #********REPRODUCE FIGURES*********
        model.eval()

        #do all the preprocessing of the original adjacency matrix A
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj_orig = scipysp_to_pytorchsp(adj).to_dense()
        norm_w = adj_orig.shape[0] ** 2 / float((adj_orig.shape[0] ** 2 - adj_orig.sum()) * 2)
        degrees = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
        adj_norm = scipysp_to_pytorchsp(adj_norm)
        adj = adj_norm.cuda()

        #produce the A', the "fairer" version of adjacency matrix A
        #and the 'fairer' features x_aug
        adj_aug, x_aug, adj_logits = model.aug_model(adj, features, adj_orig=adj_orig.cuda())

        #reformat the variables to enable numpy computations
        adj_np = adj.to_dense().cpu().numpy()
        adj_aug_np = adj_aug.cpu().detach().numpy()
        sens_np = sens.cpu().numpy()

        #create vectors of epsilons
        epsilons_orig = calculate_epsilon(adj_np, sens_np)
        epsilons_aug = calculate_epsilon(adj_aug_np, sens_np)

        #plot the graphs of node sensitive homophily
        plot_eps(epsilons_orig, epsilons_aug, lab, key)

        #compute Spearman correlation for each feature
        correlations = np.abs([spearmanr(sens.cpu().ravel(), features.cpu()[:, i])[0] for i in range(features.shape[1])])
        top_10_indices = np.argsort(correlations)[-10:]
        top_10_cors = correlations[top_10_indices]
        top_10_cors = top_10_cors[::-1]

        #compute Spearman correlation for each feature of the 'fairer' graph
        corr_aug = np.abs([spearmanr(sens.cpu().detach().ravel(), x_aug.cpu().detach()[:, i])[0] for i in range(x_aug.shape[1])])
        x_aug_top_10_ind = np.argsort(corr_aug)[-10:]

        #for comparison:
        x_aug_old_top_10 = corr_aug[top_10_indices] #correlation with sens of the
        x_aug_new_top_10 = corr_aug[x_aug_top_10_ind]
        x_aug_old_top_10 = x_aug_old_top_10[::-1]
        x_aug_new_top_10 = x_aug_new_top_10[::-1]

        plot_corr(top_10_cors, x_aug_old_top_10, x_aug_new_top_10, lab, key)

        # Test script
        #print("Testing Beginning")
        #log_gpu_usage()
        #model.test(adj=adj_old, features=features, labels=dataset.labels, epochs=test_epochs, idx_train=dataset.idx_train,
                #  idx_val=dataset.idx_val, idx_test=dataset.idx_test, sens=sens)
        #log_gpu_usage()



# choose the dataset
if DS =='NBA':
    ds = NBA()
    hyperparams = {'default':[20, 0.9, 0.7, 1],
               'HPO':[1,1,0.1,1],
               'correspondence': [10, 0.1, 0.1, 0.5]}
elif DS =='Pokec-n':
    ds = POKEC(dataset_sample='pockec_n')
elif DS == 'Pokec-z':
    ds = POKEC(dataset_sample='pockec_z') 
else:
    raise NotImplementedError(f'this dataset is not implemented: {DS}')

# Train and evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for key in hyperparams.keys():
    run_fairgraph = run()
    alpha, beta, gamma, lam = hyperparams[key][0], hyperparams[key][1],hyperparams[key][2],hyperparams[key][3]
    run_fairgraph.run(device,dataset=ds,model='Graphair',epochs=500,test_epochs=500,
                lr=1e-4,weight_decay=1e-5, alpha = alpha, beta = beta, gamma = gamma, lam = lam, lab = DS, key = key)
