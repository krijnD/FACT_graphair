import torch
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import random
from torch_geometric.data import download_url
import ast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class POKEC():
    r"""Pokec is a social network dataset. Two `different datasets <https://github.com/EnyanDai/FairGNN/tree/main/dataset/pokec>`_ (namely pokec_z and pokec_n) are sampled
        from the original `Pokec dataset <https://snap.stanford.edu/data/soc-pokec.html>`_.

        :param data_path: The url where the dataset is found, defaults to :obj:`https://github.com/divelab/DIG_storage/raw/main/fairgraph/datasets/pockec/`
        :type data_path: str, optional

        :param root: The path to root directory where the dataset is saved, defaults to :obj:`./dataset/pokec`
        :type root: str, optional

        :param dataset_sample: The sample (should be one of `pokec_z` or `pokec_n`) to be used in choosing the POKEC dataset. Defaults to `pokec_z`
        :type dataset_sample: str, optional
        
        :raises: :obj:`Exception`
            When invalid dataset_sample is provided.
    """
    def __init__(self, 
                data_path='https://github.com/divelab/DIG_storage/raw/main/fairgraph/datasets/pockec/',
                root='./dataset/pokec',
                dataset_sample='pokec_z'):
        self.name = "POKEC_Z"
        self.root = root
        self.dataset_sample = dataset_sample
        if self.dataset_sample=='pokec_z':
            self.dataset = 'region_job'
        elif self.dataset_sample=='pokec_n':
            self.dataset = 'region_job_2'
        else:
            raise Exception('Invalid dataset sample! Should be one of pokec_z or pokec_n')
        self.sens_attr = "region"
        self.predict_attr = "I_am_working_in_field"
        self.label_number = 50000
        self.sens_number = 20000
        self.seed = 20
        self.test_idx=False
        self.data_path = data_path
        self.process()
    
    @property
    def raw_paths(self):
        return [f"{self.dataset}.csv",f"{self.dataset}_relationship.txt",f"{self.dataset}.embedding"]
    
    def download(self):
        print('downloading raw files from:', self.data_path)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        for raw_path in self.raw_paths:
            download_url(self.data_path+raw_path,self.root)

    def read_graph(self):
        self.download()
        print(f'Loading {self.dataset} dataset from {os.path.abspath(self.root+"/"+self.raw_paths[0])}')
        # raw_paths[0] will be region_job.csv
        idx_features_labels = pd.read_csv(os.path.abspath(self.root+"/"+self.raw_paths[0]))
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(self.sens_attr)
        header.remove(self.predict_attr)


        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[self.predict_attr].values
        

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        # raw_paths[1] will be region_relationship.txt
        edges_unordered = np.genfromtxt(os.path.abspath(self.root+"/"+self.raw_paths[1]), dtype=int)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)

        
        random.seed(self.seed)
        label_idx = np.where(labels>=0)[0]
        random.shuffle(label_idx)
        idx_train = label_idx[:min(int(0.1 * len(label_idx)),self.label_number)]
        idx_val = label_idx[int(0.1 * len(label_idx)):int(0.2 * len(label_idx))]
        if self.test_idx:
            idx_test = label_idx[self.label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.2 * len(label_idx)):]

        sens = idx_features_labels[self.sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = torch.LongTensor(list(sens_idx))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train


    def feature_norm(self,features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]

        return 2*(features - min_values).div(max_values-min_values) - 1

    def process(self):
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = self.read_graph()
        features = self.feature_norm(features)

        labels[labels>1]=1
        sens[sens>0]=1

        self.features = features.cuda()
        self.labels = labels.cuda()
        self.idx_train = idx_train.cuda()
        self.idx_val = idx_val.cuda()
        self.idx_test = idx_test.cuda()
        self.sens = sens.cuda()
        self.idx_sens_train = idx_sens_train.long().cuda()

        self.adj = adj

class NBA():
    r'''
        `NBA <https://github.com/EnyanDai/FairGNN/tree/main/dataset/NBA>`_ is an NBA on court performance dataset along salary, social engagement etc.

        :param data_path: The url where the dataset is found, defaults to :obj:`https://github.com/divelab/DIG_storage/raw/main/fairgraph/datasets/nba/`
        :type data_path: str, optional

        :param root: The path to root directory where the dataset is saved, defaults to :obj:`./dataset/nba`
        :type root: str, optional
    '''
    def __init__(self, 
                data_path='https://github.com/divelab/DIG_storage/raw/main/fairgraph/datasets/nba/',
                root='./dataset/nba'):
        self.name = "NBA"
        self.root = root
        self.dataset = 'nba'
        self.sens_attr = "country"
        self.predict_attr = "SALARY"
        self.label_number = 100
        self.sens_number = 500
        self.seed = 20
        self.test_idx=False
        self.data_path = data_path
        self.process()

    @property
    def raw_paths(self):
        return ["nba.csv","nba_relationship.txt","nba.embedding"]
    
    def download(self):
        print('downloading raw files from:', self.data_path)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        for raw_path in self.raw_paths:
            download_url(self.data_path+raw_path,self.root)

    def read_graph(self):
        self.download()
        print(f'Loading {self.dataset} dataset from {os.path.abspath(self.root+"/"+self.raw_paths[0])}')
        idx_features_labels = pd.read_csv(os.path.abspath(self.root+"/"+self.raw_paths[0]))
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(self.sens_attr)
        header.remove(self.predict_attr)


        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[self.predict_attr].values
        

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        # raw_paths[1] will be nba_relationship.txt
        edges_unordered = np.genfromtxt(os.path.abspath(self.root+"/"+self.raw_paths[1]), dtype=int)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)

        
        random.seed(self.seed)
        label_idx = np.where(labels>=0)[0]
        random.shuffle(label_idx)
        idx_train = label_idx[:min(int(0.2 * len(label_idx)),self.label_number)]
        idx_val = label_idx[int(0.2 * len(label_idx)):int(0.55 * len(label_idx))]
        if self.test_idx:
            idx_test = label_idx[self.label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.55 * len(label_idx)):]

        sens = idx_features_labels[self.sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = torch.LongTensor(list(sens_idx))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train


    def feature_norm(self,features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]

        return 2*(features - min_values).div(max_values-min_values) - 1

    def process(self):
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = self.read_graph()
        features = self.feature_norm(features)

        labels[labels>1]=1
        sens[sens>0]=1

        self.features = features.cuda()
        self.labels = labels.cuda()
        self.idx_train = idx_train.cuda()
        self.idx_val = idx_val.cuda()
        self.idx_test = idx_test.cuda()
        self.sens = sens.cuda()
        self.idx_sens_train = idx_sens_train.long().cuda()

        self.adj = adj


class Congress():
    '''
    Congress dataset class for handling data operations.

    :param data_path: The path where the dataset files are located, defaults to 'benchmark_dataset/'
    :type data_path: str, optional

    :param root: The path to root directory where the dataset is processed and saved, defaults to './dataset/cng'
    :type root: str, optional
    '''

    def __init__(self, data_path='benchmark_dataset/', root='./dataset/cng', sens_attr="gender_feat"):
        self.name = "CNG"
        self.root = root
        self.dataset = 'cng'
        self.sens_attr = sens_attr
        self.predict_attr = "class_net_worth"
        self.label_number = 100
        self.seed = 20
        self.test_idx = False
        self.data_path = data_path
        self.process()

    @property
    def raw_paths(self):
        # Paths to the raw data files relative to `self.root`
        if self.sens_attr == "religion_sensitivity":
            data_csv = "encoded_data_wo_religions.csv"
        else:
            data_csv = "encoded_data.csv"

        data_csv
        return [os.path.join(self.data_path, data_csv),
                os.path.join(self.data_path, "cng_relationship.txt"),
                ]

    def convert_string_to_list(vector_string):
        # Attempt to clean and convert the string representation of a vector into a list
        try:
            # Insert commas between values and remove newline characters
            cleaned_string = vector_string.replace('\n', '').replace(' ', ', ')
            # Ensure the string starts with '[' and ends with ']' to form a valid list format
            if not cleaned_string.startswith('['):
                cleaned_string = '[' + cleaned_string
            if not cleaned_string.endswith(']'):
                cleaned_string = cleaned_string + ']'
            # Safely evaluate the cleaned string as a Python literal
            vector_list = ast.literal_eval(cleaned_string)
            return vector_list
        except Exception as e:
            # Handle exceptions or malformed strings
            print(f"Error converting string to list: {e}")
            return []

    def preprocess_vectors(self, df):
        # Apply the conversion to 'first_name_vector' and 'last_name_vector' columns
        df['first_name_vector'] = df['first_name_vector'].apply(self.convert_string_to_list)
        df['last_name_vector'] = df['last_name_vector'].apply(self.convert_string_to_list)
        return df

    def read_graph(self):
        print(f'Loading {self.dataset} dataset from {os.path.abspath(self.raw_paths[0])}')
        idx_features_labels = pd.read_csv(self.raw_paths[0])

        # Preprocess 'first_name_vector' and 'last_name_vector' columns before any numerical processing
        idx_features_labels = self.preprocess_vectors(idx_features_labels)

        # Continue with the existing operations...
        # Remove columns not part of features
        features = idx_features_labels.drop(
            columns=[self.sens_attr, self.predict_attr, "numeric_id", "first_name_vector", "last_name_vector"])

        # Create feature matrix
        features_matrix = sp.csr_matrix(features, dtype=np.float32)
        labels = idx_features_labels[self.predict_attr].values

        # If your dataset includes relationships, adjust the following section
        edges_unordered = np.genfromtxt(self.raw_paths[1], dtype=int)
        edges = np.array(list(map(lambda x: (idx_features_labels.index[idx_features_labels["numeric_id"] == x[0]].tolist()[0],
                                             idx_features_labels.index[idx_features_labels["numeric_id"] == x[1]].tolist()[0]),
                                  edges_unordered)),
                         dtype=int)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # Build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        # Split dataset
        random.seed(self.seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)
        idx_train = label_idx[:min(int(0.2 * len(label_idx)), self.label_number)]
        idx_val = label_idx[int(0.2 * len(label_idx)):int(0.55 * len(label_idx))]
        if self.test_idx:
            idx_test = label_idx[self.label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.55 * len(label_idx)):]

        # Handle sensitive attributes
        sens = idx_features_labels[self.sens_attr].values
        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = torch.LongTensor(list(sens_idx))

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

    def feature_norm(self, features):
        # Normalize features
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def process(self):
        # Main processing function
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = self.read_graph()
        features = self.feature_norm(features)

        labels[labels > 1] = 1  # Assuming binary classification, adjust if necessary
        sens[sens > 0] = 1  # Assuming binary sensitive attribute, adjust if necessary

        # Move data to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features = features.to(device)
        self.labels = labels.to(device)
        self.idx_train = idx_train.to(device)
        self.idx_val = idx_val.to(device)
        self.idx_test = idx_test.to(device)
        self.sens = sens.to(device)
        self.idx_sens_train = idx_sens_train.to(device).long()

        self.adj = adj.to(device)