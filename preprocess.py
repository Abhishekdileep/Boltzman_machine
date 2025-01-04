import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import kagglehub 
# Download latest version
# path = kagglehub.dataset_download("odedgolden/movielens-1m-dataset")
# print("Path to dataset files:", path)
# path = kagglehub.dataset_download("prajitdatta/movielens-100k-dataset")
# print("Path to dataset files:", path)


def train_test_split():
    users = pd.read_csv('ml-1m/users.dat' , sep='::' , header=None , engine= 'python' , encoding='latin-1')
    movies = pd.read_csv('ml-1m/movies.dat' , sep='::' , header=None , engine= 'python' , encoding='latin-1')
    ratings = pd.read_csv('ml-1m/ratings.dat' , sep='::' , header=None , engine= 'python' , encoding='latin-1')

    train_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
    train_set = np.array(train_set , dtype = int)
    test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
    test_set = np.array(test_set , dtype = int)

    nb_users = int(max(max(train_set[:,0]), max(test_set[:,0])))
    nb_movies = int(max(max(train_set[:,1]), max(test_set[:,1])))

    def convert(data):
        new_data = []
        for id_users in range(1, nb_users + 1):
            id_movies = data[:,1][data[:,0] == id_users]
            id_ratings = data[:,2][data[:,0] == id_users]
            ratings = np.zeros(nb_movies)
            ratings[id_movies - 1] = id_ratings
            new_data.append(list(ratings))
        return new_data
    
    train_set = convert(train_set)
    test_set = convert(test_set)

    train_set = torch.FloatTensor(train_set)
    test_set = torch.FloatTensor(test_set)

    train_set[train_set == 0 ] = -1 
    train_set[train_set == 1 ] = 0 
    train_set[train_set == 2 ] = 0 
    train_set[train_set >= 3 ] = 1 

    test_set[test_set == 0 ] = -1 
    test_set[test_set == 1 ] = 0 
    test_set[test_set == 2 ] = 0 
    test_set[test_set >= 3 ] = 1 


    return train_set , test_set , nb_users , nb_movies
    