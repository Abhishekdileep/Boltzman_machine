from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from load_mnist import * 
from rbm import *
import torch 
import ipdb 


def plot_example(X, y):
    """Plot the first 5 images and their labels in a row."""
    for i, (img, y) in enumerate(zip(X[:5], y[:5])):
        plt.subplot(151 + i)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title(y)
    plt.show()

x_train , x_test , y_train , y_test = load_data()


nv = 784
nh = 500
batch_size = 64
rbm = RBM(nv, nh)

dataset_size = len(x_train) 
test_size = len(x_test)

nb_epoch = 20
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, dataset_size - batch_size, batch_size):
        vk = torch.tensor(np.array([np.array(i).reshape(-1,) for i in x_train[id_user:id_user+batch_size]], dtype=np.float32))
        v0 = torch.tensor(np.array([np.array(i).reshape(-1,) for i in x_train[id_user:id_user+batch_size]], dtype=np.float32))
        # ipdb.set_trace()
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            # vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0 - vk))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))


# test_loss = 0
# s = 0.
# for id_user in range(dataset_size):
#     v = train_set[id_user:id_user+1]
#     vt = test_set[id_user:id_user+1]
#     if len(vt[vt>=0]) > 0:
#         _,h = rbm.sample_h(v)
#         _,v = rbm.sample_v(h)
#         test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
#         s += 1.
# print('test loss: '+str(test_loss/s))

# Download latest versio# plot_example(X, Y)