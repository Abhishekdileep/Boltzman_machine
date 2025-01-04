# Restricted Boltzmann Machine (RBM)

## Overview
A Restricted Boltzmann Machine (RBM) is a type of artificial neural network that is used for unsupervised learning. It is a generative stochastic neural network that can learn a probability distribution over its set of inputs. RBMs are useful for dimensionality reduction, classification, regression, collaborative filtering, feature learning, and topic modeling.

## Applications

### Movie Review Dataset
RBMs can be used to analyze and predict user preferences in a movie review dataset. By learning the underlying patterns in the data, an RBM can recommend movies to users based on their past reviews and ratings.

#### Steps:
1. **Data Preprocessing**: Clean and preprocess the movie review dataset.
2. **Training the RBM**: Train the RBM on the preprocessed dataset to learn the hidden features.
3. **Making Predictions**: Use the trained RBM to predict user preferences and recommend movies.

### MNIST Dataset
The MNIST dataset consists of handwritten digit images and is commonly used for training image processing systems. RBMs can be used to learn the features of the images and perform tasks such as digit classification.

#### Steps:
1. **Data Preprocessing**: Normalize and preprocess the MNIST dataset.
2. **Training the RBM**: Train the RBM on the MNIST dataset to learn the hidden features of the digits.
3. **Digit Classification**: Use the trained RBM to classify handwritten digits.


## Conclusion
RBMs are powerful tools for unsupervised learning and can be applied to various datasets, including movie reviews and handwritten digits. By learning the hidden features in the data, RBMs can provide valuable insights and predictions.

## References
- [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)