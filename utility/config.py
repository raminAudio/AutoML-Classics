'''
Author: Ramin Anushiravani
Date: April 11th/23
Model configurations
'''


def config(m_name):
    '''
    Args:
        m_name: string, name of the model for which you want to get the list of hyperparameters for

    Returns:
        params: a list of dictionaries containing m_name hyperparameters

    More:
        a few selected hyperparameters for different models
        Helpful for figuring our which model might be its best version by evaluating it on cross validation set
    '''

    if m_name == "logistic regression":
        
        params = []
        params.append({"Model": m_name,
                       'Feature': 'raw',
                       'C': 0.9,
                       'Penalty': 'l1',
                       'num_feat': 0,
                       'class_weight': {0: 0.6,
                                        1: 0.4},
                       'solver': 'liblinear'})
        params.append({"Model": m_name,
                       'Feature': 'eng',
                       'C': 0.9,
                       'Penalty': 'l1',
                       'num_feat': 0,
                       'class_weight': {0: 0.6,
                                        1: 0.4},
                       'solver': 'liblinear'})
        params.append({"Model": m_name,
                       'Feature': 'eng',
                       'C': 0.9,
                       'Penalty': 'l2',
                       'num_feat': 0,
                       'class_weight': {0: 0.6,
                                        1: 0.4},
                       'solver': 'liblinear'})
        params.append({"Model": m_name,
                       'Feature': 'raw',
                       'C': 0.9,
                       'Penalty': 'l1',
                       'num_feat': 0,
                       'class_weight': {0: 0.5,
                                        1: 0.5},
                       'solver': 'liblinear'})
        params.append({"Model": m_name,
                       'Feature': 'raw',
                       'C': 0.9,
                       'Penalty': 'l2',
                       'num_feat': 0,
                       'class_weight': {0: 0.5,
                                        1: 0.5},
                       'solver': 'lbfgs'})
        params.append({"Model": m_name,
                       'Feature': 'pca',
                       'C': 0.9,
                       'num_feat': 128,
                       'Penalty': 'l1',
                       'class_weight': {0: 0.5,
                                        1: 0.5},
                       'solver': 'liblinear'})
        params.append({"Model": m_name,
                       'Feature': 'pca',
                       'C': 0.9,
                       'num_feat': 128,
                       'Penalty': 'l1',
                       'class_weight': {0: 0.6,
                                        1: 0.4},
                       'solver': 'liblinear'})
        params.append({"Model": m_name,
                       'Feature': 'pca',
                       'C': 0.8,
                       'num_feat': 128,
                       'Penalty': 'l1',
                       'class_weight': {0: 0.5,
                                        1: 0.5},
                       'solver': 'liblinear'})
        params.append({"Model": m_name,
                       'Feature': 'pca',
                       'C': 0.9,
                       'num_feat': 256,
                       'Penalty': 'l2',
                       'class_weight': {0: 0.5,
                                        1: 0.5},
                       'solver': 'lbfgs'})
        params.append({"Model": m_name,
                       'Feature': 'pca',
                       'C': 0.9,
                       'num_feat': 64,
                       'Penalty': 'l2',
                       'class_weight': {0: 0.5,
                                        1: 0.5},
                       'solver': 'lbfgs'})
        params.append({"Model": m_name,
                       'Feature': 'pca',
                       'C': 0.9,
                       'num_feat': 32,
                       'Penalty': 'l2',
                       'class_weight': {0: 0.5,
                                        1: 0.5},
                       'solver': 'lbfgs'})


    if m_name == 'SVM':
        
        params = []
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 32, 'C': 0.8})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 64, 'C': 0.8})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 64, 'C': 0.6})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 128, 'C': 0.6})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 128, 'C': 0.6})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 256, 'C': 0.6})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 256, 'C': 0.6})
        params.append({"Model": m_name, 'Feature': 'raw',
                      'num_feat': 0, 'C': 0.8})
        params.append({"Model": m_name, 'Feature': 'eng',
                      'num_feat': 0, 'C': 0.8})
        params.append({"Model": m_name, 'Feature': 'eng',
                      'num_feat': 0, 'C': 0.9})

    if m_name == "RF":
        
        params = []
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 32, 'depth': 9})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 64, 'depth': 9})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 64, 'depth': 7})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 128, 'depth': 7})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 128, 'depth': 7})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 256, 'depth': 7})
        params.append({"Model": m_name, 'Feature': 'pca',
                      'num_feat': 256, 'depth': 7})
        params.append({"Model": m_name, 'Feature': 'raw',
                      'num_feat': 0, 'depth': 9})
        params.append({"Model": m_name, 'Feature': 'raw',
                      'num_feat': 0, 'depth': 15})
        params.append({"Model": m_name, 'Feature': 'eng',
                      'num_feat': 0, 'depth': 15})
        params.append({"Model": m_name, 'Feature': 'eng',
                      'num_feat': 0, 'depth': 21})

    if m_name == 'Perceptron':
        
        params = []

        params.append({"Model": m_name,
                       'Feature': 'pca',
                       'num_feat': 128,
                       'hidden_layer': (1024),
                       'alpha': 0.2,
                       'lr': 0.05,
                       'activation': 'relu'})

        params.append({"Model": m_name,
                       'Feature': 'pca',
                       'num_feat': 64,
                       'hidden_layer': (512),
                       'alpha': 0.2,
                       'lr': 0.05,
                       'activation': 'relu'})

        params.append({"Model": m_name,
                       'Feature': 'pca',
                       'num_feat': 128,
                       'hidden_layer': (512),
                       'alpha': 0.2,
                       'lr': 0.05,
                       'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 128, 'hidden_layer': (
            512, 128), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 32, 'hidden_layer': (
            512, 128), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64, 'hidden_layer': (
            512, 128), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 256, 'hidden_layer': (
            256, 128, 4), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64, 'hidden_layer': (
            256, 128, 4), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 128, 'hidden_layer': (
            64, 32, 8), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64, 'hidden_layer': (
            64, 32, 8), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

        params.append({"Model": m_name,
                       'Feature': 'raw',
                       'num_feat': 0,
                       'hidden_layer': (1024),
                       'alpha': 0.2,
                       'lr': 0.05,
                       'activation': 'relu'})

        params.append({"Model": m_name,
                       'Feature': 'raw',
                       'num_feat': 0,
                       'hidden_layer': (512),
                       'alpha': 0.2,
                       'lr': 0.05,
                       'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'raw', 'num_feat': 0, 'hidden_layer': (
            512, 128), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'raw', 'num_feat': 0, 'hidden_layer': (
            256, 128, 4), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'raw', 'num_feat': 0, 'hidden_layer': (
            64, 32, 8), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

        params.append({"Model": m_name, 'Feature': 'raw', 'num_feat': 0, 'hidden_layer': (
            64, 32, 8), 'alpha': 0.2, 'lr': 0.05, 'activation': 'tanh'})

        params.append({"Model": m_name, 'Feature': 'eng', 'num_feat': 0, 'hidden_layer': (
            64, 32, 8), 'alpha': 0.2, 'lr': 0.05, 'activation': 'relu'})

    if m_name == 'FFNN':
        
        params = []
        params.append({"Model": m_name, 'Feature': 'raw', 'num_feat': 0,
                      'model_num': 1, 'lr': 0.0001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'raw', 'num_feat': 0,
                      'model_num': 2, 'lr': 0.0001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'raw', 'num_feat': 0,
                      'model_num': 3, 'lr': 0.0001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64,
                      'model_num': 1, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 128,
                      'model_num': 1, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 128,
                      'model_num': 2, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 256,
                      'model_num': 2, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64,
                      'model_num': 2, 'lr': 0.0005, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64,
                      'model_num': 2, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 128,
                      'model_num': 2, 'lr': 0.0005, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 256,
                      'model_num': 3, 'lr': 0.0005, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 256,
                      'model_num': 3, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 128,
                      'model_num': 3, 'lr': 0.0005, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64,
                      'model_num': 4, 'lr': 0.0001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 128,
                      'model_num': 4, 'lr': 0.0001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 256,
                      'model_num': 4, 'lr': 0.0001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 256,
                      'model_num': 4, 'lr': 0.0001, 'batch_size': 64})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 256,
                      'model_num': 4, 'lr': 0.001, 'batch_size': 64})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 32,
                      'model_num': 1, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 32,
                      'model_num': 2, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 32,
                      'model_num': 3, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 32,
                      'model_num': 4, 'lr': 0.001, 'batch_size': 64})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64,
                      'model_num': 5, 'lr': 0.001, 'batch_size': 64})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 128,
                      'model_num': 5, 'lr': 0.001, 'batch_size': 64})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64,
                      'model_num': 5, 'lr': 0.001, 'batch_size': 128})
        params.append({"Model": m_name, 'Feature': 'raw', 'num_feat': 64,
                      'model_num': 2, 'lr': 0.001, 'batch_size': 256})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64,
                      'model_num': 5, 'lr': 0.001, 'batch_size': 256})
        params.append({"Model": m_name, 'Feature': 'pca', 'num_feat': 64,
                      'model_num': 5, 'lr': 0.0005, 'batch_size': 80})
        params.append({"Model": m_name, 'Feature': 'eng', 'num_feat': 32,
                      'model_num': 1, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'eng', 'num_feat': 32,
                      'model_num': 2, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'eng', 'num_feat': 32,
                      'model_num': 3, 'lr': 0.001, 'batch_size': 32})
        params.append({"Model": m_name, 'Feature': 'eng', 'num_feat': 32,
                      'model_num': 4, 'lr': 0.001, 'batch_size': 64})
        params.append({"Model": m_name, 'Feature': 'eng', 'num_feat': 64,
                      'model_num': 5, 'lr': 0.001, 'batch_size': 64})

    return params
