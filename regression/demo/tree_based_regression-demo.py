import numpy as np
from regression.function import *


def bin_split_dataset(dataset, feature, value):
    mat0 = dataset[nonzero(dataset[:, feature] > value)[0], :]
    mat1 = dataset[nonzero(dataset[:, feature] <= value)[0], :]
    return mat0, mat1


def reg_leaf(dataset):
    return mean(dataset[:, 0])     # mean value of labels


def reg_err(dataset):
    return var(dataset[:, 0]) * shape(dataset)[0]      # total variance of labels


def choose_best_split(dataset, ops, leaf_type=reg_leaf, err_type=reg_err):
    tol_s = ops[0]       # tol_s is a tolerance on the error reduction
    tol_n = ops[1]       # tol_n is the minimum data instances to include in a split
    m, n = shape(dataset)
    s = err_type(dataset)
    best_s = inf
    best_index = 0
    best_value = 0
    for feat_index in range(1, n):
        for split_val in set(dataset[:, feat_index]):
            mat0, mat1 = bin_split_dataset(dataset, feat_index, split_val)
            if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n):
                continue
            new_s = err_type(mat0) + err_type(mat1)
            if new_s < best_s:
                best_index = feat_index
                best_value = split_val
                best_s = new_s
    if (s - best_s) <= tol_s:        # exit if low error reduction
        return None, leaf_type(dataset)
    mat0, mat1 = bin_split_dataset(dataset, best_index, best_value)
    if (shape(mat0)[0] <= tol_n) or (shape(mat1)[0] <= tol_n):        # exit if split creates small dataset
        return None, leaf_type(dataset)
    return best_index, best_value


def create_tree(dataset, ops, leaf_type=reg_leaf, err_type=reg_err):
    feat, val = choose_best_split(dataset, ops, leaf_type, err_type)
    if feat == None:
        return val
    ret_tree = {}
    ret_tree['sp_index'] = feat
    ret_tree['sp_value'] = val
    l_set, r_set = bin_split_dataset(dataset, feat, val)
    ret_tree['left'] = create_tree(l_set, ops, leaf_type, err_type)
    ret_tree['right'] = create_tree(r_set, ops, leaf_type, err_type)
    return ret_tree


def isTree(obj):
    return (type(obj).__name__=='dict')


def reg_tree_eval(model):
    return float(model)


def tree_forest_cast(tree, data, model_eval=reg_tree_eval):
    if not isTree(tree):
        return model_eval(tree)
    if data[tree['sp_index']] > tree['sp_value']:
        if isTree(tree['left']):
            return tree_forest_cast(tree['left'], data, model_eval)
        else:
            return model_eval(tree['left'])
    else:
        if isTree(tree['right']):
            return tree_forest_cast(tree['right'], data, model_eval)
        else:
            return model_eval(tree['right'])


def create_fore_cast(tree, test_data, model_eval=reg_tree_eval):
    m = len(test_data)
    pred = zeros((m, 1))
    for i in range(m):
        pred[i, 0] = tree_forest_cast(tree, test_data[i], model_eval)
    return pred


def corr_coef(pred, labels):
    m = len(labels)
    for i in range(m):
        print('Prediction is %f, and actual value is %f' %(pred[i],labels[i]))


path = '../data/sample_linear_regression_data.txt'
labels, features = file2matrix(path)
hoRatio = 0.20
m, n = shape(features)
test_feats = features[m-int(hoRatio*m):m]
train_feats = features[0:m-int(hoRatio*m)]
test_labels = labels[m-int(hoRatio*m):m]
train_labels = labels[0:m-int(hoRatio*m)]
train_data = np.append(train_labels, train_feats, axis=1)
test_data = np.append(test_labels, test_feats, axis=1)

tree = create_tree(train_data, ops=(1, 5))
pred = create_fore_cast(tree, test_data)
corr_coef(pred, test_labels)
