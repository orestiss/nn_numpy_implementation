from nn_model.nn_numpy import *

if __name__ == '__main__':
    from sklearn.metrics import log_loss, roc_auc_score
    from keras.datasets import mnist

    (train_x_orig, train_y), (test_x_orig, test_y) = mnist.load_data()

    # keep only one of the classes
    train_y = (train_y == 1).astype(int).T
    test_y = (test_y == 1).astype(int).T

    chosen_class_frequency = train_y[train_y == 1].shape[0]/train_y.shape[0]

    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    layers_dims = [784, 100, 20, 1] #  4-layer model

    model = NeuralNet(layers_dims, learning_rate=0.15, num_iterations = 700, print_cost = True)
    parameters = model.fit(train_x, train_y)

    pred_train = model.predict(train_x)

    pred_test = model.predict(test_x)

    print("base roc auc score: ", roc_auc_score(test_y, [chosen_class_frequency for _ in range(test_y.shape[0])]))
    print("base log loss: ", log_loss(test_y, [chosen_class_frequency for _ in range(test_y.shape[0])]))

    print("train roc auc score: ", roc_auc_score(train_y, pred_train.T))
    print("train log loss: ", log_loss(train_y, pred_train.T))

    print("roc auc score: ", roc_auc_score(test_y, pred_test.T))
    print("log loss: ", log_loss(test_y, pred_test.T))


