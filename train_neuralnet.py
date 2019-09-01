# coding: utf-8
import matplotlib.pyplot as plt
# from two_layer_net import TwoLayerNet
from multi_layer_net import MultiLayerNet
from cifar10 import load_cifar10
from optimizer import *

(x_train, t_train), (x_test, t_test) = load_cifar10(
    normalize=True, flatten=True, one_hot_label=True, data_batch_number='1')

# network = TwoLayerNet(input_size=3072, hidden_size=200, output_size=10)
network = MultiLayerNet(input_size=3072, hidden_size_list=[100, 100, 100], output_size=10, activation='relu',
                        weight_init_std='relu', weight_decay_lambda=0.1, use_dropout=True, dropout_ration=0.5,
                        use_batchnorm=True)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

optimizer = SGD()
# optimizer = Momentum()
# optimizer = Nesterov()
# optimizer = AdaGrad()
# optimizer = RMSprop()
# optimizer = Adam()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    optimizer.update(network.params, grad)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train accuracy, test accuracy | " + str(train_acc) + ", " + str(test_acc))

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train accuracy')
plt.plot(x, test_acc_list, label='test accuracy', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
