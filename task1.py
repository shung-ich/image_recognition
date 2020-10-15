import numpy as np
import matplotlib.pyplot as plt
import mnist

train_X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
train_Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
test_X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
test_Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")
# print(len(test_X))


from pylab import cm
# idx = 100
# plt.imshow(train_X[idx], cmap=cm.gray)
# plt.show()
# print (train_Y[idx])
# print(np.array(X[0]).reshape(1,784))

def preprocessing(N, M, d):
    np.random.seed(seed=32)
    W = np.random.normal(0, 1/N, (d, M))
    b = np.random.normal(0, 1/N, (1, M))
    return W, b

def input_layer(X):
    i = int(input())
    input_image = np.array(X[i])
    image_size = input_image.size
    image_num = len(X)
    class_num = 10
    input_vector = input_image.reshape(1,image_size)
    return input_vector, image_size, i, class_num

def matrix_operation(W, X, b):
    return np.dot(X, W) + b

def sigmoid(x):
    return (1 / (1 + np.exp(-1 * x)))

def softmax(a):
    alpha = np.amax(a)
    # print('max', alpha)
    exp_a = np.exp(a - alpha)
    # print('e', exp_a)
    sum_exp = np.sum(exp_a)
    # print('sum', sum_exp)
    # y =  np.exp(a_i - alpha) / sum for a_i in a
    y = exp_a / sum_exp
    return y

def postprocessing(y):
    binary_y = np.where(y == np.amax(y), 1, 0)
    print(np.where(binary_y == 1)[1][0])
    return binary_y



input_vec, image_size, i, class_sum = input_layer(test_X)
# print('input', image_size, i, class_sum )
W1, b1 = preprocessing(image_size, 30, image_size)
y1 = matrix_operation(W1, input_vec, b1)
# print('ma', y1)
y1 = sigmoid(y1)
# print('si', y1)
W2, b2 = preprocessing(30, class_sum, 30)
a = matrix_operation(W2, y1, b2)
# print('a', a)
y2 = softmax(a)
# print(y2)
binary_y = postprocessing(y2)
# print(binary_y)
