import numpy as np
import matplotlib.pyplot as plt
import mnist

train_X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
# print(type(np.array(train_X[0])))
# print(train_X.shape)
train_Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
test_X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
test_Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")
# print(len(test_X))

# from pylab import cm
# idx = 105
# plt.imshow(train_X[idx], cmap=cm.gray)
# plt.show()
# print (train_Y[idx])
# print(np.array(train_X[0]).reshape(1,784))

# def preprocessing(N, M, d):
#     np.random.seed(seed=32)
#     W = np.random.normal(0, 1/N, (d, M))
#     b = np.random.normal(0, 1/N, (1, M))
#     return W, b
class params:
    def __init__(self, M, d):
        np.random.seed(seed=32)
        self.W = np.random.normal(0, 1/d, (d, M))
        self.b = np.random.normal(0, 1/d, (1, M))
        self.eta = 0.01

    def update(self, dW, db):
        self.W -= self.eta * dW
        self.b -= self.eta * db

def create_batch(X):
    batch_size = 100
    np.random.seed(seed=32)
    # batch_index = np.random.choice(len(X), batch_size)
    batch_index = np.random.choice(len(X), (600, batch_size))
    return batch_index

def input_layer(X):
    i = int(input())
    input_image = X[i]
    image_size = input_image.size
    image_num = len(X)
    class_num = 10
    input_vector = input_image.reshape(1,image_size)
    return input_vector, image_size, i, class_num

# def input_layer2(X):
def input_layer2(X, j):
    batch_index = create_batch(X)
    # input_images = X[batch_index] / 255
    input_images = X[batch_index[j]] / 255
    # image_size = input_images[0].size
    image_size = 784    
    class_num = 10
    input_vector = input_images.reshape(100,image_size)
    return input_vector, image_size, batch_index, class_num

# def matrix_operation(W, X, b):
#     return np.dot(X, W) + b
class matrix_operation:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        # self.dX = None
        # self.dW = None
        # self.db None
    
    def forward(self, X):
        self.X = X 
        y = np.dot(X, self.W) + self.b
        return y

    def backward(self, back):
        # self.dX = np.dot(self.W.T, back)
        # self.dW = np.dot(back, self.X.T)
        # self.db = np.sum(back, axis=1)
        # return self.dX, self.dW, self.db
        dX = np.dot(back, self.W.T)
        dW = np.dot(self.X.T, back)
        db = np.sum(back, axis=0)
        return dX, dW, db


# def sigmoid(x):
#     return (1 / (1 + np.exp(-1 * x)))
class sigmoid:
    def __init__(self):
        self.y = None
        # self.dt = None
    
    def forward(self, t):
        self.y = (1 / (1 + np.exp(-1 * t)))
        return self.y

    def backward(self, back):
        # self.dt = back * (1 - self.y) * self.y
        # return self.dt
        dt = back * (1 - self.y) * self.y
        return dt


# def softmax(a):
#     alpha = np.amax(a)
#     # print('max', alpha)
#     exp_a = np.exp(a - alpha)
#     # print('e', exp_a)
#     sum_exp = np.sum(exp_a)
#     # print('sum', sum_exp)
#     y = exp_a / sum_exp
#     return y
class softmax:
    def __init__(self):
        self.y_pred = None

    def forward(self, a):
        # alpha = np.amax(a)
        alpha = np.tile(np.amax(a, axis=1), 10).reshape(10, 100).T
        # print('max', alpha)
        exp_a = np.exp(a - alpha)
        # print('e', exp_a)
        sum_exp = np.tile(np.sum(exp_a, axis=1), 10).reshape(10, 100).T
        # print('sum', sum_exp)
        self.y_pred = exp_a / sum_exp
        return self.y_pred

    def backward(self, y_ans, B):
        da = (self.y_pred - y_ans) / B
        return da


def postprocessing(y):
    binary_y = np.where(y == np.amax(y), 1, 0)
    # print(np.where(binary_y == 1)[1][0])
    return binary_y

def cross_entropy_loss(y_pred, y_ans):
    B = len(y_pred)
    E = 1 / B * np.sum((-1) * y_ans * np.log(y_pred))
    return E


# input_vec, image_size, i, class_sum = input_layer(test_X)
# print('input', image_size, i, class_sum )

# input_vec, image_size, batch_index, class_sum = input_layer2(test_X)
# batch_label = train_Y[batch_index]
# y_ans = np.identity(10)[batch_label]
# # print(batch_label)
# # print('input', image_size, batch_index, class_sum)
# W1, b1 = preprocessing(image_size, 30, image_size)
# y1 = matrix_operation(W1, input_vec, b1)
# # print('matrix', y1)
# y1 = sigmoid(y1)
# # print('sigmoid', y1)
# W2, b2 = preprocessing(30, class_sum, 30)
# a = matrix_operation(W2, y1, b2)
# # print('a', a)
# y2 = softmax(a)
# # print(y2)
# binary_y = postprocessing(y2)
# # print(binary_y)
# E = cross_entropy_loss(y2, y_ans)
# print(E)

# input_vec, image_size, batch_index, class_sum = input_layer2(train_X)
# batch_label = train_Y[batch_index]
# y_ans = np.identity(10)[batch_label]

# print(batch_label)
# print('input', image_size, batch_index, class_sum)

# params1 = params(30, image_size)
# params2 = params(class_sum, 30)
params1 = params(30, 784)
params2 = params(10, 30)
for i in range(100):
    loss = []
    for j in range(int(600)):
        input_vec, image_size, batch_index, class_sum = input_layer2(train_X, j)
        batch_label = train_Y[batch_index[j]]
        y_ans = np.identity(10)[batch_label]

        W1, b1 = params1.W, params1.b
        mo1 = matrix_operation(W1, b1)
        t = mo1.forward(input_vec)
        # print('matrix', y1)
        sig = sigmoid()
        y1 = sig.forward(t)
        # print('sigmoid', y1)
        W2, b2 = params2.W, params2.b
        mo2 = matrix_operation(W2, b2)
        a = mo2.forward(y1)
        # print('a', a)
        soft = softmax()
        y2 = soft.forward(a)
        # print(y2)
        binary_y = postprocessing(y2)
        # print(binary_y)
        E = cross_entropy_loss(y2, y_ans)
        loss.append(E)

        da = soft.backward(y_ans, 100)
        dX2, dW2, db2 = mo2.backward(da)
        dt = sig.backward(dX2)
        dX1, dW1, db1 = mo1.backward(dt)
        params1.update(dW1, db1)
        params2.update(dW2, db2)
        print(E)

    # print(np.sum(E) / 600)


