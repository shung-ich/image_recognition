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


class params:
    def __init__(self, M, d):
        np.random.seed(seed=32)
        self.W = np.random.normal(0, 1/d, (d, M))
        self.b = np.random.normal(0, 1/d, (1, M))
        self.eta = 0.01

    def update(self, dW, db):
        self.W -= self.eta * dW
        self.b -= self.eta * db

    def save(self, i):
        np.save('./w{}'.format(i), self.W)
        np.save('./b{}'.format(i), self.b)
    
def load(i):
    W_loaded = np.load('./w{}.npy'.format(i))
    b_loaded = np.load('./b{}.npy'.format(i))
    return W_loaded, b_loaded

def create_batch(X):
    batch_size = 100
    np.random.seed(seed=32)
    batch_index = np.random.choice(len(X), (600, batch_size))
    return batch_index

def input_layer(X):
    i = int(input('参照する画像データのインデックスを入力してください. '))
    input_image = X[i] / 255
    image_size = input_image.size
    image_num = len(X)
    class_num = 10
    input_vector = input_image.reshape(1,image_size)
    return input_vector, image_size, i, class_num

def input_layer2(X, j):
    batch_index = create_batch(X)
    # input_images = X[batch_index] / 255
    input_images = X[batch_index[j]] / 255
    # image_size = input_images[0].size
    image_size = 784    
    class_num = 10
    input_vector = input_images.reshape(100,image_size)
    return input_vector, image_size, batch_index, class_num

class matrix_operation:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
    
    def forward(self, X):
        self.X = X 
        y = np.dot(X, self.W) + self.b
        return y

    def backward(self, back):
        dX = np.dot(back, self.W.T)
        dW = np.dot(self.X.T, back)
        db = np.sum(back, axis=0)
        return dX, dW, db

class sigmoid:
    def __init__(self):
        self.y = None
    
    def forward(self, t):
        self.y = (1 / (1 + np.exp(-1 * t)))
        return self.y

    def backward(self, back):
        dt = back * (1 - self.y) * self.y
        return dt

class ReLU():
    def __init__(self):
        self.a = None

    def forward(self, t):
        self.a = np.where(t > 0, t, 0)
        return self.a

    def backward(self, back):
        dt = back * np.where(self.a > 0, 1, 0)
        return dt

class Dropout():
    def __init__(self, rho, mask):
        self.rho = rho
        self.mask = None

    def forward(self, t, train_flag=1):
        if train_flag == 1:
            self.mask = np.random.rand(t.shape) > self.rho
            a = t * self.mask
            return a
        else:
            a = t * (1 - self.rho)
            return  a

    def backward(self, back)
        dt = back * self.mask
        return dt


class softmax:
    def __init__(self, batch_size):
        self.y_pred = None
        self.batch_size = batch_size

    def forward(self, a):
        alpha = np.tile(np.amax(a, axis=1), 10).reshape(10, self.batch_size).T
        # print('max', alpha)
        exp_a = np.exp(a - alpha)
        # print('e', exp_a)
        sum_exp = np.tile(np.sum(exp_a, axis=1), 10).reshape(10, self.batch_size).T
        # print('sum', sum_exp)
        self.y_pred = exp_a / sum_exp
        return self.y_pred

    def backward(self, y_ans, B):
        da = (self.y_pred - y_ans) / B
        return da

def postprocessing(y):
    binary_y = np.where(y == np.amax(y, axis=1), 1, 0)
    # print(np.where(binary_y == 1)[1][0])
    return binary_y

def cross_entropy_loss(y_pred, y_ans):
    B = len(y_pred)
    E = 1 / B * np.sum((-1) * y_ans * np.log(y_pred))
    return E

class neural_network():
    def __init__(self, batch_size, epoch, middle_layer, last):
        self.batch_size = batch_size
        self.epoch = epoch
        self.middle_layer = middle_layer
        self.last = last

    def learning(self):
        params1 = params(self.middle_layer, 784)
        params2 = params(self.last, self.middle_layer)
        for i in range(self.epoch):
            loss = []
            for j in range(int(60000 / self.batch_size)):
                input_vec, image_size, batch_index, class_sum = input_layer2(train_X, j)
                batch_label = train_Y[batch_index[j]]
                y_ans = np.identity(10)[batch_label]

                W1, b1 = params1.W, params1.b
                mo1 = matrix_operation(W1, b1)
                t = mo1.forward(input_vec)
                # print('matrix', t)
                sig = sigmoid()
                y1 = sig.forward(t)
                # print('sigmoid', y1)
                W2, b2 = params2.W, params2.b
                mo2 = matrix_operation(W2, b2)
                a = mo2.forward(y1)
                # print('a', a)
                soft = softmax(self.batch_size)
                y2 = soft.forward(a)
                # print(y2)
                # binary_y = postprocessing(y2)
                # print(binary_y)
                E = cross_entropy_loss(y2, y_ans)
                loss.append(E)

                da = soft.backward(y_ans, self.batch_size)
                dX2, dW2, db2 = mo2.backward(da)
                dt = sig.backward(dX2)
                dX1, dW1, db1 = mo1.backward(dt)
                params1.update(dW1, db1)
                params2.update(dW2, db2)

            print(np.sum(loss) / len(loss))

        params1.save(1)
        params2.save(2)
    
    def testing(self):
        input_vector, image_size, i, class_num = input_layer(test_X)
        # y_ans = np.identity(10)[test_Y[i]]
        W1, b1 = load(1)
        mo1 = matrix_operation(W1, b1)
        t = mo1.forward(input_vector)
        # print('matrix', y1)
        sig = sigmoid()
        y1 = sig.forward(t)
        # print('sigmoid', y1)
        W2, b2 = load(2)
        mo2 = matrix_operation(W2, b2)
        a = mo2.forward(y1)
        # print('a', a)
        soft = softmax(1)
        y2 = soft.forward(a)
        # print(y2)
        binary_y = postprocessing(y2)
        print(np.where(binary_y == 1)[1][0], test_Y[i])


nn = neural_network(100, 100, 50, 10)
print('学習を開始します. ')
nn.learning()
print('テストを開始します. ')
nn.testing()
