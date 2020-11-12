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
        # self.op1 = optimize('default')
        # self.op2 = optimize('default')
        approach = 'Adam'
        self.op1 = optimize(approach)
        self.op2 = optimize(approach)

    def update(self, dW, db):
        # self.W -= self.eta * dW
        # self.b -= self.eta * db
        self.W += self.op1.update(dW)
        self.b += self.op2.update(db)

    def save(self, i):
        np.save('./w{}'.format(i), self.W)
        np.save('./b{}'.format(i), self.b)

class optimize:    
    def __init__(self, approach):
        self.approach = approach
        self.diff = 0
        if approach == 'default':
            self.eta = 0.01
        elif approach == 'SGD':
            self.eta = 0.01
            self.alpha = 0.9
        elif approach == 'AdaGrad':
            self.h = 1e-8
            self.eta = 0.001
        elif approach == 'RMSProp':
            self.h = 0
            self.eta = 0.001
            self.rho = 0.9
            self.epsilon = 1e-8
        elif approach == 'AdaDelta':
            self.h = 0
            self.s = 0
            self.rho = 0.95
            self.epsilon = 1e-6
        elif approach == 'Adam':
            self.t = 0
            self.m = 0
            self.v = 0
            self.alpha = 0.001
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8

    def update(self, d_):
        if self.approach == 'default':
            self.diff = (-1) * self.eta * d_
        elif self.approach == 'SGD':
            self.diff = self.alpha * self.diff - self.eta * d_
        elif self.approach == 'AdaGrad':
            self.h = self.h + d_ * d_
            self.diff = (-1) * self.eta / np.sqrt(self.h) * d_
        elif self.approach == 'RMSProp':
            self.h = self.rho * self.h + (1 - self.rho) * d_ * d_
            self.diff = (-1) * self.eta / (np.sqrt(self.h) + self.epsilon) * d_
        elif self.approach == 'AdaDelta':
            self.h = self.rho * self.h + (1 - self.rho) * d_ * d_
            self.diff = (-1) * np.sqrt(self.s + self.epsilon) / np.sqrt(self.h + self.epsilon) * d_
            self.s = self.rho * self.s + (1 - self.rho) * self.diff * self.diff
        elif self.approach == 'Adam':
            self.t = self.t + 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * d_
            self.v = self.beta2 * self.v + (1 - self.beta2) * d_ * d_
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            self.diff = (-1) * self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return self.diff

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

def input_layer_test(X, i):
    input_image = X[i] / 255
    image_size = input_image.size
    image_num = len(X)
    class_num = 10
    input_vector = input_image.reshape(1,image_size)
    return input_vector, image_size, i, class_num

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
    def __init__(self, rho):
        self.rho = rho
        self.mask = None

    def forward(self, t, train_flag=1):
        if train_flag == 1:
            self.mask = np.random.rand(*t.shape) > self.rho
            a = t * self.mask
            return a
        else:
            a = t * (1 - self.rho)
            return  a

    def backward(self, back):
        dt = back * self.mask
        return dt

class Batch_Normalization():
    mean_list = []
    var_list = []

    def __init__(self):
        self.batch_size = None
        self.gamma = 1
        self.beta = 0
        self.x = None
        self.mean = None
        self.var = None
        self.normalized_x = None
        self.epsilon = 1e-7
        self.op1 = optimize('Adam')
        self.op2 = optimize('Adam')

    def forward(self, x, train_flag=1):
        self.x = x
        if train_flag == 1:
            self.batch_size = x.shape[0]
            # mean = np.sum(x, axis=0) / self.batch_size
            self.mean = np.mean(x, axis=0)
            Batch_Normalization.mean_list = np.append(Batch_Normalization.mean_list, self.mean, axis=0)
            self.var = np.var(x, axis=0)
            Batch_Normalization.var_list = np.append(Batch_Normalization.var_list, self.var, axis=0)
            # print('x: ',  x.shape)
            self.normalized_x = (x - self.mean) / np.sqrt(self.var + self.epsilon)
            y = self.gamma * self.normalized_x + self.beta
        else:
            y = self.gamma / np.sqrt(np.mean(self.var_list, axis=0) + self.epsilon) * x + \
                (self.beta - self.gamma * np.mean(self.mean_list, axis=0) / np.sqrt(np.mean(self.var_list, axis=0) + self.epsilon))
        return y

    def backward(self, back):
        dn_x = back * self.gamma
        dvar = np.sum(dn_x * (self.x - self.mean) * (-1 / 2) * (self.var + self.epsilon) ** (-3 / 2), axis=0)
        dmean = np.sum(dn_x * (-1) / np.sqrt(self.var + self.epsilon), axis=0) + dvar * np.sum(-2 * (self.x - self.mean), axis=0) / self.batch_size
        dx = dn_x / np.sqrt(self.var + self.epsilon) + dvar * 2 * (self.x - self.mean) / self.batch_size + dmean / self.batch_size
        dgamma = np.sum(back * self.normalized_x, axis=0)
        dbeta = np.sum(back, axis=0)
        self.gamma += self.op1.update(dgamma)
        self.beta += self.op2.update(dbeta)
        return dx

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

                bn = Batch_Normalization()
                y_bn = bn.forward(t)

                # sig = sigmoid()
                # y1 = sig.forward(t)
                re = ReLU()
                y_re = re.forward(y_bn)
                # print('sigmoid', y1)

                dr = Dropout(0.2)
                y1 = dr.forward(y_re)

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

                dt_dr = dr.backward(dX2)

                # dt = sig.backward(dX2)
                dt_re = re.backward(dt_dr)

                dt = bn.backward(dt_re)
                
                dX1, dW1, db1 = mo1.backward(dt)
                params1.update(dW1, db1)
                params2.update(dW2, db2)

            print(np.sum(loss) / len(loss))

        params1.save(1)
        params2.save(2)
    
    def testing(self):
        # input_vector, image_size, i, class_num = input_layer(test_X)
        ans = []
        for k in range(test_Y.shape[0]):
            input_vector, image_size, i, class_num = input_layer_test(test_X, k)
            # y_ans = np.identity(10)[test_Y[i]]
            W1, b1 = load(1)
            mo1 = matrix_operation(W1, b1)
            t = mo1.forward(input_vector)

            bn = Batch_Normalization()
            y_bn = bn.forward(t, 0)

            # print('matrix', y1)
            # sig = sigmoid()
            # y1 = sig.forward(t)
            re = ReLU()
            y_re = re.forward(y_bn)
            # print('sigmoid', y1)

            dr = Dropout(0.2)
            y1 = dr.forward(y_re, 0)

            W2, b2 = load(2)
            mo2 = matrix_operation(W2, b2)
            a = mo2.forward(y1)
            # print('a', a)

            soft = softmax(1)
            y2 = soft.forward(a)
            # print(y2)
            binary_y = postprocessing(y2)
            # print(np.where(binary_y == 1)[1][0], test_Y[i])
            eq = 1 if np.where(binary_y == 1)[1][0] == test_Y[i] else 0
            ans.append(eq)
        print(np.mean(ans))


nn = neural_network(100, 5, 50, 10)
print('学習を開始します. ')
nn.learning()
print('テストを開始します. ')
nn.testing()
