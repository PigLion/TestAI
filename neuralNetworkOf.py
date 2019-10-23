# coding:utf8
import cPickle
import numpy as np
import matplotlib.pyplot as plt


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # L(n-1)->L(n)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b_, w_ in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w_, a)+b_)
        return a

    def SGD(self, training_data, test_data,epochs, mini_batch_size, eta):
        n_test = len(test_data)
        n = len(training_data)
        plt.xlabel('epoch')
        plt.title('cost')
        cy=[]
        cx=range(epochs)
        for j in cx:
            self.cost = 0.0
            np.random.shuffle(training_data)  # shuffle
            for k in xrange(0, n, mini_batch_size):
                mini_batch = training_data[k:k+mini_batch_size]
                self.update_mini_batch(mini_batch, eta)
            cy.append(self.cost/n)
            print "Epoch {0}: {1} / {2}".format(
                j, self.evaluate(test_data), n_test)
        plt.plot(cx,cy)
        plt.scatter(cx,cy)
        plt.show()

    def update_mini_batch(self, mini_batch, eta):
        for x, y in mini_batch:
            delta_b, delta_w,cost = self.backprop(x, y)
            self.weights -= eta/len(mini_batch)*delta_w
            self.biases -= eta/len(mini_batch)*delta_b
            self.cost += cost

    def backprop(self, x, y):
        b=np.zeros_like(self.biases)
        w=np.zeros_like(self.weights)
        a_ = x
        a = [x]
        for b_, w_ in zip(self.biases, self.weights):
            a_ = self.sigmoid(np.dot(w_, a_)+b_)
            a.append(a_)
        for l in xrange(1, self.num_layers):
            if l==1:
                # delta= self.sigmoid_prime(a[-1])*(a[-1]-y)  # O(k)=a[-1], t(k)=y
                delta= a[-1]-y  # cross-entropy
            else:
                sp = self.sigmoid_prime(a[-l])   # O(j)=a[-l]
                delta = np.dot(self.weights[-l+1].T, delta) * sp
            b[-l] = delta
            w[-l] = np.dot(delta, a[-l-1].T)
        cost=0.5*np.sum((b[-1])**2)
        return (b, w,cost)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):
        return z*(1-z)

if __name__ == '__main__':

    def get_label(i):
        c=np.zeros((10,1))
        c[i]=1
        return c

    def get_data(data):
        return [np.reshape(x, (784,1)) for x in data[0]]

    f = open('mnist.pkl', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    training_inputs = get_data(training_data)
    training_label=[get_label(y_) for y_ in training_data[1]]
    data = zip(training_inputs,training_label)
    test_inputs = training_inputs = get_data(test_data)
    test = zip(test_inputs,test_data[1])
    net = Network([784, 30, 10])
    net.SGD(data[:5000],test[:5000],50,10, 3.0,)   # 4476/5000 (4347/5000)