"""
在firstNeuralNetwork的基础上使用python深度学习的数据
"""
import numpy
import scipy.special
import matplotlib.pyplot
import pylab

class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.onodes=outputnodes
        self.hnodes=hiddennodes
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.lr=learningrate
        self.activation_function=lambda x:scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot((output_errors *
                                         final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors *
                                         hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose
                                        (inputs))
        return final_outputs
    def query(self,inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs =self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final outputlayer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.12
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)



def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = numpy.zeros((10, 1))
    e[j] = 1.0
    return e






import _pickle as cPickle
import gzip
def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f,encoding='bytes')
    f.close()

    # 单个的测试代码函数
    # testNumber=5
    # all_values=test_data_list[testNumber].split(',')
    # for k in range(0,len(training_data[0])):
    #     pict=training_data[0][k]
    #     image=numpy.asfarray(pict).reshape(28,28)
    #     matplotlib.pyplot.imshow(image,cmap='Greys',interpolation='None')
    #     pylab.show()
    #     print(training_data[1][k])
    #     pylab.time.sleep(2)

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""

    tr_d, va_d, te_d = load_data()
    training_inputs = [numpy.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [numpy.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [numpy.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)




#————————————————————————————————————————————————————————
# 第二个包里的数据训练
trainning_data_list, validation_data, test_data=load_data()
epochs=1
for e in range(epochs):
    # for k in range(len(trainning_data_list[0])):
    for k in range(len(trainning_data_list[0])):
        # inputs=(numpy.asfarray(trainning_data_list[0][k][0:])/255.0*0.99)+0.01
        inputs=trainning_data_list[0][k]
        targets=numpy.zeros(output_nodes)+0.01
        targets[int(trainning_data_list[1][k])]=0.99
        # # print(trainning_data_list[0][k])
        # print(inputs)
        # pylab.time.sleep(5)

        n.train(inputs,targets)
        if(k>5000):
            print("kkkkkkkkkkkkkk",k)
            break
        pass
    pass
#————————————————————————————————————————————————————————



#————————————————————————————————————————————————————————
#第二个包里的数据测试
scorecard = []
for k in range(len(test_data[0])):
    # all_values = numpy.reshape(test_data[0][k], (784, 1))
    all_values=test_data[0][k]
    correct_label = int(test_data[1][k])
    print(correct_label, "correct label")
    # inputs = (numpy.asfarray(all_values[0:]) / 255.0 * 0.99) + 0.01
    inputs=all_values
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print(label, "network's answer")
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    if(k>100):
        break
    pass
print(scorecard)
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() /
       scorecard_array.size)
#————————————————————————————————————————————————————————



"""
trainning_data_file=open("../_data/handwrittingNumber/mnist_train.csv",'r')
trainning_data_list=trainning_data_file.readlines()
trainning_data_file.close()
epochs=2
for e in range(epochs):
    for record in trainning_data_list:
        all_values= record.split(',')
        inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets=numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass
    pass


test_data_file=open("../_data/handwrittingNumber/mnist_test.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()

# #但个的测试代码函数
# testNumber=5
# all_values=test_data_list[testNumber].split(',')
# image=numpy.asfarray(all_values[1:]).reshape(28,28)
# matplotlib.pyplot.imshow(image,cmap='Greys',interpolation='None')
# pylab.show()
# final=n.query(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
# print(final)



#得分的测试
# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []
# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print(label, "network's answer")
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass
print(scorecard)
# 正确率显示
# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() /
       scorecard_array.size)
"""

