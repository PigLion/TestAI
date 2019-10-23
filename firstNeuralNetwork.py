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
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights,recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors *
                                         final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors *
                                         hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose
                                        (inputs))
        pass
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
learning_rate=0.3
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)


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


