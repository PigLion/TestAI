"""
对firstNN_2_MGBD 进行修改
把每个的小批量改成随机
把一个小批量的数据用矩阵的形式输入计算
"""
import numpy
import scipy.special
import matplotlib.pyplot
import pylab

class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate,batch_size):
        self.inodes=inputnodes
        self.onodes=outputnodes
        self.hnodes=hiddennodes
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.lr=learningrate
        self.batch_size=batch_size
        self.activation_function=lambda x:scipy.special.expit(x)
        self.acount=batch_size
        self.arrih=numpy.zeros((self.hnodes,self.inodes),float)
        self.arrho=numpy.zeros((self.onodes,self.hnodes),float)
        pass
    def train(self,inputs_list,targets_list):
        # inputs = numpy.array(inputs_list, ndmin=2).T
        # targets = numpy.array(targets_list, ndmin=2).T
        inputs = inputs_list
        targets =targets_list
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        # output_errors+= lamda*(numpy.sum(numpy.square(self.wih)) + numpy.sum(numpy.square(self.who)))/trainning_length/2
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += (self.lr * numpy.dot((output_errors *
                                            final_outputs * (1.0 - final_outputs)),
                                           numpy.transpose(hidden_outputs)))/batch_size
        self.wih += (self.lr * numpy.dot((hidden_errors *
                                            hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose
                                           (inputs)))/batch_size
        return final_outputs
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs =self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.12
batch_size=10
trainning_length=-1
test_length=-1
epochs=10
# lamda=0
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate,batch_size)


def divide_batch(trainning_data):
    extend=numpy.zeros((10,trainning_length),float)
    for k in range(trainning_length):
        targets=numpy.zeros(output_nodes)+0.01
        targets[int(trainning_data_list[1][k])]=0.99
        extend[:,k]=targets
        pass
    trainning_data=(numpy.transpose(trainning_data[0]),extend)
    permutation = numpy.random.permutation(trainning_length)
    X_shuffle = trainning_data[0][:, permutation]
    Y_shuffle = trainning_data[1][:, permutation]

    mini_batches = []
    batch_num = trainning_length//batch_size
    for i in range(batch_num):
        mini_batch_x =  X_shuffle[:, i * batch_size: (i + 1) * batch_size]
        mini_batch_y =  Y_shuffle[:, i * batch_size: (i + 1) * batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    if batch_num * batch_size < trainning_length:
        mini_batch_x = X_shuffle[:, batch_num * batch_size: trainning_length]
        mini_batch_y = Y_shuffle[:, batch_num * batch_size: trainning_length]
        mini_batch = [mini_batch_x, mini_batch_y]
        mini_batches.append(mini_batch)

    mini_batches = numpy.array(mini_batches)
    return mini_batches



import _pickle as cPickle
import gzip
def load_data():
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

#————————————————————————————————————————————————————————
# 第二个包里的数据训练
trainning_data_list, validation_data, test_data=load_data()
if(trainning_length>0):
    trainning_data_list=(trainning_data_list[0][0:trainning_length],trainning_data_list[1][0:trainning_length])
else:
    trainning_length=numpy.shape(trainning_data_list[0])[0]
if(test_length>0):
    test_data=(test_data[0][0:test_length],test_data[1][0:test_length])
    validation_data=(validation_data[0][0:test_length],validation_data[1][0:test_length])
else:
    test_length=numpy.shape(test_data[0])[0]
# print(trainning_data_list[0].shape)
# print(trainning_data_list[1].shape)
# print(trainning_data_list[1][0])
# print(test_data[0].shape)
# print(test_data[1].shape)
# print(validation_data[0].shape)
# print(validation_data[1].shape)
mini_batches=divide_batch(trainning_data_list)
# print(mini_batches.shape)
for e in range(epochs):
    for k in range(len(mini_batches)):
        n.train(mini_batches[k][0],mini_batches[k][1])
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

