"""
在 firstNN_7的基础上
尝试添加一层隐藏层
//准确率可达0.9901960784313726
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
        self.whh=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.hnodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.bih=numpy.random.normal(0.0,1,(self.hnodes,1))
        self.bhh=numpy.random.normal(0.0,1,(self.hnodes,1))
        self.bho=numpy.random.normal(0.0,1,(self.onodes,1))
        self.lr=learningrate
        self.batch_size=batch_size
        self.activation_function=lambda x:scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = inputs_list
        targets =targets_list
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_inputs+=self.bih
        hidden_outputs = self.activation_function(hidden_inputs)

        hidden_inputs2=numpy.dot(self.whh,hidden_outputs)
        hidden_inputs2+=self.bhh
        hidden_outputs2=self.activation_function(hidden_inputs2)

        final_inputs = numpy.dot(self.who, hidden_outputs2)
        final_inputs+=self.bho
        final_outputs = self.activation_function(final_inputs)

        dZ3 = final_outputs-targets
        dwho=self.lr *(( numpy.dot(dZ3,numpy.transpose(hidden_outputs2)))/batch_size+lamda/batch_size*self.who)
        self.who-=dwho
        self.bho-=self.lr *(numpy.sum(dZ3,axis=1,keepdims=True)/batch_size)

        dZ2=numpy.dot(self.who.T,dZ3)*hidden_outputs2 * (1.0 - hidden_outputs2)
        dwhh=self.lr *((numpy.dot(dZ2,numpy.transpose (hidden_inputs)))/batch_size+lamda/batch_size*self.whh)
        dbhh=self.lr *(numpy.sum(dZ2,axis=1,keepdims=True)/batch_size)
        self.whh-=dwhh
        self.bhh-=dbhh

        dZ1=numpy.dot(self.whh.T,dZ2)*hidden_outputs*(1.0-hidden_outputs)
        dwih=self.lr*((numpy.dot(dZ1,numpy.transpose(inputs)))/batch_size+lamda/batch_size*self.wih)
        dbih=self.lr*(numpy.sum(dZ1,axis=1,keepdims=True)/batch_size)
        self.wih-=dwih
        self.bih-=dbih

        return final_outputs
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        inputs=numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_inputs+=self.bih
        hidden_outputs = self.activation_function(hidden_inputs)

        hidden_inputs2=numpy.dot(self.whh,hidden_outputs)
        hidden_inputs2+=self.bhh
        hidden_outputs2=self.activation_function(hidden_inputs2)

        final_inputs = numpy.dot(self.who, hidden_outputs2)
        final_inputs+=self.bho
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
input_nodes=784
hidden_nodes=30
output_nodes=10
learning_rate=0.1
batch_size=10
trainning_length=-1
test_length=-1
epochs=30
# lamda=0
lamda=0.001
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
mini_batches=divide_batch(trainning_data_list)
for e in range(epochs):
    for k in range(len(mini_batches)):
        n.train(mini_batches[k][0],mini_batches[k][1])
    pass
#————————————————————————————————————————————————————————



#————————————————————————————————————————————————————————

scorecard = []
for k in range(len(test_data[0])):
    all_values=test_data[0][k]
    correct_label = int(test_data[1][k])
    print(correct_label, "correct label")
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
