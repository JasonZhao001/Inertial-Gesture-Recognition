'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is work on the gesture recognition benchmark created by Chunyu & Baochang, etc. 
    Chunyu Xie, Shangzhen Luan, Hainan Wang, Baochang Zhang: Gesture Recognition Benchmark Based on Mobile Phone. CCBR 2016: 432-440
Long Short Term Memory paper: Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural computation, 1997, 9(8): 1735-1780.

******************************About author*************************************
Author: Zijian Zhao
E-mail: zhaozijian@buaa.edu.cn

******************************How to run***************************************
1. git clone https://github.com/bczhangbczhang/Inertial-Gesture-Recognition.git
2. move this file (LSTM.py) to the project root folder
3. modify the datafile to your specific path that point to the "struc.mat" data
4. run the following commond in the terminal at the project root path:
   (1) train: 
    $ python LSTM.py -t False
   (2) test only:
    $ python LSTM.py -t True
   ** one thing to note is that the test result will automatically printed after training
   ** the argparse is added for your performing the test process straightly.
******************************Notes********************************************
1. The data is divided into 120*8 training set and 20*8 testing set.
2. The model size is only  863.6 kB and run in real time, so it can be easily built in the cell phone. 
3. Result: test accuracy: 100.00% 
'''
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import scipy.io as scio
import numpy as np
import os
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='If is only test: True or else')
    parser.add_argument('--test', '-t', dest='is_only_test') # True or False
    args = parser.parse_args()
    return args

# data file and model save file
datafile='/home/zhao/workspace/Inertial-Gesture-Recognition/struct.mat'
struct = scio.loadmat(datafile)
LOGDIR = os.path.join('./save')
checkpoint_path = os.path.join(LOGDIR, "model.ckpt")

# Parameters
learning_rate = 0.001
training_iters = 150
batch_size = 32
display_step = 10

# Network Parameters
n_input = 2 # dimension of input_data 
n_steps = 34 # timesteps
n_hidden = 32 # hidden layer num of features
n_classes = 8 # gesture total class 8: 'one','two','three','four','A','B','C','D'

# Prepare training_data and test_data
def one_hot(i):
        a = np.zeros(8, 'uint8')
        a[i] = 1.
        return a

class_list = ['one','two','three','four','A','B','C','D']
char_to_ind = dict(zip(class_list, range(n_classes)))
train_data = []
train_label = []
test_data = []
test_label = []
for l in class_list:
    S = struct['struct_'+ l]
    for i in range(len(S[0])):
        if i >= 120:
            test_data.append(S[0][0][0])
            label1 = char_to_ind[l]
            one_hot_label1 = one_hot(label1)
            test_label.append(one_hot_label1)
        else:
            train_data.append(S[0][0][0])
            label2 = char_to_ind[l]
            one_hot_label2 = one_hot(label2)
            train_label.append(one_hot_label2)
train_data = np.asarray(train_data)
train_data = train_data.reshape(120*8, n_steps, 2)
train_label = np.asarray(train_label)
train_label = train_label.reshape(120*8, n_classes)
test_data = np.asarray(test_data)
test_data = test_data.reshape(20*8, n_steps, 2)
test_label = np.asarray(test_label)
test_label = test_label.reshape(20*8, n_classes)
# Shuffle the train data
perm = np.arange(120*8)
np.random.shuffle(perm)
train_data = train_data[perm]
train_label = train_label[perm]

# define the batch fetching function
def next_batch(batch_size, index_in_epoch, epochs_completed):
    """Return the next `batch_size` examples from this data set."""
    global train_data
    global train_label
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch >= 8*120:
        # Finished epoch
        epochs_completed += 1
        # Shuffle the data
        perm = np.arange(120*8)
        np.random.shuffle(perm)
        train_data = train_data[perm]
        train_label = train_label[perm]
        # Start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= 120*8
    end = index_in_epoch
    return train_data[start:end], train_label[start:end], index_in_epoch, epochs_completed

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def LSTM(x, weights, biases):

    # Prepare data shape to match `LSTM` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

args = parse_args()
pred = LSTM(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Launch the graph
with tf.Session() as sess:
    #sess.run(init)
    saver = tf.train.Saver()
    
    if args.is_only_test == 'True':
        try:
            saver.restore(sess, checkpoint_path)
            print("LOADED CHECKPOINT")
        except:
            print("FAILED TO LOAD CHECKPOINT")
            exit()
    else:
        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)
        

        index_in_epoch = 0
        epoch_completed = 0
        step = 1
        # Keep training until reach max iterations
        train_start_time = time.time()
        while step <= training_iters:
	        batch_x, batch_y, index_in_epoch, epoch_completed= next_batch(32, index_in_epoch, epoch_completed)

	        # Run optimization operation
	        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # Run evaluation on the current training batch
	        if step % display_step == 0:
	            # Calculate batch accuracy
	            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
	            # Calculate batch loss
	            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
	            print("Iter " + str(step) + ", Minibatch Loss= " + \
	                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
	                  "{:.5f}".format(acc))
		    if step % 100 == 0:
		        if not os.path.exists(LOGDIR):
		            os.makedirs(LOGDIR)
		        saver.save(sess, checkpoint_path)
	        step += 1
        train_duration = time.time() - train_start_time
        print("Training Finished in {:.5f} s!".format(train_duration))

    # Calculate accuracy for 160 test data
    test_start_time = time.time()
    test_accuracy = sess.run(accuracy, feed_dict={x: test_data, y: test_label})
    test_duration = time.time() - test_start_time
    print("Testing finished in {:.5f} s.".format(test_duration))
    print("Testing Accuracy:{:.5f}".format(test_accuracy))
