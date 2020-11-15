#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
from torch.autograd import Variable

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}
wordVectorDimension = 100
wordVectors = GloVe(name='6B', dim=wordVectorDimension)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """

    ratingOutput = ratingOutput.round()
    ratingOutput[ratingOutput > 1] = 1.0
    ratingOutput[ratingOutput < 0] = 0.0

    categoryOutput = categoryOutput.round()
    categoryOutput[categoryOutput > 5] = 5.0
    categoryOutput[categoryOutput < 1] = 1.0

    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        multl=[50, 2, 0.5]

        binary=[50, 2, 0.5]
        
        #binary classifier #binary=[hidden size, layer,dropout] for dealing with ratingoutput
        self.batch_size = 32

        self.hidden_size_bi = binary[0]
        self.layers_bi = binary[1]

        # Initializing the look-up table.

        self.lstm_bi= tnn.LSTM(wordVectorDimension, self.hidden_size_bi, self.layers_bi)
        self.label_bi = tnn.Linear(self.hidden_size_bi, 2)
        self.dropout_bi = tnn.Dropout(binary[2])


        #multi classifier #multi=[hidden size, layer,dropout] for dealing with businesscatogary 

        self.hidden_size = multl[0]
        self.layers = multl[1]

        # Initializing the look-up table.

        self.lstm= tnn.LSTM(wordVectorDimension, self.hidden_size, self.layers)
        self.label = tnn.Linear(self.hidden_size, 5)
        self.dropout = tnn.Dropout(multl[2])

    def forward(self, input, length):
        
        embed=self.dropout(input)
        
        if length is None:
              h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device)) # Initial hidden state of the LSTM
              c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device)) # Initial cell state of the LSTM

              h_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device)) # Initial hidden state of the LSTM
              c_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device)) # Initial cell state of the LSTM
        else:
              h_0 = Variable(torch.zeros(1, length[0], self.hidden_size_bi).to(device))
              c_0 = Variable(torch.zeros(1, length[0], self.hidden_size_bi).to(device))

              h_1 = Variable(torch.zeros(1, length[1], self.hidden_size).to(device))
              c_1 = Variable(torch.zeros(1, length[1], self.hidden_size).to(device))


        output_bi, _ = self.lstm_bi(embed, (h_0,c_0))
        final_output_bi = self.label_bi(output_bi[:,-1,:])

        output, _ = self.lstm(embed, (h_1,c_1))
        final_output = self.label(output[:,-1,:])

        return final_output_bi,final_output
class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        rloss = tnn.functional.cross_entropy(ratingOutput,ratingTarget)

        closs = tnn.functional.cross_entropy(categoryOutput, categoryTarget)

        loss = 2 * closs + rloss
        return loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.01)

