#!/usr/bin/env python3
"""
This program is done by group Team-SOLID

Answer to the question:
For the pre-processing, we remove the useless symbols and punctuation by simple for loop.
and we reverse the identations by simply replace them. We also defined a function to remove
non-English text and any illegal texts with no length. For postprocessing, as we want to improve
the efficiency of network learning, we remove some infrequent words from the texts to help
analyse the dataset

We use bidirectional - LSTM to model the stucture of the network as well as GRU. With the following
flow. Lstm -> Linear -> relu() -> Linear -> Output. Using tanh would reduce the accuracy, hence we
use relu here. Also, we use dropout rate of 0.5 after several tests in case of over-fitting. For the
loss function, we use common loss function that is provided by the torch which combines log_softmax
and nll_loss in a single function. Since rating correct and category correct have different score.
we decided to set loss of rating output two times of loss of category output in order to increse
correctness of rating output.

For the parameters that we have chosen, we use 300 instead of 50 or 100 because it is fastest of the
rest when training. We made changes to training parameters as well. We use trainValSplit ratio 0.85
instead of defaultvalue 0.8 simply because we want more data to be trained. Moreover, we use Adam as
optimizer other than SGD since we found Adam is better at converging the loss with a smaller learning
rate 0.001.We also increase the batch size to 128, but the accuracy decreased so we use default setting
of 32. The epoch is left as default.
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
    remove symbols and non English texts
    """
    def isword(str):
        if len(str)<=1:
            return False
        for s in str:
            if s in 'aeiou':
                return True
        return False

    result = []
    symbols = [",",".","?","!","(",")","$","~"]
    for data in sample:
        for sym in symbols:
            data = data.replace(sym,"")
        """
        identation reverse
        """
        data=data.replace("isn't","is not")
        data=data.replace("aren't","are not")
        data=data.replace("wasn't","was not")
        data=data.replace("weren't","were not")
        data=data.replace("doesn't","does not")
        data=data.replace("don't","do not")
        data=data.replace("didn't","did not")
        data=data.replace("haven't","have not")
        data=data.replace("hasn't","has not")
        data=data.replace("hadn't","had not")
        data=data.replace("i've","i have")
        data=data.replace("i'm","i am")
        data=data.replace("wouldn't","would not")
        data=data.replace("shouldn't","should not")
        data=data.replace("couldn't","could not")
        data=data.replace("won't","will not")
        data=data.replace("shan't","shall not")
        data=data.replace("needn't","need not")
        data=data.replace("mustn't","must not")
        data=data.replace("mightn't","might not")
        data=data.replace("should've","should have")

        if data.isalpha()and len(data)>1 and isword(data):
            result.append(data)

    return result

def postprocessing(batch, vocab):
    """
    remove infrequent words of sentence
    """
    freq = vocab.freqs
    Ints = vocab.itos
    for line in batch:
        for index,word in enumerate(line):
            if (freq[Ints[word]]) <3:
                line[index] = -1

    return batch

stopWords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',  'ma'}
wordVectorDimension = 300
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
    ratingOutput = torch.sigmoid(ratingOutput).argmax(dim=1)
    categoryOutput = torch.softmax(categoryOutput,dim=1).argmax(dim=1)

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

        binary=[50, 2, 0.5]
        multl=[50, 2, 0.5]

        #binary classifier #binary=[hidden size, layer,dropout] for dealing with ratingoutput
        self.batch_size = 32

        self.hidden_size_bi = binary[0]
        self.layers_bi = binary[1]

        # Initializing the look-up table.

        self.lstm_bi= tnn.LSTM(wordVectorDimension,self.hidden_size_bi, self.layers_bi,batch_first=True,bidirectional = True,dropout=binary[2])
        self.label_L1_bi = tnn.Linear(self.hidden_size_bi * self.layers_bi * 2, 64)
        self.Relu_bi = tnn.ReLU()
        self.label_L2_bi = tnn.Linear(64, 2)
        # output(0,1)


        #multi classifier #multi=[hidden size, layer,dropout] for dealing with businesscatogary

        self.hidden_size = multl[0]
        self.layers = multl[1]

        # Initializing the look-up table.

        self.gru = tnn.GRU(wordVectorDimension, self.hidden_size, self.layers, batch_first=True,bidirectional=True,dropout=multl[2])
        self.fc1 = tnn.Linear(self.hidden_size*2, 150)
        self.fc2 = tnn.Linear(150, 5)
        # output(0,1,2,3,4)

    def forward(self, input, length):

        # Network structure: LSTM (bidirectional) -> Linear1 -> Relu -> Linear2 -> output
        output_bi, _ = self.lstm_bi(input)
        x_bi = torch.cat((output_bi[:, -1, :], output_bi[:, 0, :]), dim=1)
        L1_bi = self.label_L1_bi(x_bi)
        L2_bi = self.Relu_bi(L1_bi)
        final_output_bi = self.label_L2_bi(L2_bi)

        # Network structure: GRU (bidirectional) -> Linear1 -> Linear2 -> output
        output, _ = self.gru(input)
        L1 = self.fc1(output)
        L2 = self.fc2(L1)
        final_output = L2[:,-1,:]

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

        loss = 0.66*closs + 0.34*rloss
        return loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.85
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.001)
