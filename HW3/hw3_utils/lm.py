import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from hw3_utils import vocab

class NameGenerator(nn.Module):
    def __init__(self, input_vocab_size, n_embedding_dims, n_hidden_dims, n_lstm_layers, output_vocab_size):
        """
        Initialize our name generator, following the equations laid out in the assignment. In other words,
        we'll need an Embedding layer, an LSTM layer, a Linear layer, and LogSoftmax layer. 
        
        Note: Remember to set batch_first=True when initializing your LSTM layer!

        Also note: When you build your LogSoftmax layer, pay attention to the dimension that you're 
        telling it to run over!
        """
        super(NameGenerator, self).__init__()
        self.lstm_dims = n_hidden_dims
        self.lstm_layers = n_lstm_layers
        #raise NotImplementedError
        
        # Our input embedding layer:
        self.input_lookup = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=n_embedding_dims)
        
        # Note the use of batch_first in the LSTM initialization- this has to do with the layout of the
        # data we use as its input. See the docs for more details
        self.lstm = nn.LSTM(input_size=n_embedding_dims, hidden_size=n_hidden_dims, num_layers=n_lstm_layers, batch_first=True)
        
        # The output softmax classifier: first, the linear layer:
        self.output = nn.Linear(in_features=n_hidden_dims, out_features=output_vocab_size)
        
        # Then, the actual log-softmaxing:
        # Note that we are using LogSoftmax here, since we want to use negative log-likelihood as our loss function.
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, history_tensor, prev_hidden_state):
        """
        Given a history, and a previous timepoint's hidden state, predict the next character. 
        
        Note: Make sure to return the LSTM hidden state, so that we can use this for
        sampling/generation in a one-character-at-a-time pattern, as in Goldberg 9.5!
        """        
        #raise NotImplementedError
        history_tensor = history_tensor.long()
        #prev_hidden_state = prev_hidden_state.long()
        embeddings = self.input_lookup(history_tensor)
        
        lstm_output = self.lstm(embeddings, prev_hidden_state)
        
        linear_output = self.output(lstm_output[0])
        #print(linear_output)
        #print(linear_output.type())
        #print(self.softmax(linear_output).type())
        
        softmax_output = self.softmax(linear_output)
        
        #print(softmax_output)
        #print(linear_output.type())
        
        #print("lstm output[-1][0]")
        #print(lstm_output[-1][0].shape)
        #print(lstm_output[-1][1].shape)
        #return(softmax_output, lstm_output)
        return(softmax_output, (lstm_output[-1][0], lstm_output[-1][1]))
        
    def init_hidden(self):
        """
        Generate a blank initial history value, for use when we start predicting over a fresh sequence.
        """
        h_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        c_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        #print(h_0)
        #return (h_0, c_0)
    
### Utility functions

def train(model, epochs, training_data, c2i):
    """
    Train model for the specified number of epochs, over the provided training data.
    
    Make sure to shuffle the training data at the beginning of each epoch!
    """
    #raise NotImplementedError
    opt = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.NLLLoss(reduction = 'sum') # since our model gives negative log probs on the output side
    #loss_batch_size = 100
    
    i2c = dict()
    for char in c2i:
        i2c[c2i[char]] = char
        
    for i in range(epochs):
        
        training_data_tensors = [vocab.sentence_to_tensor(example, c2i, True) for example in training_data]
        x_train = [example[:,:-1] for example in training_data_tensors]
        y_train = [example[:,1:] for example in training_data_tensors]
        
        #print(x_train)
        #print(y_train)
            
        pairs = list(zip(x_train, y_train))
        random.shuffle(pairs)    
    
        loss = 0
        
        for x_idx, (x, y) in enumerate(pairs):
            #print(x,y)
            #print(x_idx)
            #if len(x) == 0 or len(y) == 0:
            #    print("zero length")
            #    continue
            #if len(x) ==1 & len(y) == 1:
            #    print("one length")
            #    continue
            if x_idx == 0:
                opt.zero_grad()
            
            #x_tens = vocab.sentence_to_tensor(x, c2i)
            
            model_result = model(x, model.init_hidden())
            y_hat = model_result[0]
            
            #if x_idx == 0:
            #    model_result = model(x_tens, model.init_hidden())
            #    y_hat = model_result[0]
            #    prev_hidden_state = model_result[1]
            #else:
            #    model_result = model(x_tens, prev_hidden_state)            
            #    y_hat = model_result[0]
            #    prev_hidden_state = model_result[1]
             
                
            #y_tens = vocab.sentence_to_tensor(y, c2i)
            #print(y_hat.shape)
            #print(y_tens.shape)
            
            if y_hat.squeeze().dim() <2:
                print("ugh dim")
                print((x,y))
            #loss += loss_func(y_hat.unsqueeze(0), y_tens.unsqueeze(0).long())
            loss += loss_func(y_hat.squeeze(), y.squeeze().long())
            
            if x_idx % 1000 == 0:
                print(x_idx)
                for i in range(3):
                    print(sample(model, c2i, i2c))
            #    print(f"{x_idx}/{len(pairs)} average per-item loss: {loss / loss_batch_size}")
                
            #if x_idx % loss_batch_size == 0 and x_idx > 0:
                #print("going backwards")
            # send back gradients:
            loss.backward()
                # now, tell the optimizer to update our weights:
            opt.step()
            loss = 0
            opt.zero_grad()
            #prev_hidden_state = model.init_hidden()
        
        # now one last time:
        #loss.backward()
        #opt.step()
        
    return model

def sample(model, c2i, i2c, max_seq_len=200):
    """
    Sample a new sequence from model.
    
    The length of the resulting sequence should be < max_seq_len, and the 
    new sequence should be stripped of <bos>/<eos> symbols if necessary.
    """
    #raise NotImplementedError
    history = [vocab.BOS_SYM]
    history_tensor = vocab.sentence_to_tensor(history, c2i)
    
    y_hat, prev_hidden = model(history_tensor, model.init_hidden())
    i = 0
    while len(history)< max_seq_len and history[-1] != vocab.EOS_SYM:
        y_hat = y_hat.squeeze()
        y_hat = y_hat.exp()
        next_char = torch.multinomial(y_hat, 1)
        if i == 0:
            history.append(i2c[next_char.tolist()[0]])
        else:
            history.append(i2c[next_char.tolist()[-1][0]])
        history_tensor = vocab.sentence_to_tensor(history, c2i)
        y_hat, prev_hidden = model(history_tensor, model.init_hidden())
        i+=1
    
    history = ''.join(history[1:-1])
    return history
        
    

    
def compute_prob(model, sentence, c2i):
    """
    Compute the negative log probability of p(sentence)
    
    Equivalent to equation 3.3 in Jurafsky & Martin.
    """
    
    nll = nn.NLLLoss(reduction='sum')
    
    with torch.no_grad():
        s_tens = vocab.sentence_to_tensor(sentence, c2i, True)
        x = s_tens[:,:-1]
        y = s_tens[:,1:]
        print(x)
        print(y)
        #print(y)
        #print(y.squeeze())
        y_hat, _ = model(x, model.init_hidden())
        #print(y_hat)
        #print(y_hat.squeeze())
        #y_hat = y_hat.scalar()
        print(y_hat.shape)
        print(y.shape)
        return nll(y_hat.squeeze(), y.squeeze().long()).item() # get rid of first dimension of each