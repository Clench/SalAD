import math
import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import os


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        """

        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], # (b,t,c,h,w)
                                              h_cur=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]
        return layer_output_list, last_state_list
        # return torch.FloatTensor(layer_output_list), torch.FloatTensor(last_state_list)

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# if __name__ == '__main__':
#     # set CUDA device
#     os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#     # detect if CUDA is available or not
#     use_gpu = torch.cuda.is_available()
#     if use_gpu:
#         dtype = torch.cuda.FloatTensor # computation in GPU
#     else:
#         dtype = torch.FloatTensor

#     height = width = 6
#     channels = 256
#     hidden_dim = [32, 64]
#     kernel_size = (3,3) # kernel size for two stacked hidden layer
#     num_layers = 2 # number of stacked hidden layer
#     model = ConvGRU(input_size=(height, width),
#                     input_dim=channels,
#                     hidden_dim=hidden_dim,
#                     kernel_size=kernel_size,
#                     num_layers=num_layers,
#                     dtype=dtype,
#                     batch_first=True,
#                     bias = True,
#                     return_all_layers = False)

#     batch_size = 1
#     time_steps = 1
#     input_tensor = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w)
#     layer_output_list, last_state_list = model(input_tensor)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class ConvLSTM_Net(nn.Module):
    def __init__(self, seq_len, output_len, n_features, out_n_features, embedding_dim=128, kernel_size=3):
        super(ConvLSTM_Net, self).__init__()

        self.kernel_size = kernel_size
        self.n_features = n_features
        self.num_encoder_layers = 6
        self.seq_len = seq_len
        self.output_len = output_len
        self.out_n_features = out_n_features
        # # ConvLSTM input shape = (b, t, c, w, h) --> (batch, window size=100=t, d_model) = (b, window size=100=t, self.d_model, 1, 1)

        self.convLSTM = ConvLSTM(input_dim=self.n_features, 
                                hidden_dim=[32, 64],
                                kernel_size=(3,3),
                                num_layers=2,
                                batch_first=True,
                                bias = True,
                                return_all_layers = False) # if True, return length will be equal to number of layers

        # Last linear
        self.linear = nn.Linear(64, self.out_n_features)

    def forward(self, x):

        x = x.reshape(x.shape[0], 10, x.shape[2], 10, 1)
        layer_output_list, last_state_list = self.convLSTM(x)
        out = layer_output_list[0]
        out = out.reshape(out.shape[0], self.seq_len, out.shape[2])

        out = self.linear(out)

        return out[:, self.seq_len-self.output_len:, :]



class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, input):
        return self.linear(input)
    
    
    
    
class Encoder(nn.Module):
    def __init__(self, input_dim=6, hid_dim=64, n_layers=4, model_type='RNN'):
        super().__init__()
        assert model_type in ['RNN','LSTM', 'GRU']
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.cell = eval('nn.{}(input_dim, hid_dim, n_layers, batch_first=True)'.format(model_type))
        
        
    def forward(self, src):
        self.cell.flatten_parameters()
        outputs, hidden = self.cell(src)
        
        
        return outputs, hidden

class DecoderCell(nn.Module):
    def __init__(self, output_dim=6, hid_dim=64, n_layers=4, model_type='RNN'):
        super().__init__()
        assert model_type in ['RNN','LSTM', 'GRU']
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.cell = eval('nn.{}(hid_dim, hid_dim, n_layers, batch_first=True)'.format(model_type))
        
    def forward(self, x, hidden):
        output, hidden = self.cell(x, hidden)
        
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_dim=6, output_dim=6, hid_dim=256, n_layers=4, model_type='RNN', pred_len=10, device='cuda'):
        super().__init__()
        self.device = device
        
        self.pred_len = pred_len
        self.hid_dim = hid_dim

        self.encoder = Encoder(input_dim, hid_dim, n_layers, model_type)
        self.decoder = DecoderCell(output_dim, hid_dim, n_layers, model_type)
        
        self.fc = nn.Linear(hid_dim, output_dim)

              
    def forward(self, x):
        batch_size = x.shape[0]
        enc_outputs, hidden = self.encoder(x)
        
        de_input = enc_outputs[:, -1, :].unsqueeze(1)
        
        outputs = torch.zeros(batch_size, self.pred_len, self.hid_dim, device=self.device)

        for t in range(self.pred_len):
            output, hidden = self.decoder(de_input, hidden)
            outputs[:, t, :] = output.squeeze(1)
            de_input = output
        
        outputs = self.fc(outputs)
       
        return enc_outputs, outputs



class RecurEncoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, cell='lstm'):
        super(RecurEncoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = (
            embedding_dim, 2 * embedding_dim
        )
        self.cell = cell
        
        if self.cell == 'lstm':
            self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )
            self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
            )
        elif self.cell =='gru':
            self.rnn1 = nn.GRU(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )
            self.rnn2 = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
            )
        else:
            assert('Select LSTM or GRU')
    def forward(self, x):
        if self.cell == 'lstm':
            x, (_, _) = self.rnn1(x)
            x, (hidden_n, _) = self.rnn2(x)

        elif self.cell == 'gru':
            x, _ = self.rnn1(x)
            x, hidden_n = self.rnn2(x)
        return  x, x[:,-1,:]


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class RecurDecoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1, cell='lstm'):
        super(RecurDecoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.cell = cell

        if self.cell == 'lstm':
            self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
            )
            self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )
        elif self.cell == 'gru':
            self.rnn1 = nn.GRU(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
            )
            self.rnn2 = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )
        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)
        self.timedist = TimeDistributed(self.output_layer)
        
    def forward(self, x):
        x=x.reshape(-1,1,self.input_dim).repeat(1,self.seq_len,1)      
        if self.cell == 'lstm':
            x, (hidden_n, cell_n) = self.rnn1(x)
            x, (hidden_n, cell_n) = self.rnn2(x)

        elif self.cell == 'gru':
            x, hidden_n = self.rnn1(x)
            x, hidden_n = self.rnn2(x)
        return self.timedist(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, out_n_features, embedding_dim=128, cell='lstm'):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)#.to(device)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)#.to(device)
    def forward(self, x):
        enc_x, x = self.encoder(x)
        x = self.decoder(x)
        return enc_x, x


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length=20):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean), self.latent_mean, self.latent_logvar
        else:
            return self.latent_mean, self.latent_mean, self.latent_logvar


class VariationalRecurDecoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1, cell='lstm'):
        super(VariationalRecurDecoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.cell = cell
        self.latent_length = 20


        self.latent_to_hidden = nn.Linear(self.latent_length, self.input_dim)


        if self.cell == 'lstm':
            self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
            )
            self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )
        elif self.cell == 'gru':
            self.rnn1 = nn.GRU(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
            )
            self.rnn2 = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )

        

        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)
        self.timedist = TimeDistributed(self.output_layer)
        
    def forward(self, latent):
        h_state = self.latent_to_hidden(latent)
        x = h_state.reshape(-1,1,self.input_dim).repeat(1,self.seq_len,1)    
        if self.cell == 'lstm':
            x, (hidden_n, cell_n) = self.rnn1(x)
            x, (hidden_n, cell_n) = self.rnn2(x)

        elif self.cell == 'gru':
            x, hidden_n = self.rnn1(x)
            x, hidden_n = self.rnn2(x)
        return self.timedist(x)



class VariationalRecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, out_n_features, embedding_dim=128, cell='lstm'):
        super(VariationalRecurrentAutoencoder, self).__init__()
        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)#.to(device)
        self.lmbd = Lambda(embedding_dim)
        self.decoder = VariationalRecurDecoder(seq_len, embedding_dim, out_n_features, cell)#.to(device)

    def forward(self, x):
        enc_x, x = self.encoder(x)
        latent_t, latent_mean, latent_logvar = self.lmbd(x)
        dec_x = self.decoder(latent_t)
        return dec_x, latent_mean, latent_logvar



class TransformerModule(nn.Module):
    def __init__(self, d_model, n_head, num_encoder_layers):
        super(TransformerModule, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_head, num_encoder_layers=6)

    def forward(self, res_concat, x):
        enc_out, out = self.transformer(res_concat, x)
        return enc_out, out



class SalTransformer(nn.Module):
    def __init__(self, seq_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, n_head=4, cell='lstm'):
        super(SalTransformer, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)
        
        # Average filter to calculate the saliency
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = n_head
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*2
        self.num_encoder_layers = 6
        self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        
        # Last linear
        self.linear = nn.Linear(self.d_model, out_n_features)

    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])

        res_concat = torch.cat([x, sal_dec_x], dim=2) # Feature-wise concatenation

        # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3

        out = self.linear(out)

        return sal, dec_x, out


class SalAE(nn.Module):
    def __init__(self, seq_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, cell='lstm'):
        super(SalAE, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)
        
        # Average filter to calculate the saliency
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = 4
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*2
        self.num_encoder_layers = 6
        # self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        

        # Just for testing AE
        self.reconmodel_module = RecurrentAutoencoder(seq_len, self.d_model, self.d_model, embedding_dim, cell)

        # Last linear
        self.linear = nn.Linear(self.d_model, out_n_features)

    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])
        res_concat = torch.cat([x, sal_dec_x], dim=2) # Feature-wise concatenation

        # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        # enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3


        # Just for testing AE
        enc_trans_x, out = self.reconmodel_module(res_concat)


        out = self.linear(out)

        return sal, dec_x, out



class SalSCINet(nn.Module):
    def __init__(self, seq_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, cell='lstm'):
        super(SalSCINet, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)
        
        # Average filter to calculate the saliency
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = 4
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*2
        self.num_encoder_layers = 6
        # self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        

        # Just for testing AE
        # self.reconmodel_module = RecurrentAutoencoder(seq_len, self.d_model, self.d_model, embedding_dim, cell)

        SCINet_args = {
            'output_len':10,
            'input_len':seq_len,
            'input_dim':self.d_model,
            'hid_size':1,
            'num_stacks':1,
            'num_levels':2,
            'concat_len':0,
            'groups':1,
            'kernel':3,
            'dropout':0.2,
            'single_step_output_One':0,
            'positionalE':True,
            'modified':True
        }

        self.forecastmodule = SCINet(output_len = SCINet_args['output_len'], input_len= SCINet_args['input_len'], input_dim = SCINet_args['input_dim'], hid_size = SCINet_args['hid_size'], num_stacks = SCINet_args['num_stacks'],
                    num_levels = SCINet_args['num_levels'], concat_len = SCINet_args['concat_len'], groups = SCINet_args['groups'], kernel = SCINet_args['kernel'], dropout = SCINet_args['dropout'],
                        single_step_output_One = SCINet_args['single_step_output_One'], positionalE =  SCINet_args['positionalE'], modified = SCINet_args['modified'])

        # Last linear
        self.linear = nn.Linear(self.d_model, out_n_features)

    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])
        res_concat = torch.cat([x, sal_dec_x], dim=2) # Feature-wise concatenation

        # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        # enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3


        # Just for testing AE
        out = self.forecastmodule(res_concat)


        out = self.linear(out)

        return sal, dec_x, out



class SalGATVAE(nn.Module):
    def __init__(self, seq_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, cell='lstm'):
        super(SalGATVAE, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)
        

        # GAT
        self.feature_gat = FeatureAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)
        self.temporal_gat = TemporalAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)


        # Average filter to calculate the saliency
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = 4
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*4
        self.num_encoder_layers = 6
        # self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        

        self.vae = VariationalRecurrentAutoencoder(seq_len, self.d_model, out_n_features, embedding_dim=embedding_dim, cell=cell)


    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])

        # GAT
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        res_concat = torch.cat([x, sal_dec_x, h_feat, h_temp], dim=2) # Feature-wise concatenation --> (b,n,4k)
        # res_concat = torch.cat([x, h_feat, h_temp], dim=2) # Ablation study without saliency module

        # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        # enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3


        out, latent_mean, latent_logvar = self.vae(res_concat)

        return sal, dec_x, out, latent_mean, latent_logvar


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=False, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=False, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)




class SalGATSCINet(nn.Module):
    def __init__(self, seq_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, cell='lstm'):
        super(SalGATSCINet, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)
        

        # GAT
        self.feature_gat = FeatureAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)
        self.temporal_gat = TemporalAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)


        # Average filter to calculate the saliency
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = 4
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*4
        self.num_encoder_layers = 6
        # self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        

        # Just for testing AE
        # self.reconmodel_module = RecurrentAutoencoder(seq_len, self.d_model, self.d_model, embedding_dim, cell)

        SCINet_args = {
            'output_len':10,
            'input_len':seq_len,
            'input_dim':self.d_model,
            'hid_size':1,
            'num_stacks':1,
            'num_levels':2,
            'concat_len':0,
            'groups':1,
            'kernel':3,
            'dropout':0.2,
            'single_step_output_One':0,
            'positionalE':True,
            'modified':True
        }

        self.forecastmodule = SCINet(output_len = SCINet_args['output_len'], input_len= SCINet_args['input_len'], input_dim = SCINet_args['input_dim'], hid_size = SCINet_args['hid_size'], num_stacks = SCINet_args['num_stacks'],
                    num_levels = SCINet_args['num_levels'], concat_len = SCINet_args['concat_len'], groups = SCINet_args['groups'], kernel = SCINet_args['kernel'], dropout = SCINet_args['dropout'],
                        single_step_output_One = SCINet_args['single_step_output_One'], positionalE =  SCINet_args['positionalE'], modified = SCINet_args['modified'])

        # Last linear
        self.linear = nn.Linear(self.d_model, out_n_features)

    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])

        # GAT
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        res_concat = torch.cat([x, sal_dec_x, h_feat, h_temp], dim=2) # Feature-wise concatenation --> (b,n,4k)
        # res_concat = torch.cat([x, h_feat, h_temp], dim=2) # Ablation study without saliency module

        # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        # enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3


        # Just for testing AE
        out = self.forecastmodule(res_concat)


        out = self.linear(out)

        return sal, dec_x, out



class SalGATSCINetV2(nn.Module):
    def __init__(self, seq_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, cell='lstm'):
        super(SalGATSCINetV2, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)
        

        # GAT
        self.feature_gat = FeatureAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)
        # self.temporal_gat = TemporalAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)


        # Average filter to calculate the saliency
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = 4
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*3
        self.num_encoder_layers = 6
        # self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        

        # Just for testing AE
        # self.reconmodel_module = RecurrentAutoencoder(seq_len, self.d_model, self.d_model, embedding_dim, cell)

        SCINet_args = {
            'output_len':10,
            'input_len':seq_len,
            'input_dim':self.d_model,
            'hid_size':1,
            'num_stacks':1,
            'num_levels':2,
            'concat_len':0,
            'groups':1,
            'kernel':3,
            'dropout':0.2,
            'single_step_output_One':0,
            'positionalE':True,
            'modified':True
        }

        self.forecastmodule = SCINet(output_len = SCINet_args['output_len'], input_len= SCINet_args['input_len'], input_dim = SCINet_args['input_dim'], hid_size = SCINet_args['hid_size'], num_stacks = SCINet_args['num_stacks'],
                    num_levels = SCINet_args['num_levels'], concat_len = SCINet_args['concat_len'], groups = SCINet_args['groups'], kernel = SCINet_args['kernel'], dropout = SCINet_args['dropout'],
                        single_step_output_One = SCINet_args['single_step_output_One'], positionalE =  SCINet_args['positionalE'], modified = SCINet_args['modified'])

        # Last linear
        self.linear = nn.Linear(self.d_model, out_n_features)

    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])

        # GAT
        h_feat = self.feature_gat(x)
        # h_temp = self.temporal_gat(x)
        res_concat = torch.cat([x, sal_dec_x, h_feat], dim=2) # Feature-wise concatenation --> (b,n,3k)

        # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        # enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3


        # Just for testing AE
        out = self.forecastmodule(res_concat)


        out = self.linear(out)

        return sal, dec_x, out


class SalGATConvLSTM(nn.Module):
    def __init__(self, seq_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, cell='lstm'):
        super(SalGATConvLSTM, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)
        

        # GAT
        self.feature_gat = FeatureAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)
        self.temporal_gat = TemporalAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)


        # Average filter to calculate the saliency
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = 4
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*4
        self.num_encoder_layers = 6
        # self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        
        # ConvLSTM input shape = (b, t, c, w, h) --> (batch, window size=100, d_model) = (b, d_model, 1, 10,10)
        self.convlstm = ConvLSTM(input_dim=self.d_model,
                        hidden_dim=[n_features, n_features, n_features],
                        # hidden_dim=[n_features],
                        kernel_size=(3, 3),
                        num_layers=3,
                        batch_first=True,
                        bias=True,
                        return_all_layers=False)

        # Last linear
        self.linear = nn.Linear(self.d_model, out_n_features)

    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])

        # GAT
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        res_concat = torch.cat([x, sal_dec_x, h_feat, h_temp], dim=2) # Feature-wise concatenation --> (b,n,4k)
        # res_concat = torch.cat([x, h_feat, h_temp], dim=2) # Ablation study without saliency module

        # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        # enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3


        # Shape has to be (b, d_model, 1, 10,10)
        res_concat = res_concat.reshape(res_concat.shape[0], res_concat.shape[2], 1, 10, 10)
        print("res_concat:", res_concat.shape)
        out = self.convlstm(res_concat)


        out = self.linear(out)

        return sal, dec_x, out

class SalGATConvGRU(nn.Module):
    def __init__(self, seq_len, output_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, cell='lstm'):
        super(SalGATConvGRU, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)

        # GAT
        self.feature_gat = FeatureAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)
        self.temporal_gat = TemporalAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)


        # Average filter to calculate the saliency
        self.seq_len = seq_len
        self.output_len = output_len
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = 4
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*4
        self.num_encoder_layers = 6
        # self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        
        # # ConvLSTM input shape = (b, t, c, w, h) --> (batch, window size=100=t, d_model) = (b, window size=100=t, self.d_model, 1, 1)
        # self.convlstm = ConvLSTM(input_dim=self.d_model,
        #                 hidden_dim=[n_features, n_features, n_features],
        #                 # hidden_dim=[n_features],
        #                 kernel_size=(3, 3),
        #                 num_layers=3,
        #                 batch_first=True,
        #                 bias=True,
        #                 return_all_layers=False)

        self.convGRU = ConvGRU(input_size=(10, 1), # height, width --> Here, we set to timewindow size = 10*10 = 100
                                input_dim=self.d_model, 
                                hidden_dim=[32, 64],
                                kernel_size=(3,3),
                                num_layers=2,
                                dtype=torch.cuda.FloatTensor,
                                batch_first=True,
                                bias = True,
                                return_all_layers = False) # if True, return length will be equal to number of layers

        # Last linear
        self.linear = nn.Linear(64, out_n_features)

    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])

        # GAT
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        res_concat = torch.cat([x, sal_dec_x, h_feat, h_temp], dim=2) # Feature-wise concatenation --> (b,n,4k)
        # res_concat = torch.cat([x, h_feat, h_temp], dim=2) # Ablation study without saliency module

        # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        # enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3

        # Shape has to be (b, window size=100=t, self.d_model, 1, 1)
        res_concat = res_concat.reshape(res_concat.shape[0], 10, res_concat.shape[2], 10, 1)
        layer_output_list, last_state_list = self.convGRU(res_concat)
        out = layer_output_list[0]
        out = out.reshape(out.shape[0], self.seq_len, out.shape[2])

        out = self.linear(out)

        return sal, dec_x, out[:, self.seq_len-self.output_len:, :]


class SalGATConvGRUwoGAT(nn.Module):
    def __init__(self, seq_len, output_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, cell='lstm'):
        super(SalGATConvGRUwoGAT, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)

        # GAT
        self.feature_gat = FeatureAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)
        self.temporal_gat = TemporalAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)


        # Average filter to calculate the saliency
        self.seq_len = seq_len
        self.output_len = output_len
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = 2
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*3
        self.num_encoder_layers = 6
        # self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        
        # # ConvLSTM input shape = (b, t, c, w, h) --> (batch, window size=100=t, d_model) = (b, window size=100=t, self.d_model, 1, 1)
        # self.convlstm = ConvLSTM(input_dim=self.d_model,
        #                 hidden_dim=[n_features, n_features, n_features],
        #                 # hidden_dim=[n_features],
        #                 kernel_size=(3, 3),
        #                 num_layers=3,
        #                 batch_first=True,
        #                 bias=True,
        #                 return_all_layers=False)

        self.convGRU = ConvGRU(input_size=(10, 1), # height, width --> Here, we set to timewindow size = 10*10 = 100
                                input_dim=self.d_model, 
                                hidden_dim=[32, 64],
                                kernel_size=(3,3),
                                num_layers=2,
                                dtype=torch.cuda.FloatTensor,
                                batch_first=True,
                                bias = True,
                                return_all_layers = False) # if True, return length will be equal to number of layers

        # Last linear
        self.linear = nn.Linear(64, out_n_features)

    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])

        # GAT
        # h_feat = self.feature_gat(x)
        # h_temp = self.temporal_gat(x)
        res_concat = torch.cat([x, sal_dec_x], dim=2) # Feature-wise concatenation --> (b,n,4k)
        # res_concat = torch.cat([x, h_feat, h_temp], dim=2) # Ablation study without saliency module

        # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        # enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3

        # Shape has to be (b, window size=100=t, self.d_model, 1, 1)
        res_concat = res_concat.reshape(res_concat.shape[0], 10, res_concat.shape[2], 10, 1)
        layer_output_list, last_state_list = self.convGRU(res_concat)
        out = layer_output_list[0]
        out = out.reshape(out.shape[0], self.seq_len, out.shape[2])

        out = self.linear(out)

        return sal, dec_x, out[:, self.seq_len-self.output_len:, :]


class SalGATConvGRUwoSal(nn.Module):
    def __init__(self, seq_len, output_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, cell='lstm'):
        super(SalGATConvGRUwoSal, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)

        # GAT
        self.feature_gat = FeatureAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)
        self.temporal_gat = TemporalAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)


        # Average filter to calculate the saliency
        self.seq_len = seq_len
        self.output_len = output_len
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = 4
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*3
        self.num_encoder_layers = 6
        # self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        
        # # ConvLSTM input shape = (b, t, c, w, h) --> (batch, window size=100=t, d_model) = (b, window size=100=t, self.d_model, 1, 1)
        # self.convlstm = ConvLSTM(input_dim=self.d_model,
        #                 hidden_dim=[n_features, n_features, n_features],
        #                 # hidden_dim=[n_features],
        #                 kernel_size=(3, 3),
        #                 num_layers=3,
        #                 batch_first=True,
        #                 bias=True,
        #                 return_all_layers=False)

        self.convGRU = ConvGRU(input_size=(10, 1), # height, width --> Here, we set to timewindow size = 10*10 = 100
                                input_dim=self.d_model, 
                                hidden_dim=[32, 64],
                                kernel_size=(3,3),
                                num_layers=2,
                                dtype=torch.cuda.FloatTensor,
                                batch_first=True,
                                bias = True,
                                return_all_layers = False) # if True, return length will be equal to number of layers

        # Last linear
        self.linear = nn.Linear(64, out_n_features)

    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])

        # GAT
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        res_concat = torch.cat([x, h_feat, h_temp], dim=2) # Feature-wise concatenation --> (b,n,4k)
        # res_concat = torch.cat([x, h_feat, h_temp], dim=2) # Ablation study without saliency module

        # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        # enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3

        # Shape has to be (b, window size=100=t, self.d_model, 1, 1)
        res_concat = res_concat.reshape(res_concat.shape[0], 10, res_concat.shape[2], 10, 1)
        layer_output_list, last_state_list = self.convGRU(res_concat)
        out = layer_output_list[0]
        out = out.reshape(out.shape[0], self.seq_len, out.shape[2])

        out = self.linear(out)

        return sal, dec_x, out[:, self.seq_len-self.output_len:, :]


class SalConvGRUwoALL(nn.Module):
    def __init__(self, seq_len, output_len, n_features, out_n_features, embedding_dim=128, kernel_size=3, cell='lstm'):
        super(SalConvGRUwoALL, self).__init__()

        self.encoder = RecurEncoder(seq_len, n_features, embedding_dim, cell)
        self.decoder = RecurDecoder(seq_len, embedding_dim, out_n_features, cell)
        # self.saliency_ae = RecurrentAutoencoder(seq_len, n_features, out_n_features, embedding_dim, cell)

        # GAT
        self.feature_gat = FeatureAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)
        self.temporal_gat = TemporalAttentionLayer(n_features, seq_len, dropout=0.2, alpha=0.2)


        # Average filter to calculate the saliency
        self.seq_len = seq_len
        self.output_len = output_len
        self.avg_filter = Variable(torch.Tensor(np.ones((1,1,kernel_size,kernel_size))/(kernel_size*kernel_size))).cuda()
        self.kernel_size = kernel_size
        # Transformer parameters
        self.n_head = 4
        # self.d_model = int(n_features * self.n_head)
        self.d_model = n_features*1
        self.num_encoder_layers = 6
        # self.transformer_module = TransformerModule(self.d_model, self.n_head, self.num_encoder_layers)
        
        # # ConvLSTM input shape = (b, t, c, w, h) --> (batch, window size=100=t, d_model) = (b, window size=100=t, self.d_model, 1, 1)
        # self.convlstm = ConvLSTM(input_dim=self.d_model,
        #                 hidden_dim=[n_features, n_features, n_features],
        #                 # hidden_dim=[n_features],
        #                 kernel_size=(3, 3),
        #                 num_layers=3,
        #                 batch_first=True,
        #                 bias=True,
        #                 return_all_layers=False)

        self.convGRU = ConvGRU(input_size=(10, 1), # height, width --> Here, we set to timewindow size = 10*10 = 100
                                input_dim=self.d_model, 
                                hidden_dim=[32, 64],
                                kernel_size=(3,3),
                                num_layers=2,
                                dtype=torch.cuda.FloatTensor,
                                batch_first=True,
                                bias = True,
                                return_all_layers = False) # if True, return length will be equal to number of layers

        # Last linear
        self.linear = nn.Linear(64, out_n_features)

    def forward(self, x):
        enc_x, enc_x_tensor = self.encoder(x)
        enc_x = enc_x.reshape(enc_x.shape[0], 1, enc_x.shape[1], enc_x.shape[2])
        avg_out = F.conv2d(enc_x, self.avg_filter, padding=self.kernel_size//2)
        sal = torch.abs(enc_x - avg_out)
        sal = sal.reshape(sal.shape[0], sal.shape[2], sal.shape[3])
        dec_x = self.decoder(enc_x_tensor)
        sal_dec_x = self.decoder(sal[:, -1, :])

        # GAT
        # h_feat = self.feature_gat(x)
        # h_temp = self.temporal_gat(x)
        res_concat = torch.cat([x], dim=2) # Feature-wise concatenation --> (b,n,4k)

        # # enc_trans_x, out = self.transformer_module(res_concat, x)          # Ablation study 1 --> Not available
        # # enc_trans_x, out = self.transformer_module(res_concat, res_concat) # Ablation study 2
        # # enc_trans_x, out = self.transformer_module(x, x)                   # Ablation study 3

        # # Shape has to be (b, window size=100=t, self.d_model, 1, 1)
        res_concat = res_concat.reshape(res_concat.shape[0], 10, res_concat.shape[2], 10, 1)
        layer_output_list, last_state_list = self.convGRU(res_concat)
        
        out = layer_output_list[0]
        out = out.reshape(out.shape[0], self.seq_len, out.shape[2])

        out = self.linear(out)

        return sal, dec_x, out[:, self.seq_len-self.output_len:, :]



class DNNAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNNAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, int(input_size/2)),
            nn.ReLU(),
            
            nn.Linear(int(input_size/2), int(input_size/4)),
            nn.ReLU(),
            
            nn.Linear(int(input_size/4), int(input_size/8)),
            
        )

        self.decoder = nn.Sequential(
            nn.Linear(int(input_size/8), int(input_size/4)),
            nn.ReLU(),
            
            nn.Linear(int(input_size/4), int(input_size/2)),
            nn.ReLU(),
            
            nn.Linear(int(input_size/2), output_size)
        )
        
    def forward(self, x):
        enc_x = self.encoder(x)
        x = self.decoder(enc_x)
        return enc_x, x
    

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self, in_planes, splitting=True,
                 kernel = 5, dropout=0.5, groups = 1, hidden_size = 1, INN = True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1 #by default: stride==1 
            pad_r = self.dilation * (self.kernel_size) // 2 + 1 #by default: stride==1 

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),

            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))

            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)

            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups , hidden_size, INN):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes = in_planes, splitting=True,
                 kernel = kernel, dropout=dropout, groups = groups, hidden_size = hidden_size, INN = INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)

class LevelSCINet(nn.Module):
    def __init__(self,in_planes, kernel_size, dropout, groups, hidden_size, INN):
        super(LevelSCINet, self).__init__()
        self.interact = InteractorLevel(in_planes= in_planes, kernel = kernel_size, dropout = dropout, groups =groups , hidden_size = hidden_size, INN = INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1) #even: B, T, D odd: B, T, D

class SCINet_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.current_level = current_level


        self.workingblock = LevelSCINet(
            in_planes = in_planes,
            kernel_size = kernel_size,
            dropout = dropout,
            groups= groups,
            hidden_size = hidden_size,
            INN = INN)


        if current_level!=0:
            self.SCINet_Tree_odd=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size, INN)
            self.SCINet_Tree_even=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size, INN)
    
    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2) #L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len: 
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_,0).permute(1,0,2) #B, L, D
        
    def forward(self, x):
        x_even_update, x_odd_update= self.workingblock(x)
        # We recursively reordered these sub-series. You can run the ./utils/recursive_demo.py to emulate this procedure. 
        if self.current_level ==0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))

class EncoderTree(nn.Module):
    def __init__(self, in_planes,  num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.levels=num_levels
        self.SCINet_Tree = SCINet_Tree(
            in_planes = in_planes,
            current_level = num_levels-1,
            kernel_size = kernel_size,
            dropout =dropout ,
            groups = groups,
            hidden_size = hidden_size,
            INN = INN)
        
    def forward(self, x):

        x= self.SCINet_Tree(x)

        return x

class SCINet(nn.Module):
    def __init__(self, output_len, input_len, input_dim = 9, hid_size = 1, num_stacks = 1,
                num_levels = 3, concat_len = 0, groups = 1, kernel = 5, dropout = 0.5,
                 single_step_output_One = 0, input_len_seg = 0, positionalE = False, modified = True, RIN=False):
        super(SCINet, self).__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE
        self.RIN=RIN

        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified)

        if num_stacks == 2: # we only implement two stacks at most.
            self.blocks2 = EncoderTree(
                in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified)

        self.stacks = num_stacks

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        
        if self.single_step_output_One: # only output the N_th timestep.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, 1,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, 1,
                                                kernel_size = 1, bias = False)
        else: # output the N timesteps.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)

        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
    
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))
    
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
    
        return signal

    def forward(self, x):
        assert self.input_len % (np.power(2, self.num_levels)) == 0 # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        ### activated when RIN flag is set ###
        if self.RIN:
            print('/// RIN ACTIVATED ///\r',end='')
            means = x.mean(1, keepdim=True).detach()
            #mean
            x = x - means
            #var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x = x * self.affine_weight + self.affine_bias

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        x = self.projection1(x)

        if self.stacks == 1:
            ### reverse RIN ###
            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x

        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)

            # the second stack
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)
            
            ### Reverse RIN ###
            if self.RIN:
                MidOutPut = MidOutPut - self.affine_bias
                MidOutPut = MidOutPut / (self.affine_weight + 1e-10)
                MidOutPut = MidOutPut * stdev
                MidOutPut = MidOutPut + means

            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x, MidOutPut


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x
