import torch.nn as nn
import torch
import numpy as np
import torch.distributions as D

from math import pi
from scipy.special import logsumexp
from utils import calculate_matmul, calculate_matmul_n_times

# import tensorflow as tf

class CNN(nn.Module):
    def __init__(self, num_channels=19, contig_len=250, out_dim=2, conf=[]):
        super(CNN, self).__init__()
        
        conv_layers = []
        in_channels = num_channels
        conv_layers_conf = conf[0]
        for layer in conv_layers_conf:
            out_channels, kernel, relu, bn, mp = layer
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel))
            if relu:
                conv_layers.append(nn.ReLU())
            if bn:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            if mp:
                conv_layers.append(nn.MaxPool1d(2))
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*conv_layers)
        fc_layers_conf = conf[1]
        
        num_in = fc_layers_conf[0]
        
        fc_layers = []
        for layer_size in fc_layers_conf[1:]:
            fc_layers.append(nn.Linear(num_in, layer_size))
            fc_layers.append(nn.ReLU())
            num_in = layer_size
        
        fc_layers.append(nn.Linear(num_in, out_dim))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        
#         in_lin, out_lin = 25, 8
#         if kernel == 16:
#             in_lin, out_lin = 17, 8
#         self.conv_layers = nn.Sequential(
#             nn.Conv1d(num_channels, num_channels, kernel),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(num_channels, num_channels * 2, kernel // 2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(num_channels * 2, num_channels * 4, kernel // 2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(num_channels * 4, 1, kernel // 2),
#             nn.ReLU()
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(in_lin, out_lin),
#             nn.ReLU(),
#             nn.Linear(8, 4),
#             nn.ReLU(),
#             nn.Linear(4, out_dim)
#         )

    def forward(self, x):
        # for layer in self.conv_layers:
        #     x = layer(x)
        #     print (x.shape)

        # x = x.view(x.shape[0], -1)
        # for layer in self.fc_layers:
        #     x = layer(x)
        #     print (x.shape)

        # input()

        out = self.conv_layers(x)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out
    

class Coupling(nn.Module):
    def __init__(self, input_shape):
        super(Coupling, self).__init__()
        output_dim = 16
        
        self.t_layers = nn.Sequential(
            nn.Linear(input_shape, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, input_shape)
        )
        
        self.s_layers = nn.Sequential(
            nn.Linear(input_shape, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, input_shape),
            nn.Tanh()
        )

    def forward(self, x):
        t = self.t_layers(x)
        s = self.s_layers(x)
        return s, t

class RealNVP(nn.Module):
    def __init__(self, num_coupling_layers, input_dim):
        super(RealNVP, self).__init__()
        self.num_coupling_layers = num_coupling_layers
        
        self.distribution = D.MultivariateNormal(torch.zeros(input_dim).cuda(), torch.eye(input_dim).cuda())
        self.masks = torch.tensor(np.array(
            [np.concatenate((np.zeros(input_dim // 2), np.ones(input_dim // 2))),
             np.concatenate((np.ones(input_dim // 2), np.zeros(input_dim // 2)))] * (num_coupling_layers // 2)
        )).float().cuda()

        self.loss_tracker = []
        self.layers_list = nn.ModuleList([Coupling(input_dim).cuda() for _ in range(num_coupling_layers)])

    def forward(self, x, dir=-1):
        log_det_inv = 0
        direction = dir
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s = s * reversed_mask
            t = t * reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * torch.exp(direction * s) + direction * t * torch.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * torch.sum(s, dim=1)

        return x, log_det_inv

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -torch.mean(log_likelihood)
    
    def score_samples(self, x):
        y, logdet = self(x, dir=1)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return log_likelihood

    def train_step(self, data):
        self.optimizer.zero_grad()
        loss = self.log_loss(data)
        loss.backward()
        self.optimizer.step()
        self.loss_tracker.append(loss.item())
        return {"loss": np.mean(self.loss_tracker)}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.append(loss.item())
        return {"loss": np.mean(self.loss_tracker)}
    

# class CNNtf(tf.keras.Model):
#   """Defines a multi-layer residual network."""
#   def __init__(self, num_channels=19, contig_len=250, out_dim=2, conf=[]):
#     super().__init__()
#     # Defines class meta data.

#     self.conv_layers = tf.keras.Sequential()
#     self.conv_layers.add(tf.keras.Input(shape=(contig_len, num_channels)))
#     conv_layers_conf = conf[0]
#     for layer in conv_layers_conf:
#         out_channels, kernel, relu, bn, mp = layer
#         self.conv_layers.add(tf.keras.layers.Conv1D(out_channels, kernel))
#         if relu:
#             self.conv_layers.add(tf.keras.layers.ReLU())
#         if bn:
#             self.conv_layers.add(tf.keras.layers.BatchNormalization())
#         if mp:
#             self.conv_layers.add(tf.keras.layers.MaxPool1D(pool_size=2))
            
#     fc_layers_conf = conf[1]
        
#     num_in = fc_layers_conf[0]

#     self.fc_layers = tf.keras.Sequential()
#     for layer_size in fc_layers_conf[1:]:
#         self.fc_layers.add(tf.keras.layers.Dense(layer_size, activation='relu'))
    
#     self.fc_layers.add(tf.keras.layers.Dense(out_dim))
        
#   def call(self, inputs):
#     out = self.conv_layers(inputs)
#     out = tf.keras.layers.Flatten()(out)
#     out = self.fc_layers(out)
#     return out

#   def encode(self, inputs):
#     out = self.conv_layers(inputs)
#     out = tf.keras.layers.Flatten()(out)
#     return out
    
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if div_term.shape[0] != pe.shape[2] // 2:
            pe[:, 0, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class Transformer1d(nn.Module):
    """
    
    Input:
        X: (batch_size, embedding_dim, seq_len)
        
    Output:
        out: (batch_size, n_classes)
        
    Pararmetes:
        
    """

    def __init__(self, n_classes, n_length, n_enc_layers, d_model, nhead, dim_feedforward, dropout, activation, dense_len, verbose=False):
        super(Transformer1d, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.n_length = n_length
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.n_classes = n_classes
        self.verbose = verbose
        self.position_encoder = PositionalEncoding(d_model=d_model, max_len=n_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)
        self.dense = nn.Linear(self.d_model, dense_len)
        
    def forward(self, x):
        
        out = x
        if self.verbose:
            print('input (batch_size, seq_len, embedding_dim)', out.shape)
        
        out = out.permute(1, 0, 2)
        if self.verbose:
            print('transpose (seq_len, batch_size, embedding_dim)', out.shape)
            
        out = self.position_encoder(out)
        if self.verbose:
            print('position_encoder', out.shape)
           
        out = self.transformer_encoder(out)
        if self.verbose:
            print('transformer_encoder', out.shape)
            
        out = out.mean(0)
        if self.verbose:
            print('mean (batch_size, embedding_dim)', out.shape)
        
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
            
        return out
    
    
class Transformer(nn.Module):

    def __init__(self, num_channels=19, seq_len=35, n_classes=2, stride_len=7, n_enc_layers=4, d_model=10, nhead=5, dim_feedforward=128, dropout=0.1, activation='relu', dense_len=32, verbose=False, cuda=True):
        super(Transformer, self).__init__()
        
        self.transformers = []
        for i in range(num_channels):
            self.transformers.append(Transformer1d(n_classes=n_classes, 
                                          n_length=seq_len, 
                                          d_model=d_model, 
                                          nhead=nhead, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout, 
                                          dense_len=dense_len,
                                          n_enc_layers=n_enc_layers,
                                          activation=activation,
                                          verbose=verbose))
            
        self.d_model = d_model
        self.stride_len = stride_len
        
        if cuda:
            for i in range(len(self.transformers)):
                self.transformers[i] = self.transformers[i].cuda()
        
        num_in = dense_len * num_channels
        
        fc_layers = []
        for layer_size in [128, 32]:
            fc_layers.append(nn.Linear(num_in, layer_size))
            fc_layers.append(nn.ReLU())
            num_in = layer_size
        
        fc_layers.append(nn.Linear(num_in, n_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        out = None
        for ch in range(x.shape[1]):
            inp = x[:, ch, :].unfold(1, self.d_model, self.stride_len)
            if out is None:
                out = self.transformers[ch](inp)
            else:
                out = torch.cat((out, self.transformers[ch](inp)), 1)
        output = self.fc_layers(out)
        return output
    

class TransformerS(nn.Module):

    def __init__(self, num_channels=19, seq_len=35, n_classes=2, stride_len=7, n_enc_layers=4, d_model=10, nhead=5, dim_feedforward=128, dropout=0.1, activation='relu', dense_len=32, verbose=False, cuda=True):
        super(TransformerS, self).__init__()
        
        self.transformer = Transformer1d(n_classes=n_classes, 
                                          n_length=seq_len, 
                                          d_model=d_model, 
                                          nhead=nhead, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout, 
                                          dense_len=dense_len,
                                          n_enc_layers=n_enc_layers,
                                          activation=activation,
                                          verbose=verbose)
            
        self.d_model = d_model
        self.stride_len = stride_len
        
        if cuda:
            self.transformer = self.transformer.cuda()
        
        num_in = dense_len
        
        fc_layers = []
        for layer_size in [32]:
            fc_layers.append(nn.Linear(num_in, layer_size))
            fc_layers.append(nn.ReLU())
            num_in = layer_size
        
        fc_layers.append(nn.Linear(num_in, n_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        # Permute for transformer
        x = x.permute(0, 2, 1)
        out = self.transformer(x)
        output = self.fc_layers(out)
        return output
    


class CNN_ns(nn.Module):
    def __init__(self, num_channels=19, kernel=8, out_dim=2):
        super(CNN_ns, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel // 2)
        self.conv3 = nn.Conv1d(num_channels, num_channels * 2, kernel // 2)
        self.conv4 = nn.Conv1d(num_channels * 2, num_channels * 4, kernel // 2)
        if args.depth == 3:
            self.conv4 = nn.Conv1d(num_channels * 2, 1, kernel // 2)
        if args.depth == 4:
            self.conv5 = nn.Conv1d(num_channels * 4, 1, kernel // 4)
        elif args.depth == 5:
            self.conv5 = nn.Conv1d(num_channels * 4, num_channels * 8, kernel // 2)
            self.conv6 = nn.Conv1d(num_channels * 8, 1, kernel // 4)
        if args.depth == 3 and args.kernel == 6:
            in_lin, out_lin = 27, 8
        elif args.depth == 3 and args.kernel == 8:
            in_lin, out_lin = 25, 8
        elif args.depth == 3 and args.kernel == 10:
            in_lin, out_lin = 23, 8
        elif args.depth == 4 and args.kernel == 6:
            in_lin, out_lin = 27, 8
        elif args.depth == 4 and args.kernel == 8:
            in_lin, out_lin = 24, 8
        elif args.depth == 4 and args.kernel == 10:
            in_lin, out_lin = 22, 8
        elif args.depth == 5 and args.kernel == 6:
            in_lin, out_lin = 25, 8
        elif args.depth == 5 and args.kernel == 8:
            in_lin, out_lin = 21, 8
        elif args.depth == 5 and args.kernel == 10:
            in_lin, out_lin = 18, 8

        self.lin1 = nn.Linear(in_lin, out_lin)
        self.lin2 = nn.Linear(8, 4)
        self.lin3 = nn.Linear(4, out_dim)

        self.relu = nn.ReLU(True)
        self.maxp = nn.MaxPool1d(2)

        for lin in [self.conv1, self.conv2, self.conv3, self.conv4, \
                    self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        if args.depth >= 4:
            nn.init.xavier_uniform_(self.conv5.weight)
            nn.init.zeros_(self.conv5.bias)
        if args.depth == 5:
            nn.init.xavier_uniform_(self.conv6.weight)
            nn.init.zeros_(self.conv6.bias)



    def forward(self, x):
        conv_out = self.maxp(self.relu(self.conv1(x)))
#                                 print (conv_out.shape)
        conv_out = self.maxp(self.relu(self.conv2(conv_out)))
#                                 print (conv_out.shape)
        conv_out = self.maxp(self.relu(self.conv3(conv_out)))
#                                 print (conv_out.shape)
        conv_out = self.relu(self.conv4(conv_out))
#                                 print (conv_out.shape)
        if args.depth >= 4:
            conv_out = self.relu(self.conv5(conv_out))
#                                     print (conv_out.shape)
        if args.depth == 5:
            conv_out = self.relu(self.conv6(conv_out))
#                                     print (conv_out.shape)
        conv_out = conv_out.reshape((input.shape[0], -1))
#         print (conv_out.shape)
        out = self.lin3(self.relu(self.lin2(self.relu(self.lin1(conv_out)))))
#         print (out.shape)
#         input()


class CNNs(nn.Module):
    def __init__(self, num_channels=3, kernel=10, out_dim=2):
        super(CNNs, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel)
        self.conv2 = nn.Conv1d(num_channels, num_channels // 2, kernel // 2)
        self.conv3 = nn.Conv1d(num_channels // 2, num_channels // 4, kernel // 2)
        self.lin1 = nn.Linear(57, 16)
        self.lin2 = nn.Linear(16, 8)
        self.lin3 = nn.Linear(8, out_dim)

        self.relu = nn.ReLU(True)
        self.maxp = nn.MaxPool1d(2)

        for lin in [self.conv1, self.conv2, self.conv3, \
                    self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, inputs):
#         print (inputs.shape)
        conv_out = self.relu(self.conv1(inputs))
#         print (conv_out.shape)
        conv_out = self.maxp(self.relu(self.conv2(conv_out)))
#         print (conv_out.shape)
        conv_out = self.maxp(self.relu(self.conv3(conv_out)))
#         print (conv_out.shape)
        conv_out = conv_out.reshape((inputs.shape[0], -1))
#         print (conv_out.shape)
        out = self.lin3(self.relu(self.lin2(self.relu(self.lin1(conv_out)))))
#         print (out.shape)
#         input()
        return out


class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, n_components, n_features, covariance_type="full", eps=1.e-6, init_params="kmeans", mu_init=None, var_init=None):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               torch.Tensor (n, 1, d)
            mu:              torch.Tensor (1, k, d)
            var:             torch.Tensor (1, k, d) or (1, k, d, d)
            pi:              torch.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_features:      int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         torch.Tensor (1, k, d)
            var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self._init_params()


    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(
                    torch.eye(self.n_features).reshape(1, 1, self.n_features, self.n_features).repeat(1, self.n_components, 1, 1),
                    requires_grad=False
                )

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1. / self.n_components)
        self.params_fitted = False


    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x


    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, as_average=False).mean() * n + free_params * np.log(n)

        return bic


    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        if self.init_params == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
                device = self.mu.device
                # When the log-likelihood assumes unbound values, reinitialize model
                self.__init__(self.n_components,
                    self.n_features,
                    covariance_type=self.covariance_type,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps)
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data, = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True


    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))


    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)


    def sample(self, n):
        """
        Samples from the model.
        args:
            n:          int
        returns:
            x:          torch.Tensor (n, d)
            y:          torch.Tensor (n)
        """
        counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.pi.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        # Only iterate over components with non-zero counts
        for k in np.arange(self.n_components)[counts > 0]: 
            if self.covariance_type == "diag":
                x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(self.var[0, k])
            elif self.covariance_type == "full":
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            x = torch.cat((x, x_k), dim=0)

        return x, y


    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = self.check_size(x)

        score = self.__score(x, as_average=False)
        return score


    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        if self.covariance_type == "full":
            mu = self.mu
            var = self.var

            precision = torch.inverse(var)
            d = x.shape[-1]

            log_2pi = d * np.log(2. * pi)

            log_det = self._calculate_log_det(precision)

            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)

            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            return -.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)

            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * prec, dim=2, keepdim=True)
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

            return -.5 * (self.n_features * np.log(2. * pi) + log_p - log_det)


    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """
        log_det = torch.empty(size=(self.n_components,)).to(var.device)
        
        for k in range(self.n_components):
            log_det[k] = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0,k]))).sum()

        return log_det.unsqueeze(-1)


    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp


    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
                            keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps

        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var


    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)


    def __score(self, x, as_average=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)


    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu


    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.size() in [(self.n_components, self.n_features, self.n_features), (1, self.n_components, self.n_features, self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features, self.n_components, self.n_features, self.n_features)

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var


    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [(1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.pi.data = pi


    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            torch.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        """
        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)
        
        min_cost = np.inf

        for i in range(init_times):
            tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return (center.unsqueeze(0)*(x_max - x_min) + x_min)