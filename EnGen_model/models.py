import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EnGen(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, device):
        
        super().__init__()
        
        self.MLP = nn.Sequential()
        
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.device = device

        self.engen_encoder = EnGen_Encoder(
            encoder_layer_sizes, latent_size).to(self.device)
        self.engen_decoder = EnGen_Decoder(
            decoder_layer_sizes, latent_size, self.device).to(self.device)

        # Weights initialization "Xavier" initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 'n' is number of inputs to each neuron
                n = len(m.weight.data[1])
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):

        
        batch_size = x.size(0)
        x.to(self.device)
        z = self.engen_encoder(x)
        recon_x = self.engen_decoder(z)
        return recon_x, z


    def load_ckpt(self, path):
        """Saves current model state dict to path."""
        self.load_state_dict(torch.load(path))

    def save_ckpt(self, path):
        """Saves current model state dict to path."""
        torch.save(self.state_dict(), path)


class EnGen_Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()


        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(0.1, True))

        self.MLP.add_module(name="Z", module=nn.Linear(layer_sizes[-1], latent_size))
        self.MLP.add_module(name="Z_bn", module=nn.BatchNorm1d(latent_size, affine=True))

    def forward(self, x):

        z = self.MLP(x)

        return z



class EnGen_Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, device):

        super().__init__()


        self.MLP = nn.Sequential()
        self.device = device
        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(0.1, True))
            else:
                #last activation
                pass
                # self.MLP.add_module(name="relu", module=nn.ReLU(True))

    def forward(self, z):
        
        x = self.MLP(z)

        return x

# A VAE based EnGen
class EnGenVAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, device):

        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.device = device

        self.encoder = EnGenVAE_Encoder(
            encoder_layer_sizes, latent_size, self.device).to(self.device)

        self.decoder = EnGenVAE_Decoder(
            decoder_layer_sizes, latent_size, self.device).to(self.device)

        # # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 'n' is number of inputs to each neuron
                n = len(m.weight.data[1])
                # "Xavier" initialization
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()


    
    
    def forward(self, x):

        # print(x.is_cuda)
        batch_size = x.size(0)

        means, log_var = self.encoder(x)
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        # print(std.is_cuda)
        # print(eps.is_cuda)
        z = eps * std + means
        # print('z shape training',z.shape)
        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def inference(self, gpu_id, n=1, using_cuda=True, random_state=42):
        Tensor = torch.cuda.FloatTensor if using_cuda else torch.FloatTensor
        batch_size = n
        # z = torch.randn([batch_size, self.latent_size]).to(device)
        np.random.seed(random_state)
        z = torch.randn(batch_size, self.latent_size)
        # print('z shape ',z.shape)
        z = Variable(z.type(Tensor)).cuda(gpu_id, non_blocking=True)
        recon_x = self.decoder(z)
        
        return recon_x

    def load_ckpt(self, path):
        """Saves current model state dict to path."""
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path), strict=False)

    def save_ckpt(self, path):
        """Saves current model state dict to path."""
        torch.save(self.state_dict(), path) 



class EnGenVAE_Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size,device):

        super().__init__()

        self.MLP = nn.Sequential()
        self.device = device
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(0.1, True))

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)


    def forward(self, x):

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars



class EnGenVAE_Decoder(nn.Module):
        
    def __init__(self, layer_sizes, latent_size, device):

        super().__init__()

        self.MLP = nn.Sequential()
        self.device = device
        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(0.1, True))
            else:
                #last activation
                pass
                # self.MLP.add_module(name="relu", module=nn.ReLU())

    def forward(self, z):

        x = self.MLP(z)

        return x
