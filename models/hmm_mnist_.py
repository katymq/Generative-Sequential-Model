import numpy as np
import torch
from torch import nn
from scipy.special import logsumexp
import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli


class HMM(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(HMM, self).__init__()
        
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        
        #ReLU [x_dim][x2s_dim]
        self.phi_x = nn.Sequential( nn.Linear( self.x_dim , self.h_dim),
                                  nn.ReLU(),
                                  nn.Linear(self.h_dim, self.h_dim ),
                                  nn.ReLU())
        
        # ReLU [z_dim][z2s_dim]
        self.phi_z = nn.Sequential( nn.Linear(self.z_dim, self.h_dim),
                                  nn.ReLU())
        
        
        # Prior: 
        # transition z_{t-1} to z_t
        # ReLU [rnn_dim] [pz_dim]
        self.prior = nn.Sequential( nn.Linear(self.h_dim, self.h_dim),
                                  nn.ReLU())
        # Linear [p_z_dim][z_dim]
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        #Softplus [p_z_dim][zdim]
        self.prior_std = nn.Sequential( nn.Linear (self.h_dim, self.z_dim), 
                                    nn.Softplus())
        
        
        
        # Encoder 
        # z|x  inference model ANalizar si multiplico por el numero de layers
        # ReLU [x2s_dim, rnn_dim] [q_z_dim]
        self.enc = nn.Sequential( nn.Linear(self.h_dim, self.h_dim),
                                nn.ReLU(),
                                nn.Linear(self.h_dim, self.h_dim),
                                nn.ReLU())
        # Linear [q_z_dim][z_dim]
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        
        # Softplus [q_z_dim][z_dim]
        self.enc_std = nn.Sequential( nn.Linear (self.h_dim, self.z_dim), 
                                    nn.Softplus())
        
        # Decoder
        # x|z
        # ReLU [z2s_dim, rnn_dim][p_x_dim]
        self.dec = nn.Sequential( nn.Linear(self.h_dim + self.h_dim , self.h_dim ),
                                nn.ReLU(),
                                nn.Linear(self.h_dim , self.h_dim ),
                                nn.ReLU())
        self.dec_mean = nn.Sequential(nn.Linear(self.h_dim, self.h_dim ),
                                      nn.ReLU(),
                                      nn.Linear(self.h_dim, self.x_dim),
                                      nn.Sigmoid())
        self.dec_std = nn.Sequential( nn.Linear (self.h_dim, self.x_dim), 
                                     nn.Softmax())       
        
        # Recurrence
        # Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        self.rnn = nn.GRU( h_dim, h_dim, n_layers,  bias)
    
    
    def encode (self , h):
        phi_x_t = h[-1]
        enc_t = self.enc(phi_x_t)
        enc_mean_t = self.enc_mean(enc_t)
        enc_std_t = self.enc_std(enc_t)
        return enc_mean_t , enc_std_t
    
    def decode (self, phi_z_t , h):
        phi_z_t = torch.cat( [ phi_z_t , h[-1] ] , 1)
        dec_t = self.dec(phi_z_t)
        dec_mean_t = self.dec_mean(dec_t)
        dec_std_t = self.dec_std(dec_t)
        return dec_mean_t , dec_std_t
    
      
    def forward(self, x):
        # Initialization
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        # KL in elbo
        # -loglikehooh in ELBO
        kld_loss = 0
        nll_loss = 0
        # hidden state
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim)
        z_t_sampled = []
        
        for t in range(x.size(0)):
            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            
            #encoder
            enc_mean_t , enc_std_t = self.encode(h)
            
            #samplig and reparametrerization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)
            z_t_sampled.append(z_t)   
            
            #decoder
            dec_mean_t , dec_std_t = self.decode(phi_z_t , h)
            
            #dec_std_t2 = self.dec_std2(dec_t)
            
            #recurrence
            _, h = self.rnn(phi_z_t.unsqueeze(0), h)
            
            # Computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            nll_loss += self._nll_ber(dec_mean_t, dec_std_t, x[t])
            
            
            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)
            
        return kld_loss, nll_loss, dec_mean_t, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std)
        
    def log_likelihood_(self, x, n_samples):       
        # hidden state
        # h initialize n_samples
        h_ = torch.zeros(n_samples, self.n_layers, x.size(1), self.h_dim)
        resample_index = torch.tensor(np.arange(0,n_samples))
        log_px = []
        
        for t in range(x.size(0)):
            logp_decoder = torch.zeros(n_samples, 1)
            logp_prior = torch.zeros(n_samples, 1)
            logp_encoder = torch.zeros(n_samples, 1)
            
            
            for i in range(n_samples):
                #prior
                h = h_[resample_index[i]]
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)
                p_prior = Norm.Normal(prior_mean_t, prior_std_t**2)
               
                #encoder
                enc_mean_t , enc_std_t = self.encode(h)
                p_encoder = Norm.Normal(enc_mean_t, enc_std_t**2)
            
                #z sampling
                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
                phi_z_t = self.phi_z(z_t)
                
                #decoder
                dec_mean_t , dec_std_t = self.decode(phi_z_t , h)

                #recurrence
                _, h = self.rnn(phi_z_t.unsqueeze(0), h) 
                
                # Probabilities
                logp_decoder[i] = -F.binary_cross_entropy(dec_mean_t, x[t], reduction='sum')
		#torch.sum(x[t]*torch.log(dec_mean_t) + (1-x[t])*torch.log(1-dec_mean_t))
                #-torch.sum(F.binary_cross_entropy(dec_mean_t, x[t], reduction='none'))
                #p_decoder.log_prob(x[t]).sum().item()
                logp_encoder[i] = p_encoder.log_prob(z_t).sum().item()
                logp_prior[i] = p_prior.log_prob(z_t).sum().item()
                
                h_[i] = h
                
                
            log_w =  logp_prior + logp_decoder - logp_encoder  
            w_norm = torch.exp(log_w - logsumexp(log_w.detach().numpy()))
            
            resample_index = torch.multinomial(w_norm.squeeze(1), n_samples, replacement = True)

            
            log_px_t = logsumexp(log_w.detach().numpy()).item() - np.log(n_samples)
            
            log_px.append(log_px_t)
            
        return log_px

      
    def reset_parameters(self, stdv = 0.1):
        for weight in self.parameters():
            #weight.normal_(0, stdv)
            weight.data.normal_(0, stdv)
            
    # Extra functions 
    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        eps = torch.FloatTensor(std.size()).normal_()
        #eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        norm_dis2 = Norm.Normal(mean_2, std_2)
        norm_dis1 = Norm.Normal(mean_1, std_1)
        kl_loss = torch.sum(KL.kl_divergence(norm_dis1, norm_dis2))
        return    kl_loss


    # def _kld_gauss2(self, mean_1, std_1, mean_2, std_2):
    #     kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) +
    #         (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
    #         std_2.pow(2) - 1)
    #     return    0.5 * torch.sum(kld_element)


    def _nll_gauss(self, mean, std, x):
        v = torch.log(std) + (x-mean).pow(2)/(2*std.pow(2))
        return torch.sum(torch.add(v , 0.5* np.log(2*np.pi))) 
    
    def _nll_ber(self, mean, std, x):
        nll_loss = torch.sum(F.binary_cross_entropy(mean, x, reduction='none'))
        return nll_loss  
    
    
    def _concatenate (self , a , b) :
        for i in range(len(b)):
            a = torch.cat([a , b[i]] , 1)
        return a  

    def sample(self, seq_len, device):
        
        sample = []
        h = torch.zeros(self.n_layers, 1, self.h_dim) 
        
        for t in range(seq_len):
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)
            
            #decoder
            dec_mean_t , dec_std_t = self.decode(phi_z_t , h)
            l_x_t = Bernoulli(dec_mean_t)
            x_t = l_x_t.sample() 
            #recurrence
            _, h = self.rnn(phi_z_t.unsqueeze(0), h)            
            sample.append(x_t.detach().numpy())

        return sample
                                
                                    