import os
import matplotlib as plt
import numpy as np
import torch
from torch import optim
from torch import nn
# from torch.nn.modules import dropout
# from torch.nn.modules.loss import NLLLoss
# from torch.random import set_rng_state
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from sklearn.model_selection import train_test_split, KFold
from MUsim import MUsim
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print("Is CUDA available? ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device in use:", device, "GPU:", torch.cuda.get_device_name(0))

# class SequentialAutoEncoder(nn.Module):
#     def __init__(self, input_dim=100, hidden_dim=128, IC_dim=10, factors_dim=40):
#         super(SequentialAutoEncoder, self).__init__()
#         self.EncoderGRU = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=False)
#         self.GeneratorGRU = nn.GRU(input_size=IC_dim, hidden_size=hidden_dim, bidirectional=False)
#         self.LinearToFactors = nn.Linear(, factors_dim)
#         self.LinearToRates = nn.Linear(factors_dim, input_dim)
    
#     def forward(self, mu_batch):
#         mu_batch = mu_batch.float()
#         # import pdb; pdb.set_trace()
#         IC = self.EncoderGRU(mu_batch)
#         genOut = self.GeneratorGRU(IC)
#         factors = self.LinearToFactors(genOut)
#         rates = self.LinearToRates(factors)
#         return torch.exp(rates)

class RNNAutoEncoder(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, IC_dim=10, factors_dim=40):
        super(RNNAutoEncoder, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.IC_dim=IC_dim
        self.factors_dim=factors_dim
        self.Encoder = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.IC_layer = nn.Linear(in_features=hidden_dim, out_features= IC_dim)
        self.Generator = nn.RNN(input_size=IC_dim, hidden_size=hidden_dim, batch_first=True)
        self.GenToFactors = nn.Linear(in_features=hidden_dim, out_features=factors_dim)
        self.FactorsToRates = nn.Linear(in_features=factors_dim, out_features=input_dim)
    
    def forward(self, mu_batch):
        mu_batch = mu_batch.float()
        full_gen_out = torch.zeros(mu_batch.shape[0],mu_batch.shape[1],self.hidden_dim).to(device=device)
        
        _, hidden = self.Encoder(mu_batch)
        IC = self.IC_layer(hidden)
        IC = torch.transpose(IC, 0, 1)
        for i in range(mu_batch.shape[1]):
            output, hidden = self.Generator(IC, hidden)
            full_gen_out[:,i,:] = torch.squeeze(output)
        factors = self.GenToFactors(full_gen_out)
        rates = self.FactorsToRates(factors)
        return rates
    
def get_model():
    model = RNNAutoEncoder()
    model.to(device=device)
    for p in model.parameters():
        pass
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    return model, optim.Adam(model.parameters()), F.poisson_nll_loss

def train(epochs, batch_iter):
    model, opt, loss_func = get_model()
    for iEpoch in range(epochs):
        for ix, iy in batch_iter:
            pred = model(ix).to(device=device)
            loss = loss_func(pred, iy).to(device=device)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        print(f'epoch [{iEpoch + 1}/{epochs}], loss:{loss.data.item()}')
    return model

# simulate motor unit data
mu = MUsim()
mu.num_units = 100     # default number of units to simulate
mu.num_trials = 500    # default number of trials to simulate
mu.num_bins_per_trial = 1000 # amount of time per trial is (num_bins_per_trial/sample_rate)
mu.sample_rate = 1/(0.006)
_, latents = mu.sample_MUs(MUmode="lorenz")
sess = mu.simulate_session()
# smth_sess = mu.convolve(sigma=30,target="session")
# mu.see('rates',session=0)
# mu.see('lorenz')

session = np.transpose(mu.session[0],(2,0,1)) # Make shape: Trials x Time x Neurons
# torch_sess = map(torch.tensor, session)
batch_size = 100

data = TensorDataset(torch.tensor(session).to(device=device), torch.tensor(session).to(device=device))
batch_iter = DataLoader(data, batch_size=batch_size)

model = train(epochs=20,batch_iter=batch_iter)