import os
from pdb import set_trace
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch import nn
from sklearn.decomposition import PCA
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

# training params
batch_size = 50
num_epochs = 50
smoothing_bin_width = 20

# simulate motor unit data
mu = MUsim()
mu.num_units = 32     # default number of units to simulate
mu.num_trials = 200    # default number of trials to simulate
mu.num_bins_per_trial = 1000 # amount of time per trial is (num_bins_per_trial/sample_rate)
mu.MUthresholds_dist="exponential"
mu.MUactivation="sigmoid"
# mu.sample_rate = 1/(0.006)
# create sessions
thresholds, responses = mu.sample_MUs(MUmode="static")
sess0 = mu.simulate_session()
smth_sess0 = mu.convolve(sigma=smoothing_bin_width,target="session")
sig1 = 5*np.sin(np.exp(mu.init_force_profile/2)-np.pi/2)
sig2 = 3*np.sin(mu.init_force_profile-np.pi/2)+3
sig3 = np.exp(mu.init_force_profile)/12
sig4 = 7*np.exp(np.sin(2*np.pi*mu.init_force_profile/mu.init_force_profile.max()-np.pi/2)-1)
sig5 = 7*np.exp(np.sin(np.pi*mu.init_force_profile/mu.init_force_profile.max())-1)
sig6 = np.abs(5*np.sin(2*np.pi*mu.init_force_profile/mu.init_force_profile.max()))
new_force_profile = sig1-sig1.min()
mu.apply_new_force(new_force_profile)
sess1 = mu.simulate_session()
smth_sess1 = mu.convolve(sigma=smoothing_bin_width,target="session")
new_force_profile = sig2-sig2.min()
mu.apply_new_force(new_force_profile)
sess2 = mu.simulate_session()
smth_sess2 = mu.convolve(sigma=smoothing_bin_width,target="session")
new_force_profile = sig3-sig3.min()
mu.apply_new_force(new_force_profile)
sess3 = mu.simulate_session()
smth_sess3 = mu.convolve(sigma=smoothing_bin_width,target="session")
new_force_profile = sig4-sig4.min()
mu.apply_new_force(new_force_profile)
sess4 = mu.simulate_session()
smth_sess4 = mu.convolve(sigma=smoothing_bin_width,target="session")
new_force_profile = sig5-sig5.min()
mu.apply_new_force(new_force_profile)
sess5 = mu.simulate_session()
smth_sess5 = mu.convolve(sigma=smoothing_bin_width,target="session")
new_force_profile = sig6-sig6.min()
mu.apply_new_force(new_force_profile)
sess6 = mu.simulate_session()
smth_sess6 = mu.convolve(sigma=smoothing_bin_width,target="session")
mu.see('force') # plot new applied force
mu.see('curves') # plot unit response curves
mu.see('spikes') # plot spike response
mu.see('smooth') # plot smoothed spike response
mu.see('thresholds') # plot MU thresholds

# session0 = np.transpose(smth_sess0,(2,0,1)) # Make shape: Trials x Time x Neurons
# session1 = np.transpose(smth_sess1,(2,0,1)) # Make shape: Trials x Time x Neurons
# session2 = np.transpose(smth_sess2,(2,0,1)) # Make shape: Trials x Time x Neurons
# torch_sess = map(torch.tensor, session)
latent_dim = 1 #mu.units[1].shape[1]

class GRUEncoder(nn.Module):
    def __init__(self, input_dim=mu.num_units, hidden_dim=64, latent_dim=latent_dim):
        super(GRUEncoder, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.latent_dim=latent_dim
        self.factors = []
        self.Encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                              batch_first=True, bidirectional=False)
        self.FactorLayer = nn.Linear(in_features=hidden_dim, out_features=latent_dim,bias=True)
        self.SELU = nn.SELU()
        self.float()
        
    def forward(self, mu_batch):
        mu_batch = mu_batch.float()
        z, _ = self.Encoder(mu_batch)
        z = self.FactorLayer(z)
        self.factors.append(z)
        z = self.SELU(z)
        return z
    
class GRUDecoder(nn.Module):
    def __init__(self, input_dim=mu.num_units, hidden_dim=64, latent_dim=latent_dim):
        super(GRUDecoder, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.latent_dim=latent_dim
        self.Generator = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.ReconstLayer1 = nn.Linear(in_features=hidden_dim, out_features=input_dim,bias=True)
        self.SELU = nn.SELU()
        self.ReconstLayer2 = nn.Linear(in_features=input_dim, out_features=input_dim,bias=True)
        self.SELU = nn.SELU()
        self.ReconstLayer3 = nn.Linear(in_features=input_dim, out_features=input_dim,bias=True)
        self.SELU = nn.SELU()
        self.float()
        
    def forward(self, latents):
        # SELU_output = self.SELU(latents)
        # full_gen_out = torch.zeros(latents.shape[0],self.input_dim,self.hidden_dim).to(device=device)
        z, _ = self.Generator(latents)
        # z = self.SELU(z)
        z = self.ReconstLayer1(z)
        z = self.SELU(z)
        z = self.ReconstLayer2(z)
        z = self.SELU(z)
        z = self.ReconstLayer3(z)
        z = self.SELU(z)
        return z

def get_models():
    Encoder = GRUEncoder()
    Decoder = GRUDecoder()
    Encoder.to(device=device)
    Decoder.to(device=device)
    for p in Encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in Decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # if Encoder.parameters().dim() > 1:
    #     nn.init.xavier_uniform_(Encoder.parameters())
    # if Decoder.parameters().dim() > 1:
    #     nn.init.xavier_uniform_(Decoder.parameters())
    return (Encoder,Decoder), (optim.Adam(Encoder.parameters()),optim.Adam(Decoder.parameters())), F.mse_loss

def test(batch_iter_test, models, loss_func):
    models[0].eval(); models[1].eval()
    with torch.no_grad():
        for ix,iy in batch_iter_test:
            latents = models[0](ix)
            reconst = models[1](latents)
            test_loss = loss_func(reconst, iy)
    return test_loss

def train(epochs, batch_iter_train, batch_iter_test):
    models, opts, loss_func = get_models()
    train_loss_list = []
    test_loss_list = []
    for iEpoch in range(epochs):
        models[0].train(); models[1].train()
        for ix, iy in batch_iter_train:
            latents = models[0](ix)
            reconst = models[1](latents)
            train_loss = loss_func(reconst, iy)
            
            train_loss.backward()
            opts[0].step()
            opts[0].zero_grad()
            opts[1].step()
            opts[1].zero_grad()
        train_loss_list.append(train_loss.data.item())
        test_loss = test(batch_iter_test, models, loss_func)
        test_loss_list.append(test_loss.data.item())
        print(f'epoch [{iEpoch + 1}/{epochs}], train loss:{train_loss.data.item()}, test loss: {test_loss.data.item()}')
        # test_loss = test(num_epochs, batch_iter_test, models, loss_func)
        # test_loss_list.append(test_loss)
    return models, loss_func, train_loss_list, test_loss_list


# data = TensorDataset(torch.tensor(session1).to(device=device).float(), torch.tensor(session1).to(device=device).float())
# batch_iter = DataLoader(data, batch_size=batch_size)

# models = train(epochs=50,batch_iter=batch_iter)

sess_list = []
for ii in range(len(mu.smooth_session)):
    # Make shape: Trials x Time x Neurons
    sess_list.append(np.transpose(mu.smooth_session[ii],(2,0,1)))

all_MU_data = np.concatenate(sess_list)
shuff_idx = np.arange(all_MU_data.shape[0])
np.random.shuffle(shuff_idx)
all_MU_data_shuff = all_MU_data[shuff_idx,:,:]
separation = int(all_MU_data.shape[0] * 0.8)
train_set = all_MU_data_shuff[:separation]
test_set = all_MU_data_shuff[separation:]

# set up training and test sets
MU_train_tensor = torch.tensor(train_set, device="cuda:0").float()
MU_test_tensor = torch.tensor(test_set, device="cuda:0").float()
MU_train_dataset = TensorDataset(MU_train_tensor, MU_train_tensor)
MU_test_dataset = TensorDataset(MU_test_tensor, MU_test_tensor)
batch_iter_train = DataLoader(MU_train_dataset, batch_size=batch_size)
batch_iter_test = DataLoader(MU_test_dataset, batch_size=len(test_set))

# train GRU autoencoder model
models, loss_func, train_loss_list, test_loss_list = train(num_epochs, batch_iter_train, batch_iter_test)

plt.plot(train_loss_list, color='black')
plt.plot(test_loss_list, color='firebrick')
plt.legend(('train','test'))
plt.title('Training and Testing loss')
plt.show()

# to compare with PCA
pca_cross_cond = PCA(n_components=latent_dim)
pca_cross_cond.fit(np.vstack(train_set))
print(f"Cross-condition PCA score: {pca_cross_cond.explained_variance_ratio_}")
pca_cross_cond_proj = pca_cross_cond.transform(np.vstack(test_set))
pca_cross_cond_proj_trials = pca_cross_cond_proj.reshape(test_set.shape[0],test_set.shape[1],latent_dim)

trial_cnt = 0
num_sessions = 7
for ii in range(num_sessions):
    mu.see('force', session=ii)
    pca_per_cond = PCA(n_components=latent_dim)
    pca_per_cond_proj = pca_per_cond.fit_transform(np.vstack(sess_list[ii]))
    pca_per_cond_proj_trials = pca_per_cond_proj.reshape(sess_list[ii].shape[0],sess_list[ii].shape[1],latent_dim)
    print(f"Per-condition PCA score: {pca_per_cond.explained_variance_ratio_}")
    for jj in shuff_idx[separation:]: # take only from test set
        if ii*mu.num_trials<jj<((ii+1)*mu.num_trials):
            kk = jj % mu.num_trials # modulus for indexing overflow across sessions versus total number of trials
            ll = int(np.where(shuff_idx[separation:]==jj)[0]) # find shuffled array index for the trial index prior to shuffling
            plt.plot(models[0](torch.tensor(
                mu.smooth_session[ii][:,:,kk], device="cuda:0")).cpu().detach().numpy())
            plt.title('Predicted Latent Drive')
            plt.show()
            plt.plot(pca_cross_cond_proj_trials[ll])
            plt.title('Corresponding Trial, Cross-condition PCA projection')
            plt.show()
            plt.plot(pca_per_cond_proj_trials[kk])
            plt.title('Corresponding Trial, Per-condition PCA projection')
            plt.show()
            plt.plot(mu.smooth_session[ii].sum(axis=1)[:,kk])
            plt.title('Corresponding Trial, Pure Summation Across Neurons')
            plt.show()
            force_tensor = torch.tensor(mu.session_forces[ii], device="cuda:0")
            force_tensor = force_tensor[:, None].float
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(1,0,mu.num_units)))
            plt.plot(models[1](force_tensor()).cpu().detach().numpy())
            plt.title('Predicted Smoothed Firing Rates')
            plt.show()
            mu.see('smooth', session=ii, trial=kk, no_offset=True)
            mu.see('spikes', trial=ii*mu.num_trials+kk)
            break # 1st idx matching current session: ii*mu.num_trials<jj<((ii+1)*mu.num_trials

