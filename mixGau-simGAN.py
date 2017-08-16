from scipy.stats import norm
import numpy as np
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os

np.random.seed(0)

g_input_size = 1
g_hidden_size = 64
g_output_size = 1

d_input_size = 1
d_hidden_size = 64
d_output_size = 1

minibatch_size = 100

d_learning_rate = 5e-5
g_learning_rate = 2e-4

optim_betas = (0.9, 0.999)

num_epochs = 120000
save_epoch = 1000
sample_size = 2500

mean_A = [-0.7, -0.1, 0.5]
std_A = [0.2, 0.07, 0.13]
num_A = [2000, 5000, 10000]


mean_B = [-0.7, 0.1, 0.7, 0.3]
std_B = [0.03, 0.2, 0.06, 0.04]
num_B = [8000, 2000, 5000, 5000]

sim_ratio = 0.001

sns.set(style="white", palette="muted", color_codes=True)
sns.despine(left=True)

class Generator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Generator, self).__init__()
		self.map1 = nn.Linear(input_size, hidden_size)
		self.map2 = nn.Linear(hidden_size, hidden_size)
		self.map3 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x = F.relu(self.map1(x))
		x = F.sigmoid(self.map2(x))
		return self.map3(x)

class Discriminator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Discriminator, self).__init__()
		self.map1 = nn.Linear(input_size, hidden_size)
		self.map2 = nn.Linear(hidden_size, hidden_size)
		self.map3 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x = F.relu(self.map1(x))
		x = F.relu(self.map2(x))
		return F.sigmoid(self.map3(x))

class Sampler():
	def __init__(self, mean, std, num, n):
		if mean != None:
			self.gauMix = np.random.normal(mean[0], std[0], num[0])
			self.gauMix = np.append(self.gauMix, np.random.normal(mean[1], std[1], num[1]))
			self.gauMix = np.append(self.gauMix, np.random.normal(mean[2], std[2], num[2]))
			if n == 4:
				self.gauMix = np.append(self.gauMix, np.random.normal(mean[3], std[3], num[3]))
			self.gmm = GaussianMixture(n_components=n, covariance_type="full", tol=0.001)
			self.gmm = self.gmm.fit(X=np.expand_dims(self.gauMix, 1))

	def gaussian(self, mu, sigma):
		return lambda m, n: torch.Tensor(np.random.normal(mu, sigma, (m, n)))

	def mixgaussian(self):
		#return lambda m, n: torch.Tensor(np.reshape(self.gmm.sample(m*n)[0], (m, n)))
		return lambda m, n: self.gmm.sample(m*n)[0]

	def uniform(self, low, upper):
		return lambda m, n: torch.Tensor(np.random.uniform(low, upper, (m, n)))

	def beta(slef, alpha, beta):
		return lambda m, n: torch.Tensor(np.random.beta(alpha, beta, (m, n)))

	def gamma(self, k):
		return lambda m, n: torch.Tensor(np.random.gamma(k, (m, n)))

def extract(v):
	return v.data.storage().tolist()

def stats(d):
	return [np.mean(d), np.std(d)]

d_sampler = Sampler(mean_A, std_A, num_A, 3).mixgaussian()
g_sampler = Sampler(mean_B, std_B, num_B, 4).mixgaussian()

data_D = np.reshape(d_sampler(sample_size, 1), (sample_size))
data_G = np.reshape(g_sampler(sample_size, 1), (sample_size))

G = Generator(input_size = g_input_size, hidden_size = g_hidden_size, output_size = g_output_size)
D = Discriminator(input_size = d_input_size, hidden_size = d_hidden_size, output_size = d_output_size)
criterion = nn.BCELoss()
sim_criterion = torch.nn.L1Loss()

d_optimizer = optim.Adam(D.parameters(), lr = d_learning_rate, betas = optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr = g_learning_rate, betas = optim_betas)

for epoch in range(num_epochs):

	np.random.shuffle(data_D)
	np.random.shuffle(data_G)
	
	d_real_data = torch.Tensor(np.random.choice(data_D, (minibatch_size, 1)))
	g_real_data = torch.Tensor(np.random.choice(data_G, (minibatch_size, 1)))#100*1 matrix
	
	d_optimizer.zero_grad()

	#using real data train D
	d_real_data = Variable(d_real_data)
	d_real_decision = D.forward(d_real_data)
	d_real_loss = criterion(d_real_decision, Variable(torch.ones(100, 1)))  # ones = true
	
	#using fake data train D
	d_gen_input = Variable(g_real_data)
	d_fake_data = G.forward(d_gen_input).detach()  # detach to avoid training G on these labels
	d_fake_decision = D.forward(d_fake_data)
	d_fake_loss = criterion(d_fake_decision, Variable(torch.zeros(100, 1)))  # zeros = fake
	
	#backward Loss
	d_loss = (d_real_loss + d_fake_loss) * 0.5
	d_loss.backward()
	d_optimizer.step()

	
	g_optimizer.zero_grad()
	#update Generator
	g_real_data = torch.Tensor(np.random.choice(data_G, (minibatch_size, 1)))
	gen_input = Variable(g_real_data)
	g_fake_data = G.forward(gen_input)
	dg_fake_decision = D.forward(g_fake_data)
	g_loss = criterion(dg_fake_decision, Variable(torch.ones((100, 1))))  # we want to fool, so pretend it's all genuine
	#sim_loss = sim_criterion(g_fake_data, gen_input.detach())

	#loss = sim_ratio * sim_loss + g_loss
	#print(loss)
	#backward Loss
	g_loss.backward()
	g_optimizer.step()  # Only optimizes G's parameters

	if epoch % save_epoch == 0:
		print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,
                                                    extract(d_real_loss)[0],
                                                    extract(d_fake_loss)[0],
                                                    extract(g_loss)[0],
                                                    stats(extract(d_real_data)),
                                                    stats(extract(d_fake_data))))
		
		plt.clf()

		d_fake_data = G.forward(Variable(torch.Tensor(np.reshape(data_G, (sample_size, 1))))).detach()	
		d_fake_data = np.array(extract(d_fake_data))

		lm = sns.distplot(data_G, hist=False, color="g", kde_kws={"shade": True})
		lm = sns.distplot(data_D, hist=False, color="m", kde_kws={"shade": True})
		lm = sns.distplot(d_fake_data, hist=False, color="b", kde_kws={"shade": True})

		plt.axhline(y = np.average(np.array(extract(dg_fake_decision))))

		axes = lm.axes
		axes.set_xlim([-2,4])
		axes.set_ylim([0,3])

		outDir = "./simGAN/mixGau-simGau-4-s/"
		try:
			os.makedirs(outDir)
		except OSError:
			pass
		outDir = outDir + str(epoch) +".png"
		plt.savefig(outDir)



