import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.mixture import GaussianMixture

import seaborn as sns
import matplotlib.pyplot as plt

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
	def __init__(self, mean, std, num):
		if mean != None:
			self.gauMix = np.random.normal(mean[0], std[0], num[0])
			self.gauMix = np.append(self.gauMix, np.random.normal(mean[1], std[1], num[1]))
			self.gauMix = np.append(self.gauMix, np.random.normal(mean[2], std[2], num[2]))
			self.gmm = GaussianMixture(n_components=3, covariance_type="full", tol=0.001)
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

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
    	#create same size tensor with input on 0/1, depends on label we want
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class Visualize():
	def __init__(self):
		pass

	def saveFig(realDtb, fakeDtb, discriminator, outDir):
		axes = plt.gca()
		axes.set_xlim([-1,10])
		axes.set_ylim([0,0.6])
		axes.set_autoscale_on(False)

		plt.axhline(y = discriminator)
		plt.plot()

		real_mean = np.mean(realDtb)
		real_std = np.std(realDtb)
		real_pdf = norm.pdf(realDtb, real_mean, real_std)
		plt.plot(realDtb, real_pdf)

		fake_mean = np.mean(fakeDtb)
		fake_std = np.std(fakeDtb)
		fake_pdf = norm.pdf(fakeDtb, fake_mean, fake_std)
		plt.plot(fakeDtb, fake_pdf)

		plt.savefig(outDir)

	def view(realDtb, fakeDtb, discriminator, outDir):
		plt.clf()

		axes = plt.gca()
		axes.set_xlim([-1,10])
		axes.set_ylim([0,0.6])
		axes.set_autoscale_on(False)

		plt.axhline(y = discriminator)
		plt.plot()

		real_mean = np.mean(realDtb)
		real_std = np.std(realDtb)
		real_pdf = norm.pdf(realDtb, real_mean, real_std)
		plt.plot(realDtb, real_pdf)

		fake_mean = np.mean(fakeDtb)
		fake_std = np.std(fakeDtb)
		fake_pdf = norm.pdf(fakeDtb, fake_mean, fake_std)
		plt.plot(fakeDtb, fake_pdf)

		plt.pause(0.00001)

data_mean = 2
data_stddev = 1.25

g_input_size = 1
g_hidden_size = 30
g_output_size = 1

d_input_size = 1
d_hidden_size = 30
d_output_size = 1

minibatch_size = 100

d_learning_rate = 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 30000
print_interval = 100

d_steps = 1
g_steps = 1

show_interval = 1000
sample_size = 2500

mean_A = [-0.5, -0.1, 0.2]
std_A = [0.2, 0.07, 0.13]
num_A = [2000, 5000, 10000]

mean_B = [2.1, 2.8, 1.5]
std_B = [0.3, 0.06, 0.1]
num_B = [8000, 2000, 5000]

sns.set(style="white", palette="muted", color_codes=True)
sns.despine(left=True)

class CycleGANModel():
	def __init__(self):
		#initialize network for cycleGAN
		self.netG_A = Generator(input_size = g_input_size, hidden_size = g_hidden_size, output_size = g_output_size)
		self.netG_B = Generator(input_size = g_input_size, hidden_size = g_hidden_size, output_size = g_output_size)
		self.netD_A = Discriminator(input_size = d_input_size, hidden_size = d_hidden_size, output_size = d_output_size)
		self.netD_B = Discriminator(input_size = d_input_size, hidden_size = d_hidden_size, output_size = d_output_size)

		print('---------- Networks initialized -------------')

		#initialize loss function
		self.criterionGAN = GANLoss()
		self.criterionCycle = torch.nn.L1Loss()

		#initialize optimizers
		self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), 
			lr = d_learning_rate, betas = optim_betas)
		self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr = d_learning_rate, betas = optim_betas)
		self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr = d_learning_rate, betas = optim_betas)

	def sample_data(self):
		a_sampler = Sampler(mean_A, std_A, num_A).mixgaussian()
		b_sampler = Sampler(mean_B, std_B, num_B).mixgaussian()

		self.data_A = np.reshape(a_sampler(sample_size, 1), (sample_size))
		self.data_B = np.reshape(b_sampler(sample_size, 1), (sample_size))

	def set_input(self):
		np.random.shuffle(self.data_A)
		np.random.shuffle(self.data_B)
		
		self.input_A = torch.Tensor(np.random.choice(self.data_A, (minibatch_size, 1)))
		self.input_B = torch.Tensor(np.random.choice(self.data_A, (minibatch_size, 1)))#100*1 matrix

	def forward(self):
		self.real_A = Variable(self.input_A)
		self.real_B = Variable(self.input_B)

	def test(self):
		self.real_A = Variable(self.input_A, volatile=True)
		self.fake_B = self.netG_A.forward(self.real_A)
		self.rec_A = self.netG_B.forward(self.fake_B)

		self.real_B = Variable(self.input_B, volatile=True)
		self.fake_A = self.netG_B.forward(self.real_B)
		self.rec_B = self.netG_A.forward(self.fake_A)

	def backward_D_basic(self, netD, real, fake):
		# Real
		pred_real = netD.forward(real)
		loss_D_real = self.criterionGAN(pred_real, True)
		# Fake
		pred_fake = netD.forward(fake.detach())
		loss_D_fake = self.criterionGAN(pred_fake, False)
		# Combined loss
		loss_D = (loss_D_real + loss_D_fake) * 0.5
		# backward
		loss_D.backward()
		return loss_D

	def backward_D_A(self):
		fake_B = self.netG_A.forward(self.real_A)
		self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

	def backward_D_B(self):
		fake_A = self.netG_B.forward(self.real_B)
		self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

	def backward_G(self):
		lambda_A = 10.0
		lambda_B = 10.0

		# Identity loss
		self.loss_idt_A = 0
		self.loss_idt_B = 0
        # GAN loss
        
        # D_A(G_A(A))
		self.fake_B = self.netG_A.forward(self.real_A)
		pred_fake = self.netD_A.forward(self.fake_B)
		self.loss_G_A = self.criterionGAN(pred_fake, True)

        # D_B(G_B(B))
		self.fake_A = self.netG_B.forward(self.real_B)
		pred_fake = self.netD_B.forward(self.fake_A)
		self.loss_G_B = self.criterionGAN(pred_fake, True)

		# Forward cycle loss
		self.rec_A = self.netG_B.forward(self.fake_B)
		self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

		# Backward cycle loss
		self.rec_B = self.netG_A.forward(self.fake_A)
		self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

		# combined loss
		self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
		self.loss_G.backward()

	def optimize_parameters(self):
		# forward
		self.forward() #two sampled A(100*1), B(100*1) have been created

		# G_A and G_B
		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()
		# D_A
		self.optimizer_D_A.zero_grad()
		self.backward_D_A()
		self.optimizer_D_A.step()
		# D_B
		self.optimizer_D_B.zero_grad()
		self.backward_D_B()
		self.optimizer_D_B.step()

	def get_current_errors(self):
		D_A = self.loss_D_A.data[0]
		G_A = self.loss_G_A.data[0]
		Cyc_A = self.loss_cycle_A.data[0]
		D_B = self.loss_D_B.data[0]
		G_B = self.loss_G_B.data[0]
		Cyc_B = self.loss_cycle_B.data[0]

		return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
	                        ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])

	def extract(self, v):
		return v.data.storage().tolist()

	def save_fig(self, epoch):

		plt.clf()

		real_A = Variable(torch.Tensor(np.reshape(self.data_A, (sample_size, 1)))) #100*1 matrix
		real_B = Variable(torch.Tensor(np.reshape(self.data_B, (sample_size, 1)))) #100*1 matrix
		fake_B = self.netG_A.forward(real_A)
		D_B_score = self.netD_A.forward(self.fake_B)

		lm = sns.distplot(np.array(self.extract(real_A)), hist=False, color="g", kde_kws={"shade": True})
		lm = sns.distplot(np.array(self.extract(fake_B)), hist=False, color="m", kde_kws={"shade": True})
		lm = sns.distplot(np.array(self.extract(real_B)), hist=False, color="b", kde_kws={"shade": True})
		plt.axhline(np.average(np.array(self.extract(D_B_score))))

		axes = lm.axes
		axes.set_xlim([-2,4])
		axes.set_ylim([0,3])

		outDir = "/Users/maureen/Documents/Python/simpleGAN/DomainB/shift/" + str(epoch) +".png"
		plt.savefig(outDir)


		plt.clf()

		real_A = Variable(torch.Tensor(np.reshape(self.data_A, (sample_size, 1)))) #100*1 matrix
		real_B = Variable(torch.Tensor(np.reshape(self.data_B, (sample_size, 1)))) #100*1 matrix
		fake_A = self.netG_B.forward(real_B)
		D_A_score = self.netD_B.forward(self.fake_A)

		lm = sns.distplot(np.array(self.extract(real_B)), hist=False, color="g", kde_kws={"shade": True})
		lm = sns.distplot(np.array(self.extract(fake_A)), hist=False, color="m", kde_kws={"shade": True})
		lm = sns.distplot(np.array(self.extract(real_A)), hist=False, color="b", kde_kws={"shade": True})
		plt.axhline(np.average(np.array(self.extract(D_A_score))))

		axes = lm.axes
		axes.set_xlim([-2, 4])
		axes.set_ylim([0,3])

		outDir = "/Users/maureen/Documents/Python/simpleGAN/DomainA/shift/" + str(epoch) +".png"
		plt.savefig(outDir)


		# real_B = Variable(b_sampler(data_size, 1)) #100*1 matrix
		# fake_A = self.netG_B.forward(real_B)
		# rec_B = self.netG_A.forward(fake_A)



####start to train cycle-GAN####
epoch = 30000
model = CycleGANModel()
model.sample_data()

for i in range(0, epoch):
	model.set_input()
	model.optimize_parameters()
	if i % show_interval == 0:
		model.save_fig(i)
		error = model.get_current_errors()
		print(error)












