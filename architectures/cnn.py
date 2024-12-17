import torch
import numpy as np

class convblock(torch.nn.Module):
	def __init__(self, in_channels, out_channels,
			convkernel=2,
			padding=0,
			stride=1,
			dilation=1,
			poolkernel=2,
			poolstride=2,
			):
		super(convblock,self).__init__()
		self.conv =	torch.nn.Conv1d(
				in_channels, out_channels,
				convkernel, padding=padding,
				stride=stride,
				dilation=dilation)
		self.batchnorm = torch.nn.BatchNorm1d(out_channels)
		self.activation = torch.nn.ReLU()
		self.pool = torch.nn.MaxPool1d(poolkernel, poolstride)
		self.block = torch.nn.Sequential(
				self.conv,
				self.batchnorm,
				self.activation,
				self.pool)
	def forward(self, x):
		return self.block(x)

def out_npts(npts, convkernel, poolkernel, poolstride):
	# number of points of output for each convolution block
	conv_length = np.floor((npts+2*0-1*(convkernel-1)-1)/1+1)
	pool_length = np.floor((conv_length+2*0-1*(poolkernel-1)-1)/poolstride+1)
	return int(pool_length)

class CNN(torch.nn.Module):
	def __init__(self, num_layer=6,
			out_classes=2, input_classes=3, npts=5500,
			convkernel=2, poolkernel=2, poolstride=2,
			drop=.3):
		super(CNN, self).__init__()
		self.num_chan = np.power(2, np.arange(9)+2)
		self.num_chan[0] = input_classes
		self.num_chan = self.num_chan[:num_layer+1]
		self.layers = [convblock(i, j, convkernel=convkernel,
				poolkernel=poolkernel, poolstride=poolstride
				) for i,j in zip(self.num_chan[:-1],self.num_chan[1:])]
		self.conv0 = self.layers[0]
		self.conv1 = self.layers[1]
		self.conv2 = self.layers[2]
		self.conv3 = self.layers[3]
		self.conv4 = self.layers[4]
		self.conv5 = self.layers[5]
		for i in range(num_layer):
			npts = out_npts(npts, convkernel, poolkernel, poolstride)
		npts_flattened = npts*self.num_chan[-1]
		self.fc = torch.nn.Linear(npts_flattened, out_classes)
		self.dropout = torch.nn.Dropout(drop)
	def forward(self, x):
		for depth,layer in enumerate(self.layers):
			out = layer(x) if depth==0 else layer(out)
		out = out.view(out.size(0), -1)
		out = self.dropout(out)
		out = self.fc(out)
		return out
