import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_

        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads

        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()

        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class EEGNet(nn.Module):
    def __init__(self,opt):
        super(EEGNet, self).__init__()
        self.T = opt.time_points

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * 7, 1)

    def forward(self, x,opt):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, opt.dropout_rate)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, opt.dropout_rate)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, opt.dropout_rate)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(-1, 4 * 2 * 7)
        #x = F.sigmoid(self.fc1(x))
        x = F.softmax(self.fc1(x))
        return x

    def encoder(self,x):
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # Layer 2
        x = self.padding1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = self.conv3(x)
        return x


class DomainDiscriminator(nn.Module):
	def __init__(self,opt):
		super(DomainDiscriminator, self).__init__()

		self.rnn_width = opt.rnn_width
		self.linear_width_l = opt.linear_width_l
		self.linear_width = opt.linear_width

		self.grl = GradientReversal(opt.domain_weight)

		width = self.rnn_width
		# if self.acoustic_modality:
		# 	width += self.rnn_width
		# if self.visual_modality:
		# 	width += self.rnn_width
		# if self.lexical_modality:
		# 	width += self.linear_width_l

		self.linear_1 = nn.Linear(width, self.linear_width)
		self.linear_2 = nn.Linear(self.linear_width, 2)
		self.softmax = nn.Softmax(dim=1)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.grl(x)
		x = self.relu(self.linear_1(x))
		x = self.softmax(self.linear_2(x))

		return x

class SpeakerDiscriminator(nn.Module):
	def __init__(self, opt):
		super(SpeakerDiscriminator, self).__init__()

		self.acoustic_modality = opt.acoustic_modality
		self.visual_modality = opt.visual_modality
		self.lexical_modality = opt.lexical_modality

		self.rnn_width = opt.rnn_width
		self.linear_width_l = opt.linear_width_l
		self.linear_width = opt.linear_width

		self.grl = GradientReversal(opt.subject_weight)

		width = 0
		if self.acoustic_modality:
			width += self.rnn_width
		if self.visual_modality:
			width += self.rnn_width
		if self.lexical_modality:
			width += self.linear_width_l
		self.linear_1 = nn.Linear(width, self.linear_width)
		self.linear_2 = nn.Linear(self.linear_width, 22)
		self.softmax = nn.Softmax(dim=1)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.grl(x)
		x = self.relu(self.linear_1(x))
		x = self.softmax(self.linear_2(x))

		return x