import sys
import os

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import joblib

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable

from .transformer.models import PositionwiseEncoder, Encoder, Decoder, \
    Transformer, MultiDecoder, MultiTransformer

from .kap_constants import *
from .kap_parse_args import parse_args
from .kap_logger import acquire_logger
from .kap_data_loader import load_data_from_config
from .kap_main_data import acquire_daily_param_data, acquire_daily_train_data, \
    acquire_random_loss_batch, acquire_random_candidate_batch, normalize_state, \
    normalized_state_to_sparse_state


def conv3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, dim=3, stride=2, zero_init_residual=False,
                 groups=1, width_per_group=64,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.stride = stride

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(dim, self.inplanes, kernel_size=7, stride=stride, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride)
        if stride == 1:
            self.layer4 = self._make_layer(block, 256, layers[3], stride=stride)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=stride)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(dim, stride, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2],
                   dim=dim, stride=stride, **kwargs)


class BaseLossModel(nn.Module, ABC):
    def __init__(self, device, parameter):
        super(BaseLossModel, self).__init__()
        self.device = device
        self.parameter = parameter

    @abstractmethod
    def generate_input(self, data):
        pass

    @abstractmethod
    def forward(self, xs):
        pass

    def predict(self, state):
        with torch.no_grad():
            y_pred = self.forward(self.generate_input(state))
            return y_pred.squeeze(0).detach().cpu().numpy()

    def save(self, save_dir, model_name):
        model_file = os.path.join(save_dir, f'{model_name}.th')
        with open(model_file, 'wb') as f:
            torch.save({'state_dict': self.state_dict()}, f)

    def load(self, save_dir, model_name):
        model_file = os.path.join(save_dir, f'{model_name}.th')
        with open(model_file, 'rb') as f:
            checkpoint = torch.load(f)
            self.load_state_dict(checkpoint['state_dict'])


class HyeonukLossModel1(BaseLossModel):
    def __init__(self, device, parameter):
        super(HyeonukLossModel1, self).__init__(device, parameter)
        (d_src, len_max, n_layers, n_head, d_k, d_v, d_model, d_inner,
         dropout, vector_input, position, sinusoid, relative) = (
            STATE_DIM, MAX_POINT_NUM, 4, 12, 32, 32, 384, 1536, 0.2, True, False, True, False)

        self.encoder = Encoder(d_src, len_max, n_layers, n_head, d_k, d_v, d_model, d_inner,
                               dropout, vector_input, position, sinusoid, relative)

        self.fc = torch.nn.Linear(d_model, 1, bias=True)
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.to(device)
        return

    def generate_input(self, state):
        x_one = normalize_state(state, self.parameter)
        x = torch.FloatTensor(x_one).unsqueeze(0).to(self.device)
        pos = torch.LongTensor(np.arange(len(state))).unsqueeze(0).to(self.device)
        return x, pos

    def forward(self, xs):  # bs x seq_len x d_src, bs x seq_len
        # state
        # tenor param today selected current_curve yesterday_curve
        x, pos = xs
        pos_mask_float = pos.ne(-1).type(torch.float).unsqueeze(-1)  # bs x seq_len x 1
        #  x: bs x seq_len
        x = self.encoder(x, pos, use_mask=False, return_attns=False)  # bs x seq_len x d_model
        x = (x * pos_mask_float).sum(dim=1) / pos_mask_float.sum(dim=1)  # bs x d_model
        x = self.fc(x).squeeze(-1)  # bs
        return x


class HyeonukLossModel2(BaseLossModel):
    def __init__(self, device, parameter):
        super(HyeonukLossModel2, self).__init__(device, parameter)
        self.resnet = resnet18(dim=STATE_DIM-1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 1)
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.to(device)
        return

    def generate_input(self, state):
        x_one, point_locs_one, tenors_one = normalized_state_to_sparse_state(normalize_state(state, self.parameter))
        x = torch.FloatTensor(x_one).unsqueeze(0).to(self.device)
        point_locs = torch.BoolTensor(point_locs_one).unsqueeze(0).to(self.device)
        return x, point_locs

    def forward(self, xs):  # bs x seq_len x d_src
        x, point_locs = xs
        # sparse state
        # param today selected current_curve yesterday_curve
        x = x.permute(0, 2, 1)  # bs x d_src x seq_len
        x = self.resnet(x)  # bs x 512 x ...
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # bs x 1
        x = x.squeeze(-1)  # bs
        return x


class HyeonukLossModel3(BaseLossModel):
    def __init__(self, device, parameter):
        super(HyeonukLossModel3, self).__init__(device, parameter)
        self.fc1 = nn.Linear((STATE_DIM-1) * DAYS_PER_YEAR * 3, 1024)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        self.fc4 = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        self.fc = nn.Linear(1024, 1)
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.to(device)
        return

    def generate_input(self, state):
        x_one, point_locs_one, tenors_one = normalized_state_to_sparse_state(normalize_state(state, self.parameter))
        x = torch.FloatTensor(x_one).unsqueeze(0).to(self.device)
        point_locs = torch.BoolTensor(point_locs_one).unsqueeze(0).to(self.device)
        return x, point_locs

    def forward(self, xs):  # bs x seq_len x d_src
        x, point_locs = xs
        # sparse state
        # param today selected current_curve yesterday_curve
        x = torch.flatten(x, 1)  # bs x (d_src*seq_len)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc(x)  # bs x 1
        x = x.squeeze(-1)  # bs
        return x


class HyeonukLossModel4(BaseLossModel):
    def __init__(self, device, parameter):
        super(HyeonukLossModel4, self).__init__(device, parameter)
        self.svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1, verbose=False)
        return

    def generate_input(self, state):
        x_one, point_locs_one, tenors_one = normalized_state_to_sparse_state(normalize_state(state, self.parameter))
        x = x_one[np.newaxis, :, :]
        point_locs = point_locs_one[np.newaxis, :]
        return x, point_locs

    def predict(self, state):
        y_pred = self.forward(self.generate_input(state))  # bs
        return y_pred[0]

    def fit(self, x, y):
        x = x.reshape(x.shape[0], -1)  # bs x (d_src*seq_len)
        self.svr.fit(x, y)

    def forward(self, xs):  # bs x seq_len x d_src
        x, point_locs = xs
        # sparse state
        # param today selected current_curve yesterday_curve
        x = x.reshape(x.shape[0], -1)  # bs x (d_src*seq_len)
        x = self.svr.predict(x)  # bs
        #print(x.shape)
        return x

    def save(self, save_dir, model_name):
        joblib.dump(self.svr, os.path.join(save_dir, f'{model_name}.pkl'))

    def load(self, save_dir, model_name):
        self.svr = joblib.load(os.path.join(save_dir, f'{model_name}.pkl'))


class BaseCandidateModel(nn.Module, ABC):
    def __init__(self, device, parameter):
        super(BaseCandidateModel, self).__init__()
        self.device = device
        self.parameter = parameter

    @abstractmethod
    def generate_input(self, data):
        pass

    @abstractmethod
    def forward(self, xs):
        pass

    def save(self, save_dir, model_name):
        model_file = os.path.join(save_dir, f'{model_name}.th')
        with open(model_file, 'wb') as f:
            torch.save({'state_dict': self.state_dict()}, f)

    def load(self, save_dir, model_name):
        model_file = os.path.join(save_dir, f'{model_name}.th')
        with open(model_file, 'rb') as f:
            checkpoint = torch.load(f)
            self.load_state_dict(checkpoint['state_dict'])


class HyeonukCandidateModel1(BaseCandidateModel):
    def __init__(self, device, parameter):
        super(HyeonukCandidateModel1, self).__init__(device, parameter)
        (d_src, len_max, n_layers, n_head, d_k, d_v, d_model, d_inner,
         dropout, vector_input, position, sinusoid, relative) = (
            STATE_DIM, MAX_POINT_NUM, 4, 12, 32, 32, 384, 1536, 0.2, True, False, True, False)

        self.encoder = Encoder(d_src, len_max, n_layers, n_head, d_k, d_v, d_model, d_inner,
                               dropout, vector_input, position, sinusoid, relative)

        self.fc = torch.nn.Linear(d_model, 1, bias=True)
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.to(device)
        return

    def generate_input(self, state):
        x_one = normalize_state(state, self.parameter)
        x = torch.FloatTensor(x_one).unsqueeze(0).to(self.device)
        pos = torch.LongTensor(np.arange(len(state))).unsqueeze(0).to(self.device)
        return x, pos

    def predict(self, state):
        with torch.no_grad():
            y_pred = self.forward(self.generate_input(state))
            return y_pred.squeeze(0).detach().cpu().numpy()

    def forward(self, xs):  # bs x seq_len x d_src, bs x seq_len
        # state
        # tenor param today selected current_curve yesterday_curve
        x, pos = xs
        #  x: bs x seq_len
        x = self.encoder(x, pos, use_mask=False, return_attns=False)  # bs x seq_len x d_model
        x = self.fc(x).squeeze(-1)  # bs x seq_len
        x[pos == -1] = 0.
        return x


class HyeonukCandidateModel2(BaseCandidateModel):
    def __init__(self, device, parameter):
        super(HyeonukCandidateModel2, self).__init__(device, parameter)
        self.resnet = resnet18(dim=STATE_DIM-1, stride=1)
        self.fc = nn.Linear(256, 1)
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.to(device)
        return

    def generate_input(self, state):
        x_one, point_locs_one, tenors_one = normalized_state_to_sparse_state(normalize_state(state, self.parameter))
        x = torch.FloatTensor(x_one).unsqueeze(0).to(self.device)
        point_locs = torch.BoolTensor(point_locs_one).unsqueeze(0).to(self.device)
        return x, point_locs, tenors_one

    def predict(self, state):
        with torch.no_grad():
            x, point_locs, tenors_one = self.generate_input(state)
            y_pred = self.forward((x, point_locs))
            return y_pred.squeeze(0).detach().cpu().numpy()[tenors_one-1]

    def forward(self, xs):  # bs x seq_len x d_src
        x, point_locs = xs
        # state
        # param today selected current_curve yesterday_curve
        x = x.permute(0, 2, 1)  # bs x d_src x seq_len
        x = self.resnet(x)  # bs x 512 x seq_len
        x = x.permute(0, 2, 1)  # bs x seq_len x 512
        x = self.fc(x)  # bs x seq_len x 1
        x = x.squeeze(-1)  # bs x seq_len
        x[~point_locs] = 0
        return x


class HyeonukCandidateModel3(BaseCandidateModel):
    def __init__(self, device, parameter):
        super(HyeonukCandidateModel3, self).__init__(device, parameter)
        self.fc1 = nn.Linear((STATE_DIM-1) * DAYS_PER_YEAR * 3, 1024)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        self.fc4 = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        self.fc = nn.Linear(1024, DAYS_PER_YEAR * 3)
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.to(device)
        return

    def generate_input(self, state):
        x_one, point_locs_one, tenors_one = normalized_state_to_sparse_state(normalize_state(state, self.parameter))
        x = torch.FloatTensor(x_one).unsqueeze(0).to(self.device)
        point_locs = torch.BoolTensor(point_locs_one).unsqueeze(0).to(self.device)
        return x, point_locs, tenors_one

    def predict(self, state):
        with torch.no_grad():
            x, point_locs, tenors_one = self.generate_input(state)
            y_pred = self.forward((x, point_locs))
            return y_pred.squeeze(0).detach().cpu().numpy()[tenors_one-1]

    def forward(self, xs):  # bs x seq_len x d_src
        x, point_locs = xs
        # state
        # param today selected current_curve yesterday_curve
        x = torch.flatten(x, 1)  # bs x (d_src*seq_len)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc(x)  # bs x seq_len
        x[~point_locs] = 0
        return x


class HyeonukCandidateModel4(BaseCandidateModel):
    def __init__(self, device, parameter):
        super(HyeonukCandidateModel4, self).__init__(device, parameter)
        self.svr = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1, verbose=False))
        return

    def generate_input(self, state):
        x_one, point_locs_one, tenors_one = normalized_state_to_sparse_state(normalize_state(state, self.parameter))
        x = x_one[np.newaxis, :, :]
        point_locs = point_locs_one[np.newaxis, :]
        return x, point_locs, tenors_one

    def predict(self, state):
        x, point_locs, tenors_one = self.generate_input(state)
        y_pred = self.forward((x, point_locs))
        return y_pred[0, tenors_one-1]

    def fit(self, x, y):
        x = x.reshape(x.shape[0], -1)  # bs x (d_src*seq_len)
        self.svr.fit(x, y)

    def forward(self, xs):  # bs x seq_len x d_src
        x, point_locs = xs
        # sparse state
        # param today selected current_curve yesterday_curve
        x = x.reshape(x.shape[0], -1)  # bs x (d_src*seq_len)
        x = self.svr.predict(x)  # bs x seq_len
        x[~point_locs] = 0
        #print(x.shape)
        return x

    def save(self, save_dir, model_name):
        joblib.dump(self.svr, os.path.join(save_dir, f'{model_name}.pkl'))

    def load(self, save_dir, model_name):
        self.svr = joblib.load(os.path.join(save_dir, f'{model_name}.pkl'))


if __name__ == '__main__':
    # Acquiring market data
    config = parse_args()
    logger = acquire_logger(config)
    market_data, raw_param_data, coef_data, all_days, coef_days, market_days = load_data_from_config(
        config, logger, mode='train')

    # Acquiring train data
    all_data = []
    lengths = []
    for day_index in range(1, len(all_days)):
        yesterday = all_days[day_index-1]
        day = all_days[day_index]
        if not (yesterday in coef_days and day in coef_days):
            continue
        daily_param_data = acquire_daily_param_data(
            day, yesterday, market_data, raw_param_data, coef_data, parameter=config.TRAIN_PARAMETER_TYPE)
        daily_train_data = acquire_daily_train_data(daily_param_data, parameter=config.TRAIN_PARAMETER_TYPE)
        all_data.append(daily_train_data)
        lengths.append(len(daily_train_data[1]))
    indices = np.arange(len(all_data))
    np.random.shuffle(indices)
    print(indices)
    train_data = [all_data[i] for i in indices[:-len(all_data)//10]]
    test_data = [all_data[i] for i in indices[-len(all_data)//10:]]

    # Distribution test for loss model
    x, pos, y, lengths = acquire_random_loss_batch(train_data, parameter=config.TRAIN_PARAMETER_TYPE)

    # Test the loss model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = HyeonukLossModel1(device, parameter=config.TRAIN_PARAMETER_TYPE)
    x_tensor = torch.FloatTensor(x).to(device)
    pos_tensor = torch.LongTensor(pos).to(device)
    # print(test_model.predict(x[0]))
    print(test_model((x_tensor, pos_tensor)))
    print(y)
