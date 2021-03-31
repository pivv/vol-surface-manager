import sys
import os

import numpy as np
import pandas as pd

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

import time
import datetime

import matplotlib
from matplotlib import pyplot as plt

from .kap_constants import *
from .kap_parse_args import parse_args
from .kap_logger import acquire_logger
from .kap_data_loader import load_data_from_config
from .kap_main_data import acquire_daily_param_data, acquire_daily_train_data, \
    acquire_random_loss_batch, acquire_random_candidate_batch, \
    acquire_random_loss_sparse_batch, acquire_random_candidate_sparse_batch

from .kap_models import HyeonukLossModel1 as LossModel
from .kap_models import HyeonukCandidateModel1 as CandidateModel


def get_batches(num_data, batch_size, shuffle=True, remove_last=True):
    indices = np.arange(num_data)
    if shuffle:
        np.random.shuffle(indices)
    if remove_last:
        batches = [indices[i*batch_size:(i+1)*batch_size] for i in range(num_data // batch_size)]
    else:
        batches = [indices[i*batch_size:min((i+1)*batch_size, num_data)] for i in
                   range(int(np.ceil(float(num_data) / batch_size)))]
    return batches


class BaseLossTrainer(ABC):
    def __init__(self, device, parameter):
        self.device = device
        self.parameter = parameter
        self.train_batch_size = 100


class HyeonukLossTrainer1(BaseLossTrainer):
    def __init__(self, logger, model, device, parameter):
        super(HyeonukLossTrainer1, self).__init__(device, parameter)
        self.logger = logger
        self.model = model
        self.num_epoch = 4000
        self.logging_frequency = 20
        self.initial_learning_rate = 0.0001
        self.learning_rate_frequency = 40
        self.save_frequency = 20
        self.train_batch_size = 128
        # self.eval_batch_size = 512
        self.clip_amount = 3.
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()),
                                    betas=(0.9, 0.98), eps=1e-09, lr=self.initial_learning_rate)

    def get_loss_batches(self, train_data, train_candidate_probs, mode='train'):
        x, pos, y, lengths = acquire_random_loss_batch(train_data, train_candidate_probs, parameter=self.parameter)
        if mode == 'train':
            batches = get_batches(len(x), self.train_batch_size, shuffle=True, remove_last=False)
        else:
            assert(mode == 'eval')
            batches = [np.arange(len(x))]
        return x, pos, y, lengths, batches

    def set_lr(self, iepoch, mode='train'):
        if mode == 'train':
            if iepoch % self.learning_rate_frequency == 0:
                lr = self.initial_learning_rate * (iepoch // self.learning_rate_frequency + 1) ** (-0.5)
                self.logger.info('Learning with rate %.8e.' % lr)
                self.logger.info('')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

    def run_epoch(self, iepoch, train_data, train_candidate_probs, mode='train'):
        self.set_lr(iepoch, mode)
        crit = torch.nn.MSELoss(reduction='mean')
        x, pos, y, _, batches = self.get_loss_batches(train_data, train_candidate_probs, mode)

        total_loss = 0.
        total_num = 0
        y_pred = np.zeros_like(y)
        for batch in batches:
            x_batch = torch.FloatTensor(x[batch]).to(self.device)
            pos_batch = torch.LongTensor(pos[batch]).to(self.device)
            y_batch = torch.FloatTensor(y[batch]).to(self.device)

            if mode == 'train':
                self.model.train()
                y_pred_batch = self.model((x_batch, pos_batch))
            else:
                assert(mode == 'eval')
                self.model.eval()
                with torch.no_grad():
                    y_pred_batch = self.model((x_batch, pos_batch))

            loss = crit(y_pred_batch, y_batch)
            if mode == 'train' and len(batch) == self.train_batch_size:
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_amount)
                self.optimizer.step()

            total_loss += loss.item() * len(batch)
            total_num += len(batch)
            y_pred[batch] = y_pred_batch.detach().cpu().numpy()

        return total_loss/total_num, y_pred


class HyeonukLossTrainer2(BaseLossTrainer):
    def __init__(self, logger, model, device, parameter):
        super(HyeonukLossTrainer2, self).__init__(device, parameter)
        self.logger = logger
        self.model = model
        self.num_epoch = 4000
        self.logging_frequency = 20
        self.initial_learning_rate = 0.0001
        self.learning_rate_frequency = 40
        self.save_frequency = 20
        self.train_batch_size = 128
        # self.eval_batch_size = 512
        self.clip_amount = 3.
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()),
                                    betas=(0.9, 0.98), eps=1e-09, lr=self.initial_learning_rate)

    def get_loss_batches(self, train_data, train_candidate_probs, mode='train'):
        x, y, point_locs = acquire_random_loss_sparse_batch(train_data, train_candidate_probs, parameter=self.parameter)
        if mode == 'train':
            batches = get_batches(len(x), self.train_batch_size, shuffle=True, remove_last=False)
        else:
            assert(mode == 'eval')
            batches = [np.arange(len(x))]
        return x, y, point_locs, batches

    def set_lr(self, iepoch, mode='train'):
        if mode == 'train':
            if iepoch % self.learning_rate_frequency == 0:
                lr = self.initial_learning_rate * (iepoch // self.learning_rate_frequency + 1) ** (-0.5)
                self.logger.info('Learning with rate %.8e.' % lr)
                self.logger.info('')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

    def run_epoch(self, iepoch, train_data, train_candidate_probs, mode='train'):
        self.set_lr(iepoch, mode)
        crit = torch.nn.MSELoss(reduction='mean')
        x, y, point_locs, batches = self.get_loss_batches(train_data, train_candidate_probs, mode)

        total_loss = 0.
        total_num = 0
        y_pred = np.zeros_like(y)
        for batch in batches:
            x_batch = torch.FloatTensor(x[batch]).to(self.device)
            y_batch = torch.FloatTensor(y[batch]).to(self.device)
            point_locs_batch = torch.BoolTensor(point_locs[batch]).to(self.device)

            if mode == 'train':
                self.model.train()
                y_pred_batch = self.model((x_batch, point_locs_batch))
            else:
                assert(mode == 'eval')
                self.model.eval()
                with torch.no_grad():
                    y_pred_batch = self.model((x_batch, point_locs_batch))

            loss = crit(y_pred_batch, y_batch)
            if mode == 'train' and len(batch) == self.train_batch_size:
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_amount)
                self.optimizer.step()

            total_loss += loss.item() * len(batch)
            total_num += len(batch)
            y_pred[batch] = y_pred_batch.detach().cpu().numpy()

        return total_loss/total_num, y_pred


HyeonukLossTrainer3 = HyeonukLossTrainer2


class HyeonukLossTrainer4(BaseLossTrainer):  # For SVM
    def __init__(self, logger, model, device, parameter):
        super(HyeonukLossTrainer4, self).__init__(device, parameter)
        self.logger = logger
        self.model = model
        self.num_epoch = 1
        self.logging_frequency = 1
        self.save_frequency = 1

    def get_loss_batches(self, train_data, train_candidate_probs, mode='train'):
        x, y, point_locs = [], [], []
        for i in range(40):
            if (i+1) % 10 == 0:
                print(f"{i+1}'th random batch")
            x_sub, y_sub, point_locs_sub = acquire_random_loss_sparse_batch(
                train_data, train_candidate_probs, parameter=self.parameter)
            x.append(x_sub)
            y.append(y_sub)
            point_locs.append(point_locs_sub)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        point_locs = np.concatenate(point_locs, axis=0)
        batches = [np.arange(len(x))]
        return x, y, point_locs, batches

    def set_lr(self, iepoch, mode='train'):
        pass

    def run_epoch(self, iepoch, train_data, train_candidate_probs, mode='train'):
        self.set_lr(iepoch, mode)
        x, y, point_locs, batches = self.get_loss_batches(train_data, train_candidate_probs, mode)
        self.logger.info('Data acquirement ended.')
        print(x.shape, y.shape, point_locs.shape)

        if mode == 'train':
            self.model.fit(x, y)
        y_pred = self.model((x, point_locs))
        loss = np.mean((y_pred - y)**2)

        return loss, y_pred


class BaseCandidateTrainer(ABC):
    def __init__(self, device, parameter):
        self.device = device
        self.parameter = parameter
        self.train_batch_size = 100


class HyeonukCandidateTrainer1(BaseCandidateTrainer):
    def __init__(self, logger, model, device, parameter):
        super(HyeonukCandidateTrainer1, self).__init__(device, parameter)
        self.logger = logger
        self.model = model
        self.num_epoch = 1000
        self.logging_frequency = 5
        self.initial_learning_rate = 0.0001
        self.learning_rate_frequency = 40
        self.save_frequency = 5
        self.train_batch_size = 128
        # self.eval_batch_size = 512
        self.clip_amount = 3.
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()),
                                    betas=(0.9, 0.98), eps=1e-09, lr=self.initial_learning_rate)

    def get_candidate_batches(self, train_data, train_point_scores, mode='train'):
        x, pos, y, lengths = acquire_random_candidate_batch(train_data, train_point_scores, parameter=self.parameter)
        if mode == 'train':
            batches = get_batches(len(x), self.train_batch_size, shuffle=True, remove_last=False)
        else:
            assert(mode == 'eval')
            batches = [np.arange(len(x))]
        return x, pos, y, lengths, batches

    def set_lr(self, iepoch, mode='train'):
        if mode == 'train':
            if iepoch % self.learning_rate_frequency == 0:
                lr = self.initial_learning_rate * (iepoch // self.learning_rate_frequency + 1) ** (-0.5)
                self.logger.info('Learning with rate %.8e.' % lr)
                self.logger.info('')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

    def run_epoch(self, iepoch, train_data, train_point_scores, mode='train'):
        self.set_lr(iepoch, mode)
        crit = torch.nn.MSELoss(reduction='mean')
        x, pos, y, _, batches = self.get_candidate_batches(train_data, train_point_scores, mode)

        total_loss = 0.
        total_num = 0
        y_pred = np.zeros_like(y)
        for batch in batches:
            x_batch = torch.FloatTensor(x[batch]).to(self.device)
            pos_batch = torch.LongTensor(pos[batch]).to(self.device)
            y_batch = torch.FloatTensor(y[batch]).to(self.device)

            if mode == 'train':
                self.model.train()
                y_pred_batch = self.model((x_batch, pos_batch))
            else:
                assert(mode == 'eval')
                self.model.eval()
                with torch.no_grad():
                    y_pred_batch = self.model((x_batch, pos_batch))

            loss = crit(y_pred_batch[pos_batch != -1], y_batch[pos_batch != -1])
            if mode == 'train' and len(batch) == self.train_batch_size:
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_amount)
                self.optimizer.step()

            total_loss += loss.item() * (pos_batch != -1).sum().item()
            total_num += (pos_batch != -1).sum().item()
            y_pred[batch] = y_pred_batch.detach().cpu().numpy()

        return total_loss/total_num, y_pred


class HyeonukCandidateTrainer2(BaseCandidateTrainer):
    def __init__(self, logger, model, device, parameter):
        super(HyeonukCandidateTrainer2, self).__init__(device, parameter)
        self.logger = logger
        self.model = model
        self.num_epoch = 1000
        self.logging_frequency = 5
        self.initial_learning_rate = 0.0001
        self.learning_rate_frequency = 40
        self.save_frequency = 5
        self.train_batch_size = 128
        # self.eval_batch_size = 512
        self.clip_amount = 3.
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()),
                                    betas=(0.9, 0.98), eps=1e-09, lr=self.initial_learning_rate)

    def get_candidate_batches(self, train_data, train_point_scores, mode='train'):
        x, y, point_locs = acquire_random_candidate_sparse_batch(
            train_data, train_point_scores, parameter=self.parameter)
        if mode == 'train':
            batches = get_batches(len(x), self.train_batch_size, shuffle=True, remove_last=False)
        else:
            assert(mode == 'eval')
            batches = [np.arange(len(x))]
        return x, y, point_locs, batches

    def set_lr(self, iepoch, mode='train'):
        if mode == 'train':
            if iepoch % self.learning_rate_frequency == 0:
                lr = self.initial_learning_rate * (iepoch // self.learning_rate_frequency + 1) ** (-0.5)
                self.logger.info('Learning with rate %.8e.' % lr)
                self.logger.info('')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

    def run_epoch(self, iepoch, train_data, train_point_scores, mode='train'):
        self.set_lr(iepoch, mode)
        crit = torch.nn.MSELoss(reduction='mean')
        x, y, point_locs, batches = self.get_candidate_batches(train_data, train_point_scores, mode)

        total_loss = 0.
        total_num = 0
        y_pred = np.zeros_like(y)
        for batch in batches:
            x_batch = torch.FloatTensor(x[batch]).to(self.device)
            y_batch = torch.FloatTensor(y[batch]).to(self.device)
            point_locs_batch = torch.BoolTensor(point_locs[batch]).to(self.device)

            if mode == 'train':
                self.model.train()
                y_pred_batch = self.model((x_batch, point_locs_batch))
            else:
                assert(mode == 'eval')
                self.model.eval()
                with torch.no_grad():
                    y_pred_batch = self.model((x_batch, point_locs_batch))

            loss = crit(y_pred_batch[point_locs_batch], y_batch[point_locs_batch])
            if mode == 'train' and len(batch) == self.train_batch_size:
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_amount)
                self.optimizer.step()

            total_loss += loss.item() * point_locs_batch.sum().item()
            total_num += point_locs_batch.sum().item()
            y_pred[batch] = y_pred_batch.detach().cpu().numpy()

        return total_loss/total_num, y_pred


HyeonukCandidateTrainer3 = HyeonukCandidateTrainer2


class HyeonukCandidateTrainer4(BaseCandidateTrainer):  # For SVM
    def __init__(self, logger, model, device, parameter):
        super(HyeonukCandidateTrainer4, self).__init__(device, parameter)
        self.logger = logger
        self.model = model
        self.num_epoch = 1
        self.logging_frequency = 1
        self.save_frequency = 1

    def get_candidate_batches(self, train_data, train_point_scores, mode='train'):
        x, y, point_locs = [], [], []
        for _ in range(1):
            x_sub, y_sub, point_locs_sub = acquire_random_candidate_sparse_batch(
                train_data, train_point_scores, parameter=self.parameter)
            x.append(x_sub)
            y.append(y_sub)
            point_locs.append(point_locs_sub)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        point_locs = np.concatenate(point_locs, axis=0)
        batches = [np.arange(len(x))]
        return x, y, point_locs, batches

    def set_lr(self, iepoch, mode='train'):
        pass

    def run_epoch(self, iepoch, train_data, train_point_scores, mode='train'):
        self.set_lr(iepoch, mode)
        x, y, point_locs, batches = self.get_candidate_batches(train_data, train_point_scores, mode)
        self.logger.info('Data acquirement ended.')
        print(x.shape, y.shape, point_locs.shape)

        if mode == 'train':
            self.model.fit(x, y)
        y_pred = self.model((x, point_locs))
        loss = np.mean((y_pred[point_locs] - y[point_locs])**2)

        return loss, y_pred


def train_loss_model(logger, trainer, train_data, test_data,
                     train_candidate_probs, test_candidate_probs,
                     save_dir='', save_models=True, log_path=''):
    logger.info('')
    logger.info('Start training.')
    logger.info('')

    start_time = time.time()
    training_history = []
    losses = []
    val_losses = []
    best_file = 0
    best_val_loss = 1e10
    best_moving_average_val_loss = 1e10
    moving_count = trainer.save_frequency
    for iepoch in range(trainer.num_epoch):
        loss, y_pred = trainer.run_epoch(iepoch, train_data, train_candidate_probs, mode='train')
        losses.append(loss)
        val_loss, y_val_pred = trainer.run_epoch(0, test_data, test_candidate_probs, mode='eval')
        val_losses.append(val_loss)
        elapsed_time = time.time() - start_time

        moving_average_val_loss = float(np.mean(val_losses[-moving_count:]))
        if moving_average_val_loss < best_moving_average_val_loss:
            best_file = iepoch + 1
            best_val_loss = val_loss
            best_moving_average_val_loss = moving_average_val_loss
            # Save best file
            if save_dir:
                trainer.model.save(save_dir, 'model_best')

        if (iepoch + 1) % trainer.save_frequency == 0:
            logger.info("Epoch [%4d/%d]. %3d data. Loss: %.6f. Moving average loss: %.6f. Elapsed time: %.3f seconds." % (
                iepoch + 1, trainer.num_epoch, len(y_pred), loss, float(np.mean(losses[-moving_count:])), elapsed_time))
            logger.debug('Computed predictions: ' + ', '.join('%.6f' % v for v in y_val_pred))
            if save_dir:
                if save_models:
                    trainer.model.save(save_dir, 'model_%d' % (iepoch+1))
                    logger.info("Epoch %d saved. Val loss: %.6f. Moving average val loss: %.6f." % (
                        iepoch + 1, val_loss, moving_average_val_loss))
            logger.info("Best epoch: %d. Val loss: %.6f. Moving average val loss: %.6f." % (
                best_file, best_val_loss, best_moving_average_val_loss))
            logger.info('')

        training_history.append([iepoch + 1, loss, val_loss,
                                 np.mean(losses[-moving_count:]), np.mean(val_losses[-moving_count:]),
                                 elapsed_time])
        if (iepoch + 1) % trainer.logging_frequency == 0:
            if log_path:
                np.savetxt(log_path,
                           np.array(training_history),
                           fmt=['%d', '%.16e', '%.16e', '%.16e', '%.16e', '%.3f'],
                           delimiter=",",
                           header="epoch,loss,val_loss,moving_average_loss,moving_average_val_loss,elapsed_time",
                           comments='')

    logger.info('Training ended.')
    logger.info('')


def train_candidate_model(logger, trainer, train_data, test_data, train_point_scores, test_point_scores,
                          save_dir='', save_models=True, log_path=''):
    logger.info('')
    logger.info('Start training.')
    logger.info('')

    start_time = time.time()
    training_history = []
    best_file = 0
    best_val_loss = 1e10
    for iepoch in range(trainer.num_epoch):
        loss, y_pred = trainer.run_epoch(iepoch, train_data, train_point_scores, mode='train')
        val_loss, y_val_pred = trainer.run_epoch(0, test_data, test_point_scores, mode='eval')
        elapsed_time = time.time() - start_time

        if val_loss < best_val_loss:
            best_file = iepoch + 1
            best_val_loss = val_loss
            # Save best file
            if save_dir:
                trainer.model.save(save_dir, 'model_best')

        if (iepoch + 1) % trainer.save_frequency == 0:
            logger.info(("Epoch [%4d/%d]. %3d data, Loss: %.6f. " +
                         "Positive pred: %3d. Negative pred: %3d. Elapsed time: %.3f seconds.") % (
                iepoch + 1, trainer.num_epoch, len(y_pred), loss,
                np.sum(y_pred > 0), np.sum(y_pred < 0), elapsed_time))
            logger.debug('Computed predictions: ' + ', '.join('%.6f' % v for v in y_val_pred[0]))
            if save_dir:
                if save_models:
                    trainer.model.save(save_dir, 'model_%d' % (iepoch+1))
                    logger.info("Epoch %d saved. Val loss: %.6f." % (iepoch + 1, val_loss))
            logger.info("Best epoch: %d. Val loss: %.6f." % (best_file, best_val_loss))
            logger.info('')

        training_history.append([iepoch + 1, loss, val_loss,
                                 np.sum(y_pred > 0), np.sum(y_pred < 0),
                                 np.sum(y_val_pred > 0), np.sum(y_val_pred < 0),
                                 elapsed_time])
        if (iepoch + 1) % trainer.logging_frequency == 0:
            if log_path:
                np.savetxt(log_path, np.array(training_history),
                           fmt=['%d', '%.16e', '%.16e', '%d', '%d', '%d', '%d', '%.3f'],
                           delimiter=",",
                           header="epoch,loss,val_loss,positive_pred,negative_pred," +
                                  "positive_val_pred,negative_val_pred,elapsed_time",
                           comments='')

    logger.info('Training ended.')
    logger.info('')


def plot_loss_log_data(config, log_datas, save_path='', plot_result=True, parameter='Alpha', model_names=None, ylim=None):
    if isinstance(parameter, list):
        log_datas = log_datas
        parameters = parameter
        ylims = ylim
    else:
        assert(isinstance(parameter, str))
        log_datas = [log_datas]
        parameters = [parameter]
        ylims = [ylim]

    fontsize = 18
    fig = plt.figure(figsize=(7.5*len(parameters), 5*2))
    fig.subplots_adjust(wspace=0.25, hspace=0.3)
    for row in range(2):
        for iparam, (one_log_datas, parameter, ylim) in enumerate(zip(log_datas, parameters, ylims)):
            ax = fig.add_subplot(2, len(parameters), row*len(parameters)+iparam+1)
            if row == 0:
                ax.set_title(r'%s, $\%s$, Train' % (
                    config.DATA_TYPE.replace('&', '\&'), parameter.lower()), fontsize=fontsize)
            else:
                ax.set_title(r'%s, $\%s$, Validation' % (
                    config.DATA_TYPE.replace('&', '\&'), parameter.lower()), fontsize=fontsize)
            ax.set_xlabel('Epoch', fontsize=fontsize)
            ax.set_ylabel('Loss', fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            if row == 0:
                for log_data, model_name in zip(one_log_datas, model_names):
                    ax.plot(log_data['epoch'], log_data['moving_average_loss'],
                             label=f'{model_name}, Training')
            else:
                for log_data, model_name in zip(one_log_datas, model_names):
                    ax.plot(log_data['epoch'], log_data['moving_average_val_loss'],
                            label=f'{model_name}, Eval')
            if ylim is None:
                ax.set_ylim([0, 2.*np.mean(log_data['moving_average_val_loss'][50:100])])
            else:
                ax.set_ylim(ylim)
            ax.legend(fontsize=15)
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    if plot_result:
        plt.show()


def plot_candidate_log_data(config, log_data, save_path='', plot_result=True, parameter='Alpha', model_names=None, ylim=None):
    if isinstance(parameter, list):
        log_datas = log_data
        parameters = parameter
        ylims = ylim
    else:
        assert(isinstance(parameter, str))
        log_datas = [log_data]
        parameters = [parameter]
        ylims = [ylim]

    fontsize = 18
    fig = plt.figure(figsize=(7.5*len(parameters), 5*2))
    fig.subplots_adjust(wspace=0.25, hspace=0.3)
    for row in range(2):
        for iparam, (one_log_datas, parameter, ylim) in enumerate(zip(log_datas, parameters, ylims)):
            ax = fig.add_subplot(2, len(parameters), row*len(parameters)+iparam+1)
            if row == 0:
                ax.set_title(r'%s, $\%s$, Train' % (
                    config.DATA_TYPE.replace('&', '\&'), parameter.lower()), fontsize=fontsize)
            else:
                ax.set_title(r'%s, $\%s$, Validation' % (
                    config.DATA_TYPE.replace('&', '\&'), parameter.lower()), fontsize=fontsize)
            #ax.set_title('%s, %s Candidate model, loss graph.' % (
            #    config.DATA_TYPE.replace('&', '\&'), parameter), fontsize=fontsize)
            ax.set_xlabel('Epoch', fontsize=fontsize)
            ax.set_ylabel('Loss', fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            if row == 0:
                for log_data, model_name in zip(one_log_datas, model_names):
                    ax.plot(log_data['epoch'], log_data['loss'],
                            label=f'{model_name}, Training')
            else:
                for log_data, model_name in zip(one_log_datas, model_names):
                    ax.plot(log_data['epoch'], log_data['val_loss'],
                            label=f'{model_name}, Eval')
            if ylim is None:
                ax.set_ylim([0, 2.*np.mean(log_data['val_loss'][10:20])])
            else:
                ax.set_ylim(ylim)
            ax.legend(fontsize=15)
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    if plot_result:
        plt.show()


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

    # Using pre-defined indices for safe evaluation.
    if config.DATA_TYPE == 'Kospi200':
        indices = np.copy(KOSPI200_INDICES)
    elif config.DATA_TYPE == 'S&P500':
        indices = np.copy(SNP500_INDICES)
    elif config.DATA_TYPE == 'Eurostoxx50':
        indices = np.copy(EUROSTOXX50_INDICES)
    else:
        assert(config.DATA_TYPE == 'HSCEI')
        indices = np.copy(HSCEI_INDICES)

    train_data = [all_data[i] for i in indices[:-len(all_data)//10]]
    test_data = [all_data[i] for i in indices[-len(all_data)//10:]]
    print('Using %d number of train data, %d number of test data.' % (len(train_data), len(test_data)))
    print('')

    # Acquiring model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LossModel(device, parameter=config.TRAIN_PARAMETER_TYPE)
    model.train()
    trainer = HyeonukLossTrainer1(logger, model, device, parameter=config.TRAIN_PARAMETER_TYPE)

    # Acquiring directories
    print('Save directories')
    LOSS_MODEL_DIRNAME = 'loss_models'
    loss_model_base_dir = os.path.join(config.MODEL_BASE_DIR, LOSS_MODEL_DIRNAME)
    if not os.path.exists(loss_model_base_dir):
        os.mkdir(loss_model_base_dir)
    print(loss_model_base_dir)
    loss_model_base_dir_2 = os.path.join(loss_model_base_dir, config.DATA_TYPE)
    if not os.path.exists(loss_model_base_dir_2):
        os.mkdir(loss_model_base_dir_2)
    print(loss_model_base_dir_2)
    loss_model_save_dir = os.path.join(loss_model_base_dir_2, config.TRAIN_PARAMETER_TYPE)
    if not os.path.exists(loss_model_save_dir):
        os.mkdir(loss_model_save_dir)
    print(loss_model_save_dir)
    LOG_DIRNAME = 'logs'
    loss_model_log_dir = os.path.join(loss_model_save_dir, LOG_DIRNAME)
    if not os.path.exists(loss_model_log_dir):
        os.mkdir(loss_model_log_dir)
    print(loss_model_log_dir)
    loss_model_log_path = os.path.join(loss_model_log_dir, 'training_log_%s.csv' % (
        datetime.datetime.today().strftime('%Y%m%d%H%M%S')))
    print(loss_model_log_path)
    print('')

    # Training
    if config.SAVE_MODELS == 'True':
        save_models = True
    else:
        assert(config.SAVE_MODELS == 'False')
        save_models = False
    train_loss_model(logger, trainer, train_data, test_data,
                     loss_model_save_dir, save_models, loss_model_log_path)
