import sys
import os

import numpy as np
import pandas as pd

import torch

import time
import datetime

import pickle

import warnings

from vol_surface_manager.kap_constants import *
from vol_surface_manager.kap_parse_args import parse_args
from vol_surface_manager.kap_logger import acquire_logger
from vol_surface_manager.kap_data_loader import load_data_from_config
from vol_surface_manager.kap_main_data import acquire_daily_param_data, acquire_daily_train_data, \
    acquire_all_point_candidate_scores, compute_candidate_probs

from vol_surface_manager.kap_models import HyeonukLossModel1 as LossModel
from vol_surface_manager.kap_models import HyeonukCandidateModel1 as CandidateModel
from vol_surface_manager.kap_train import HyeonukLossTrainer1 as LossTrainer
from vol_surface_manager.kap_train import HyeonukCandidateTrainer1 as CandidateTrainer

#from vol_surface_manager.kap_models import HyeonukLossModel2 as LossModel
#from vol_surface_manager.kap_models import HyeonukCandidateModel2 as CandidateModel
#from vol_surface_manager.kap_train import HyeonukLossTrainer2 as LossTrainer
#from vol_surface_manager.kap_train import HyeonukCandidateTrainer2 as CandidateTrainer

#from vol_surface_manager.kap_models import HyeonukLossModel3 as LossModel
#from vol_surface_manager.kap_models import HyeonukCandidateModel3 as CandidateModel
#from vol_surface_manager.kap_train import HyeonukLossTrainer3 as LossTrainer
#from vol_surface_manager.kap_train import HyeonukCandidateTrainer3 as CandidateTrainer

#from vol_surface_manager.kap_models import HyeonukLossModel4 as LossModel
#from vol_surface_manager.kap_models import HyeonukCandidateModel4 as CandidateModel
#from vol_surface_manager.kap_train import HyeonukLossTrainer4 as LossTrainer
#from vol_surface_manager.kap_train import HyeonukCandidateTrainer4 as CandidateTrainer

from vol_surface_manager.kap_train import train_loss_model, train_candidate_model


if __name__ == '__main__':
    # ignore by message
    warnings.filterwarnings("ignore", message="overflow encountered in square")
    warnings.filterwarnings("ignore", message="overflow encountered in multiply")
    warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
    warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")

    # 0. Parsing arguments
    config = parse_args()
    logger = acquire_logger(config)

    logger.info('')
    logger.info('Considering %s index. Training %s model.' % (config.DATA_TYPE, config.TRAIN_PARAMETER_TYPE))

    # 1. Acquiring market data
    market_data, raw_param_data, coef_data, all_days_init, coef_days, market_days = load_data_from_config(
        config, logger, mode='train')
    logger.info('Market data is loaded.')

    # 2. Acquiring train data
    all_days = []
    all_data = []
    lengths = []
    for day_index in range(1, len(all_days_init)):
        yesterday = all_days_init[day_index-1]
        day = all_days_init[day_index]
        if not (yesterday in coef_days and day in coef_days):
            continue
        daily_param_data = acquire_daily_param_data(
            day, yesterday, market_data, raw_param_data, coef_data, parameter=config.TRAIN_PARAMETER_TYPE)
        daily_train_data = acquire_daily_train_data(daily_param_data, parameter=config.TRAIN_PARAMETER_TYPE)
        all_days.append(day)
        all_data.append(daily_train_data)
        lengths.append(len(daily_train_data[1]))
    logger.info('Training data is acquired.')

    point_scores_save_path = os.path.join(config.TRAIN_DATA_BASE_DIR, '%s_%s_point_scores.pickle' % (
        config.DATA_TYPE, config.TRAIN_PARAMETER_TYPE))
    try:
        with open(point_scores_save_path, 'rb') as handle:
            all_point_scores = pickle.load(handle)
            logger.debug('%d point scores are loaded.' % len(all_point_scores))
            logger.debug('Example: ' + ', '.join('%.6f' % s for s in all_point_scores[0]))
            logger.info('Using existing file to load point scores.')
    except:
        all_point_scores = acquire_all_point_candidate_scores(logger, all_data, score_count=1000,
                                                              parameter=config.TRAIN_PARAMETER_TYPE)
        with open(point_scores_save_path, 'wb') as handle:
            pickle.dump(all_point_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('Computed point scores are saved for time saving.')
    assert(len(all_point_scores) == len(all_data))

    # 3. Acquire train and test data
    #indices = np.arange(len(all_data))
    #np.random.shuffle(indices)

    # Using pre-defined indices for safe evaluation.
    #if config.DATA_TYPE == 'Kospi200':
    #    indices = np.copy(KOSPI200_INDICES)
    #elif config.DATA_TYPE == 'S&P500':
    #    indices = np.copy(SNP500_INDICES)
    #elif config.DATA_TYPE == 'Eurostoxx50':
    #    indices = np.copy(EUROSTOXX50_INDICES)
    #else:
    #    assert(config.DATA_TYPE == 'HSCEI')
    #    indices = np.copy(HSCEI_INDICES)
    #train_indices = indices[:-len(all_data)//10]
    #test_indices = indices[-len(all_data)//10:]

    indices = np.arange(len(all_data))
    assert(len(all_days) == len(indices))
    train_boundary = sum(day < datetime.datetime.strptime('20190101', '%Y%m%d') for day in all_days)
    train_indices = indices[:train_boundary]
    test_indices = indices[train_boundary:]

    train_data = [all_data[i] for i in train_indices]
    test_data = [all_data[i] for i in test_indices]
    train_point_scores = [all_point_scores[i] for i in train_indices]
    test_point_scores = [all_point_scores[i] for i in test_indices]
    assert(len(train_point_scores) == len(train_data))
    assert(len(test_point_scores) == len(test_data))
    logger.info('Using %d number of train data, %d number of test data.' % (len(train_data), len(test_data)))

    # 4. Acquiring model directories
    logger.debug('')
    logger.debug('Summarizing model directories')
    try:
        assert(os.path.isdir(config.MODEL_BASE_DIR))
    except Exception as e:
        logger.fatal('Invalid model directory: "%s."' % config.MODEL_BASE_DIR)
        raise e
    if config.TRAIN_MODEL_TYPE == 'Loss':
        LOSS_MODEL_DIRNAME = 'loss_models'
        loss_model_base_dir = os.path.join(config.MODEL_BASE_DIR, LOSS_MODEL_DIRNAME)
        if not os.path.exists(loss_model_base_dir):
            os.mkdir(loss_model_base_dir)
        logger.debug(loss_model_base_dir)
        loss_model_base_dir_2 = os.path.join(loss_model_base_dir, config.DATA_TYPE)
        if not os.path.exists(loss_model_base_dir_2):
            os.mkdir(loss_model_base_dir_2)
        logger.debug(loss_model_base_dir_2)
        loss_model_save_dir = os.path.join(loss_model_base_dir_2, config.TRAIN_PARAMETER_TYPE)
        if not os.path.exists(loss_model_save_dir):
            os.mkdir(loss_model_save_dir)
        logger.debug(loss_model_save_dir)
        LOG_DIRNAME = 'logs'
        loss_model_log_dir = os.path.join(loss_model_save_dir, LOG_DIRNAME)
        if not os.path.exists(loss_model_log_dir):
            os.mkdir(loss_model_log_dir)
        logger.debug(loss_model_log_dir)
        loss_model_log_path = os.path.join(loss_model_log_dir, 'training_log_%s.csv' % (
            datetime.datetime.today().strftime('%Y%m%d%H%M%S')))
        logger.debug(loss_model_log_path)

        save_dir = loss_model_save_dir
        log_path = loss_model_log_path
    else:
        assert(config.TRAIN_MODEL_TYPE == 'Candidate')
        CANDIDATE_MODEL_DIRNAME = 'candidate_models'
        candidate_model_base_dir = os.path.join(config.MODEL_BASE_DIR, CANDIDATE_MODEL_DIRNAME)
        if not os.path.exists(candidate_model_base_dir):
            os.mkdir(candidate_model_base_dir)
        logger.debug(candidate_model_base_dir)
        candidate_model_base_dir_2 = os.path.join(candidate_model_base_dir, config.DATA_TYPE)
        if not os.path.exists(candidate_model_base_dir_2):
            os.mkdir(candidate_model_base_dir_2)
        logger.debug(candidate_model_base_dir_2)
        candidate_model_save_dir = os.path.join(candidate_model_base_dir_2, config.TRAIN_PARAMETER_TYPE)
        if not os.path.exists(candidate_model_save_dir):
            os.mkdir(candidate_model_save_dir)
        logger.debug(candidate_model_save_dir)
        LOG_DIRNAME = 'logs'
        candidate_model_log_dir = os.path.join(candidate_model_save_dir, LOG_DIRNAME)
        if not os.path.exists(candidate_model_log_dir):
            os.mkdir(candidate_model_log_dir)
        logger.debug(candidate_model_log_dir)
        candidate_model_log_path = os.path.join(candidate_model_log_dir, 'training_log_%s.csv' % (
            datetime.datetime.today().strftime('%Y%m%d%H%M%S')))
        logger.debug(candidate_model_log_path)

        save_dir = candidate_model_save_dir
        log_path = candidate_model_log_path
    logger.debug('')

    # 5. Constructing model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.TRAIN_MODEL_TYPE == 'Loss':
        model = LossModel(device, parameter=config.TRAIN_PARAMETER_TYPE)
        model.train()
        trainer = LossTrainer(logger, model, device, parameter=config.TRAIN_PARAMETER_TYPE)
    else:
        assert(config.TRAIN_MODEL_TYPE == 'Candidate')
        model = CandidateModel(device, parameter=config.TRAIN_PARAMETER_TYPE)
        model.train()
        trainer = CandidateTrainer(logger, model, device, parameter=config.TRAIN_PARAMETER_TYPE)
    logger.info('Model is constructed.')

    # Acquiring candidate probabilities
    train_candidate_probs = []
    for daily_train_data, point_scores in zip(train_data, train_point_scores):
        candidate_probs, _, _ = compute_candidate_probs(daily_train_data, point_scores,
                                                        point_score_thresh=POINT_SCORE_THRESH)
        train_candidate_probs.append(candidate_probs)
    test_candidate_probs = []
    for daily_train_data, point_scores in zip(test_data, test_point_scores):
        candidate_probs, _, _ = compute_candidate_probs(daily_train_data, point_scores,
                                                        point_score_thresh=POINT_SCORE_THRESH)
        test_candidate_probs.append(candidate_probs)

    # 6. Training
    if config.SAVE_MODELS == 'True':
        save_models = True
    else:
        assert(config.SAVE_MODELS == 'False')
        save_models = False
    if config.TRAIN_MODEL_TYPE == 'Loss':
        train_loss_model(logger, trainer, train_data, test_data,
                         train_candidate_probs, test_candidate_probs,
                         save_dir, save_models, log_path)
    else:
        assert(config.TRAIN_MODEL_TYPE == 'Candidate')
        train_candidate_model(logger, trainer, train_data, test_data,
                              train_point_scores, test_point_scores,
                              save_dir, save_models, log_path)
