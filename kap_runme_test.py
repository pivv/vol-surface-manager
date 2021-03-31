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
    acquire_all_point_candidate_scores, \
    select_daily_train_data, randomly_select_daily_train_data, plot_daily_train_data

from vol_surface_manager.kap_models import HyeonukLossModel1 as LossModel
from vol_surface_manager.kap_models import HyeonukCandidateModel1 as CandidateModel


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
    logger.info('Considering %s index. Testing %s model.' % (config.DATA_TYPE, config.TRAIN_PARAMETER_TYPE))

    # 1. Acquiring market data
    market_data, raw_param_data, coef_data, all_days, coef_days, market_days = load_data_from_config(
        config, logger, mode='train')
    logger.info('Market data is loaded.')

    # 2. Acquiring train data
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
    logger.info('Training data is acquired.')

    all_point_scores = None
    if config.TRAIN_MODEL_TYPE == 'Candidate':
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
    train_point_scores, test_point_scores = None, None
    if config.TRAIN_MODEL_TYPE == 'Candidate':
        train_point_scores = [all_point_scores[i] for i in indices[:-len(all_data) // 10]]
        test_point_scores = [all_point_scores[i] for i in indices[-len(all_data) // 10:]]
    logger.info('Using %d number of train data, %d number of test data.' % (len(train_data), len(test_data)))

    ### TEST
    temp_days = []
    for index in range(len(test_data)):
        day = test_data[index][0]
        temp_days.append(day.strftime('%Y%m%d'))
    print(temp_days)
    assert(0 == 1)
    ### TEST

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
        logger.debug(loss_model_base_dir)
        loss_model_base_dir_2 = os.path.join(loss_model_base_dir, config.DATA_TYPE)
        logger.debug(loss_model_base_dir_2)
        loss_model_save_dir = os.path.join(loss_model_base_dir_2, config.TRAIN_PARAMETER_TYPE)
        logger.debug(loss_model_save_dir)

        save_dir = loss_model_save_dir
    else:
        assert(config.TRAIN_MODEL_TYPE == 'Candidate')
        CANDIDATE_MODEL_DIRNAME = 'candidate_models'
        candidate_model_base_dir = os.path.join(config.MODEL_BASE_DIR, CANDIDATE_MODEL_DIRNAME)
        logger.debug(candidate_model_base_dir)
        candidate_model_base_dir_2 = os.path.join(candidate_model_base_dir, config.DATA_TYPE)
        logger.debug(candidate_model_base_dir_2)
        candidate_model_save_dir = os.path.join(candidate_model_base_dir_2, config.TRAIN_PARAMETER_TYPE)
        logger.debug(candidate_model_save_dir)

        save_dir = candidate_model_save_dir
    logger.debug('')

    # 5. Constructing model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.TRAIN_MODEL_TYPE == 'Loss':
        model = LossModel(device, parameter=config.TRAIN_PARAMETER_TYPE)
        model.eval()
    else:
        assert(config.TRAIN_MODEL_TYPE == 'Candidate')
        model = CandidateModel(device, parameter=config.TRAIN_PARAMETER_TYPE)
        model.eval()
    logger.info('Model is constructed.')

    # 6. Loading model
    if config.TEST_MODEL_INDEX == -1:  # Using best model.
        model_name = 'model_best'
    else:
        model_name = 'model_%d' % config.TEST_MODEL_INDEX
    logger.info('Using model "%s" to test.' % model_name)

    try:
        model.load(save_dir, model_name)
        logger.info('Model is loaded.')
    except Exception as e:
        logger.fatal('Cannot load model from path "%s".' % os.path.join(save_dir, model_name))
        raise e

    # 7. Testing model
    logger.info('')
    logger.info('Start testing.')
    logger.info('')
    if config.TRAIN_MODEL_TYPE == 'Loss':
        if config.TRAIN_PARAMETER_TYPE == 'Alpha':
            overflow_thresh = ALPHA_OVERFLOW_THRESH
        elif config.TRAIN_PARAMETER_TYPE == 'Rho':
            overflow_thresh = RHO_OVERFLOW_THRESH
        else:
            assert(config.TRAIN_PARAMETER_TYPE == 'Nu')
            overflow_thresh = NU_OVERFLOW_THRESH

        for it in range(config.TEST_NUMBER):
            index = np.random.randint(len(test_data))
            day = test_data[index][0]
            logger.info('Test [%d/%d]. Date: %s.' % (it, config.TEST_NUMBER, day.strftime('%Y/%m/%d')))
            augmented_daily_train_data = np.copy(test_data[index])
            max_value = -1e100
            max_state = None
            min_value = 1e100
            min_state = None
            for i in range(5000):
                while True:
                    augmented_daily_train_data = randomly_select_daily_train_data(
                        augmented_daily_train_data, update=True, selection_range=(5, 15),
                        parameter=config.TRAIN_PARAMETER_TYPE)
                    day, state, coefs, yesterday_coefs = augmented_daily_train_data
                    if np.max(np.fabs(state)) < overflow_thresh:  # not overflow
                        break
                value = model.predict(state)
                if value < min_value:
                    min_value = value
                    min_state = np.copy(state)
                if value > max_value:
                    max_value = value
                    max_state = np.copy(state)
            augmented_daily_train_data = day, min_state, coefs, yesterday_coefs
            logger.info('Plotting the worst example. Value: %.6f' % min_value)
            plot_daily_train_data(config, augmented_daily_train_data, parameter=config.TRAIN_PARAMETER_TYPE)
            augmented_daily_train_data = day, max_state, coefs, yesterday_coefs
            logger.info('Plotting the best example. Value: %.6f' % max_value)
            plot_daily_train_data(config, augmented_daily_train_data, parameter=config.TRAIN_PARAMETER_TYPE)
            logger.info('')
    else:
        assert(config.TRAIN_MODEL_TYPE == 'Candidate')
        for it in range(config.TEST_NUMBER):
            index = np.random.randint(len(test_data))
            day = test_data[index][0]
            logger.info('Test [%d/%d]. Date: %s.' % (it, config.TEST_NUMBER, day.strftime('%Y/%m/%d')))
            augmented_daily_train_data = np.copy(test_data[index])
            day, state, coefs, yesterday_coefs = augmented_daily_train_data
            values = model.predict(state)
            candidate_is_selected = np.zeros((len(state),), dtype=bool)
            for idx in np.argsort(values)[-15:]:
                candidate_is_selected[idx] = True
            augmented_daily_train_data = select_daily_train_data(
                augmented_daily_train_data, candidate_is_selected, update=True,
                parameter=config.TRAIN_PARAMETER_TYPE)
            logger.info('Plotting the best example. Point scores: ' + ', '.join('%.6f' % v for v in values))
            plot_daily_train_data(config, augmented_daily_train_data, parameter=config.TRAIN_PARAMETER_TYPE)
            logger.info('')

    logger.info('Testing ended.')
    logger.info('')
