import sys
import os

import numpy as np
import pandas as pd

import time
import datetime

import warnings

from vol_surface_manager.kap_constants import *
from vol_surface_manager.kap_parse_args import parse_args
from vol_surface_manager.kap_logger import acquire_logger

from vol_surface_manager.kap_train import plot_loss_log_data, plot_candidate_log_data


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
    logger.info('Considering %s index. Plotting log for %s model.' % (config.DATA_TYPE, config.TRAIN_PARAMETER_TYPE))

    # 1. Acquiring log file
    try:
        assert(os.path.isdir(config.MODEL_BASE_DIR))
    except Exception as e:
        logger.fatal('Invalid model directory: "%s."' % config.MODEL_BASE_DIR)
        raise e

    if config.TRAIN_MODEL_TYPE == 'Loss':
        LOSS_MODEL_DIRNAME = 'loss_models'
        loss_model_base_dir = os.path.join(config.MODEL_BASE_DIR, LOSS_MODEL_DIRNAME)
        loss_model_base_dir_2 = os.path.join(loss_model_base_dir, config.DATA_TYPE)
        loss_model_save_dir = os.path.join(loss_model_base_dir_2, config.TRAIN_PARAMETER_TYPE)
        LOG_DIRNAME = 'logs'
        loss_model_log_dir = os.path.join(loss_model_save_dir, LOG_DIRNAME)

        if config.DATA_TYPE == 'Kospi200':
            if config.TRAIN_PARAMETER_TYPE == 'Alpha':
                log_datetime = '20190920153952'
            elif config.TRAIN_PARAMETER_TYPE == 'Rho':
                log_datetime = '20190920214916'
            else:
                assert(config.TRAIN_PARAMETER_TYPE == 'Nu')
                log_datetime = '20190920154024'
        elif config.DATA_TYPE == 'S&P500':
            if config.TRAIN_PARAMETER_TYPE == 'Alpha':
                log_datetime = '20190920060516'
            elif config.TRAIN_PARAMETER_TYPE == 'Rho':
                log_datetime = '20190920060559'
            else:
                assert(config.TRAIN_PARAMETER_TYPE == 'Nu')
                log_datetime = '20190920060622'
        elif config.DATA_TYPE == 'Eurostoxx50':
            if config.TRAIN_PARAMETER_TYPE == 'Alpha':
                log_datetime = '20190921154426'
            elif config.TRAIN_PARAMETER_TYPE == 'Rho':
                log_datetime = '20190921154432'
            else:
                assert(config.TRAIN_PARAMETER_TYPE == 'Nu')
                log_datetime = '20190921154434'
        else:
            assert(config.DATA_TYPE == 'HSCEI')
            if config.TRAIN_PARAMETER_TYPE == 'Alpha':
                log_datetime = '20190922025420'
            elif config.TRAIN_PARAMETER_TYPE == 'Rho':
                log_datetime = '20190921231652'
            else:
                assert(config.TRAIN_PARAMETER_TYPE == 'Nu')
                log_datetime = '20190921231732'

        log_path = os.path.join(loss_model_log_dir, 'training_log_%s.csv' % log_datetime)

    else:
        assert(config.TRAIN_MODEL_TYPE == 'Candidate')

        CANDIDATE_MODEL_DIRNAME = 'candidate_models'
        candidate_model_base_dir = os.path.join(config.MODEL_BASE_DIR, CANDIDATE_MODEL_DIRNAME)
        candidate_model_base_dir_2 = os.path.join(candidate_model_base_dir, config.DATA_TYPE)
        candidate_model_save_dir = os.path.join(candidate_model_base_dir_2, config.TRAIN_PARAMETER_TYPE)
        LOG_DIRNAME = 'logs'
        candidate_model_log_dir = os.path.join(candidate_model_save_dir, LOG_DIRNAME)

        if config.DATA_TYPE == 'Kospi200':
            if config.TRAIN_PARAMETER_TYPE == 'Alpha':
                log_datetime = '20190920054013'
            elif config.TRAIN_PARAMETER_TYPE == 'Rho':
                log_datetime = '20190920054005'
            else:
                assert(config.TRAIN_PARAMETER_TYPE == 'Nu')
                log_datetime = '20190920054008'
        elif config.DATA_TYPE == 'S&P500':
            if config.TRAIN_PARAMETER_TYPE == 'Alpha':
                log_datetime = '20190921110703'
            elif config.TRAIN_PARAMETER_TYPE == 'Rho':
                log_datetime = '20190920225213'
            else:
                assert(config.TRAIN_PARAMETER_TYPE == 'Nu')
                log_datetime = '20190921074539'
        elif config.DATA_TYPE == 'Eurostoxx50':
            if config.TRAIN_PARAMETER_TYPE == 'Alpha':
                log_datetime = '20190923052323'
            elif config.TRAIN_PARAMETER_TYPE == 'Rho':
                log_datetime = '20190922123443'
            else:
                assert(config.TRAIN_PARAMETER_TYPE == 'Nu')
                log_datetime = '20190922150520'
        else:
            assert(config.DATA_TYPE == 'HSCEI')
            if config.TRAIN_PARAMETER_TYPE == 'Alpha':
                log_datetime = '20190924064347'
            elif config.TRAIN_PARAMETER_TYPE == 'Rho':
                log_datetime = '20190923114325'
            else:
                assert(config.TRAIN_PARAMETER_TYPE == 'Nu')
                log_datetime = '20190924012559'

        log_path = os.path.join(candidate_model_log_dir, 'training_log_%s.csv' % log_datetime)

    try:
        log_data = pd.read_csv(log_path)
        logger.info('Log data is loaded from path "%s".' % log_path)
    except Exception as e:
        logger.fatal('Cannot load log data from path "%s".' % log_path)
        raise e

    # 2. Plot log file
    try:
        assert(os.path.isdir(config.RESULT_BASE_DIR))
    except Exception as e:
        logger.fatal('Invalid result directory: "%s."' % config.RESULT_BASE_DIR)
        raise e
    if config.PLOT_FIGURES == 'True':
        plot_result = True
    else:
        assert(config.PLOT_FIGURES == 'False')
        plot_result = False
    logger.info('Now plotting the log file.')
    if config.TRAIN_MODEL_TYPE == 'Loss':
        if config.SAVE_FIGURES == 'True':
            save_path = os.path.join(config.RESULT_BASE_DIR, 'loss_model_log_graph_%s_%s.png' % (
                config.DATA_TYPE, config.TRAIN_PARAMETER_TYPE))
        else:
            assert(config.SAVE_FIGURES == 'False')
            save_path = ''
        plot_loss_log_data(config, log_data, save_path, plot_result, parameter=config.TRAIN_PARAMETER_TYPE)
    else:
        assert(config.TRAIN_MODEL_TYPE == 'Candidate')
        if config.SAVE_FIGURES == 'True':
            save_path = os.path.join(config.RESULT_BASE_DIR, 'candidate_model_log_graph_%s_%s.png' % (
                config.DATA_TYPE, config.TRAIN_PARAMETER_TYPE))
        else:
            assert(config.SAVE_FIGURES == 'False')
            save_path = ''
        plot_candidate_log_data(config, log_data, save_path, plot_result, parameter=config.TRAIN_PARAMETER_TYPE)
    logger.info('')
