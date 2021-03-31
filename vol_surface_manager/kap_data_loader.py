import sys
import os

import numpy as np
import pandas as pd

from .kap_constants import *
from .kap_parse_args import parse_args
from .kap_logger import acquire_logger


def load_market_data(excel_name):
    data = pd.read_excel(excel_name, read_only=True,
                         sheet_name='Market Data')
    data['Tenor'] = (data['Exp_Date'] - data['Realday']).dt.days
    data['Moneyness'] = np.log(data['Strike'] / data['SpotPrice'])
    columns = ['Realday', 'Position', 'Tenor', 'Moneyness',
               'SpotPrice', 'Strike', 'ImpVol',
               'Last_Price', 'Bid_Price', 'Ask_Price']
    data = data.loc[:, columns]
    data = data.rename({
        'Last_Price': 'LastPrice', 'Bid_Price': 'BidPrice', 'Ask_Price': 'AskPrice'}, axis=1)
    return data.set_index('Realday')


def load_raw_param_data(excel_name):
    data = pd.read_excel(excel_name, read_only=True,
                         sheet_name='Raw')
    columns = ['REALDAY', 'TTM', 'ALPHA', 'RHO', 'NU']
    data = data.loc[:, columns]
    data = data.rename({
        'REALDAY': 'Realday', 'TTM': 'Tenor',
        'ALPHA': 'Alpha', 'RHO': 'Rho', 'NU': 'Nu'}, axis=1)
    return data.set_index('Realday')
    

def load_coef_data(excel_name):
    data = None
    for parameter in ['Alpha', 'Rho', 'Nu']:
        data_one = pd.read_excel(excel_name, read_only=True,
                                 sheet_name=parameter+'_coef')
        if parameter == 'Alpha':
            columns = ["REALDAY", "p1", "p2", 'p3', 'p4', 'p5']
            data_one = data_one.loc[:, columns]
            data_one = data_one.rename({
                'REALDAY': 'Realday',
                'p1': 'Alpha_p1', 'p2': 'Alpha_p2', 'p3': 'Alpha_p3',
                'p4': 'Alpha_p4', 'p5': 'Alpha_p5'}, axis=1)
        elif parameter == 'Rho':
            columns = ["REALDAY", "p1", "p2", 'p3']
            data_one = data_one.loc[:, columns]
            data_one = data_one.rename({
                'REALDAY': 'Realday',
                'p1': 'Rho_p1', 'p2': 'Rho_p2', 'p3': 'Rho_p3'}, axis=1)
        else:
            assert(parameter == 'Nu')
            columns = ["REALDAY", "p1", "p2", 'p3', 'P4']
            data_one = data_one.loc[:, columns]
            data_one = data_one.rename({
                'REALDAY': 'Realday',
                'p1': 'Nu_p1', 'p2': 'Nu_p2', 'p3': 'Nu_p3',
                'P4': 'Nu_p4'}, axis=1)
        if data is None:
            data = data_one
        else:
            data = pd.merge(data, data_one, on='Realday', how='outer')
    return data.set_index('Realday')


def load_irs_data(excel_name):
    data = pd.read_excel(excel_name, read_only=True,
                         sheet_name='IRS')
    columns = ['REALDAY'] + IRS_COLUMNS
    data = data.loc[:, columns]
    data = data.rename({'REALDAY': 'Realday'}, axis=1)
    return data.set_index('Realday')


def load_dividend_data(excel_name):
    data = pd.read_excel(excel_name, read_only=True,
                         sheet_name='Discrete Dividend')
    data['T'] = (data['EVENTDATE'] - data['REALDAY']).dt.days
    columns = ['REALDAY', 'T', 'DIVIDEND']
    data = data.loc[:,  columns]
    data = data.rename({'REALDAY': 'Realday', 'DIVIDEND': 'Dividend'}, axis=1)
    return data.set_index('Realday')


def load_selection_data(excel_name):
    columns = ['REALDAY', 'STATUS', 'TTM', 'VALUE', 'SELECTED']
    alpha_selection_data = None
    rho_selection_data = None
    nu_selection_data = None
    for parameter in ['Alpha', 'Rho', 'Nu']:
        data_one = pd.read_excel(excel_name, read_only=True,
                                 sheet_name=parameter+'_coef')
        data_one = data_one.loc[:, columns]
        if parameter == 'Alpha':
            alpha_selection_data = data_one
        elif parameter == 'Rho':
            rho_selection_data = data_one
        else:
            assert(parameter == 'Nu')
            nu_selection_data = data_one
    alpha_selection_data = alpha_selection_data.set_index('REALDAY')
    rho_selection_data = rho_selection_data.set_index('REALDAY')
    nu_selection_data = nu_selection_data.set_index('REALDAY')
    return alpha_selection_data, rho_selection_data, nu_selection_data


def save_selection_data_from_config(config, alpha_selection_data, rho_selection_data, nu_selection_data):
    columns = ['REALDAY', 'STATUS', 'TTM', 'VALUE', 'SELECTED']
    df_alpha = alpha_selection_data.reset_index().reindex(columns=columns)
    df_rho = rho_selection_data.reset_index().reindex(columns=columns)
    df_nu = nu_selection_data.reset_index().reindex(columns=columns)
    save_path = os.path.join(config.RESULT_BASE_DIR, config.DATA_TYPE + config.COEF_DATA_SUBNAME + '_Selection.xlsx')
    with pd.ExcelWriter(save_path, engine='openpyxl',
                        date_format='M/D/YYYY', datetime_format='M/D/YYYY', mode='w') as writer:
        df_alpha.to_excel(writer, float_format='%.16f', sheet_name='Alpha_coef', index=False)
        df_rho.to_excel(writer, float_format='%.16f', sheet_name='Rho_coef', index=False)
        df_nu.to_excel(writer, float_format='%.16f', sheet_name='Nu_coef', index=False)


def save_coef_data_from_config(config, coef_data):
    columns = ['Alpha_p1', 'Alpha_p2', 'Alpha_p3', 'Alpha_p4', 'Alpha_p5']
    df_alpha = coef_data.copy().loc[:, columns]
    df_alpha['REALDAY'] = df_alpha.index
    df_alpha['INDEXCODE'] = config.DATA_TYPE.upper()
    df_alpha = df_alpha.loc[:, ['REALDAY', 'INDEXCODE'] + columns]
    df_alpha = df_alpha.rename({'Alpha_p1': 'p1', 'Alpha_p2': 'p2', 'Alpha_p3': 'p3',
                                'Alpha_p4': 'p4', 'Alpha_p5': 'p5'}, axis=1)

    columns = ['Rho_p1', 'Rho_p2', 'Rho_p3']
    df_rho = coef_data.copy().loc[:, columns]
    df_rho['REALDAY'] = df_rho.index
    df_rho['INDEXCODE'] = config.DATA_TYPE.upper()
    df_rho = df_rho.loc[:, ['REALDAY', 'INDEXCODE'] + columns]
    df_rho = df_rho.rename({'Rho_p1': 'p1', 'Rho_p2': 'p2', 'Rho_p3': 'p3'}, axis=1)

    columns = ['Nu_p1', 'Nu_p2', 'Nu_p3', 'Nu_p4']
    df_nu = coef_data.copy().loc[:, columns]
    df_nu['REALDAY'] = df_nu.index
    df_nu['INDEXCODE'] = config.DATA_TYPE.upper()
    df_nu = df_nu.loc[:, ['REALDAY', 'INDEXCODE'] + columns]
    df_nu = df_nu.rename({'Nu_p1': 'p1', 'Nu_p2': 'p2', 'Nu_p3': 'p3', 'Nu_p4': 'P4'}, axis=1)

    save_path = os.path.join(config.RESULT_BASE_DIR, config.DATA_TYPE + config.COEF_DATA_SUBNAME + '_Prediction.xlsx')
    with pd.ExcelWriter(save_path, engine='openpyxl',
                        date_format='M/D/YYYY', datetime_format='M/D/YYYY', mode='w') as writer:
        df_alpha.to_excel(writer, float_format='%.16f', sheet_name='Alpha_coef', index=False)
        df_rho.to_excel(writer, float_format='%.16f', sheet_name='Rho_coef', index=False)
        df_nu.to_excel(writer, float_format='%.16f', sheet_name='Nu_coef', index=False)


def load_param_data(excel_name):
    data = pd.read_excel(excel_name, read_only=True,
                         sheet_name='Result_Parameter')
    columns = ['TRADEDAY', 'TENOR', 'ALPHA', 'RHO', 'NU']
    data = data.loc[:, columns]
    data = data.rename({
        'TRADEDAY': 'Realday', 'TENOR': 'Tenor',
        'ALPHA': 'Alpha', 'RHO': 'Rho', 'NU': 'Nu'}, axis=1)
    return data.set_index('Realday')


def load_data_from_config(config, logger, mode='train', acquire_market_data=True):
    if mode == 'train':
        data_base_dir = config.TRAIN_DATA_BASE_DIR
    else:
        assert(mode == 'eval')
        data_base_dir = config.DATA_BASE_DIR
    try:
        assert(os.path.isdir(data_base_dir))
    except Exception as e:
        logger.fatal('Invalid data directory: "%s."' % data_base_dir)
        raise e

    # Acquire market data
    market_path = os.path.join(data_base_dir, config.DATA_TYPE + config.MARKET_DATA_SUBNAME + '.xlsx')
    coef_path = os.path.join(data_base_dir, config.DATA_TYPE + config.COEF_DATA_SUBNAME + '.xlsx')
    raw_param_path = os.path.join(data_base_dir, config.DATA_TYPE + config.RAW_PARAM_DATA_SUBNAME +'.xlsx')

    market_data = None
    if acquire_market_data:
        try:  # Market data may not exist.
            market_data = load_market_data(market_path)
        except Exception as e:
            logger.warning('Cannot load market data from path "%s". But this is not necessary.' % market_path)
            market_data = None
    try:
        raw_param_data = load_raw_param_data(raw_param_path)
    except Exception as e:
        logger.fatal('Cannot load raw parameter data from path "%s".' % raw_param_path)
        raise e
    try:
        coef_data = load_coef_data(coef_path)
    except Exception as e:
        logger.fatal('Cannot load coefficient data from path "%s".' % coef_path)
        raise e

    # Acquire market days.

    if market_data is None:
        market_days = []
    else:
        market_days = sorted(list(set(
            [day for day in market_data.index])))
    raw_param_days = sorted(list(set(
        [day for day in raw_param_data.index])))
    coef_days = sorted(list(set(
        [day for day in coef_data.index])))  # This may one-day smaller

    all_days = raw_param_days

    return market_data, raw_param_data, coef_data, all_days, coef_days, market_days


if __name__ == '__main__':
    config = parse_args()
    logger = acquire_logger(config)
    market_data, raw_param_data, coef_data, all_days, coef_days, market_days = load_data_from_config(
        config, logger, mode='train')

    print(len(raw_param_data), len(coef_data))
    # print(market_data[-10:])
    print(raw_param_data[-3:])
    print(coef_data[-3:])
    print(len(all_days), len(coef_days), len(market_days))
