import sys
import os

import numpy as np

import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    # Shared parameters
    parser.add_argument('--LOG', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'ERROR'],
                        help='어디부터 print할지 level 설정.')
    parser.add_argument('--TRAIN_DATA_BASE_DIR', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../train_data/'),
                        help='Train data가 위치한 디렉토리.')
    parser.add_argument('--DATA_BASE_DIR', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../data/'),
                        help='Data가 위치한 디렉토리.')
    parser.add_argument('--MODEL_BASE_DIR', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../models/'),
                        help='Model을 저장할 디렉토리.')
    parser.add_argument('--RESULT_BASE_DIR', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../results/'),
                        help='결과를 저장할 디렉토리.')
    parser.add_argument('--DATA_TYPE', type=str, default='Kospi200',
                        choices=['Kospi200', 'S&P500', 'Eurostoxx50', 'HSCEI'],
                        help='어떤 index에 대해 예측할 것인지 설정.')
    parser.add_argument('--MARKET_DATA_SUBNAME', type=str, default='_Data', help='Market data 파일의 subname.')
    parser.add_argument('--RAW_PARAM_DATA_SUBNAME', type=str, default='_Parameter',
                        help='Raw parameter data 파일의 subname.')
    parser.add_argument('--COEF_DATA_SUBNAME', type=str, default='_coef_Data', help='Coef data 파일의 subname.')
    parser.add_argument('--IRS_DATA_SUBNAME', type=str, default='_IRS_Div', help='IRS data 파일의 subname.')
    parser.add_argument('--PLOT_FIGURES', type=str, default='True',
                        choices=['True', 'False'],
                        help='만들어진 figure들을 출력하여 볼지 여부를 설정')
    parser.add_argument('--SAVE_FIGURES', type=str, default='False',
                        choices=['True', 'False'],
                        help='만들어진 figure들을 저장할지 여부를 설정')

    # Train & Test parameters
    parser.add_argument('--TRAIN_MODEL_TYPE', type=str, default='Loss',
                        choices=['Loss', 'Candidate'],
                        help='어떤 model에 대해 학습할 것인지 설정.')
    parser.add_argument('--TRAIN_PARAMETER_TYPE', type=str, default='Alpha',
                        choices=['Alpha', 'Rho', 'Nu'],
                        help='어떤 parameter에 대해 학습할 것인지 설정.')
    parser.add_argument('--SAVE_MODELS', type=str, default='True',
                        choices=['True', 'False'],
                        help='모든 model들을 저장할지 여부를 설정.')

    # Test parameters
    parser.add_argument('--TEST_NUMBER', type=int, default=20,
                        help='몇번이나 테스트 할지 설정.')
    parser.add_argument('--TEST_MODEL_INDEX', type=int, default=-1,
                        help='어떤 model에 대해 테스트할 것인지 설정.')

    # Vol parameters
    parser.add_argument('--VOL_DATE', type=str, default='',
                        help='어떤 날짜에서 vol surface를 만들 것인지 설정.')

    config = parser.parse_args(argv)

    return config
