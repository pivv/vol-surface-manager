import sys
import os

import numpy as np
import pandas as pd
import scipy

import torch

import time
import datetime

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from .kap_constants import *
from .kap_utils import fit_param_curve, vol_surface, local_vol_surface, compute_params_from_coefs, compute_irs
from .kap_parse_args import parse_args
from .kap_logger import acquire_logger
from .kap_main_data import acquire_daily_param_data, acquire_daily_train_data, acquire_daily_smile_data, \
    compute_candidate_probs, select_daily_train_data, randomly_select_daily_train_data, \
    plot_daily_train_data, plot_daily_smile_data
from .kap_models import HyeonukLossModel1 as LossModel
from .kap_models import HyeonukCandidateModel1 as CandidateModel


def plot_vol_surface(config, day, pred_alpha_coefs, pred_rho_coefs, pred_nu_coefs,
                     yesterday_alpha_coefs=None, yesterday_rho_coefs=None, yesterday_nu_coefs=None,
                     gold_alpha_coefs=None, gold_rho_coefs=None, gold_nu_coefs=None,
                     save_path='', plot_result=True,
                     surface_type='gold', irs=None, dividend=None):
    assert surface_type in ['gold', 'model', 'diff', 'local_gold', 'local_model', 'local_diff']
    M_min = -0.3
    M_max = 0.3
    T = np.linspace(1./12., 3., num=1080)
    M = np.linspace(M_min, M_max, num=1080)
    T, M = np.meshgrid(T, M)
    r, R = None, None
    if surface_type in ['local_gold', 'local_model', 'local_diff']:
        assert(irs is not None and dividend is not None)
        r, R = compute_irs(irs, T)
    if surface_type in ['gold', 'model', 'diff']:
        V = vol_surface(pred_alpha_coefs, pred_rho_coefs, pred_nu_coefs, T, M)
    else:
        V = local_vol_surface(pred_alpha_coefs, pred_rho_coefs, pred_nu_coefs, T, M, r, R)
    V_max = int(10. * np.max(V)) / 10.
    #VY = None
    #if yesterday_alpha_coefs is not None:
    #    assert(yesterday_rho_coefs is not None and yesterday_nu_coefs is not None)
    #    if surface_type in ['gold', 'model', 'diff']:
    #        VY = vol_surface(yesterday_alpha_coefs, yesterday_rho_coefs, yesterday_nu_coefs, T, M)
    #    else:
    #        VY = local_vol_surface(yesterday_alpha_coefs, yesterday_rho_coefs, yesterday_nu_coefs, T, M, r, R)
    VG = None
    DV = None
    VG_max = None
    if gold_alpha_coefs is not None:
        assert(gold_rho_coefs is not None and gold_nu_coefs is not None)
        if surface_type in ['gold', 'model', 'diff']:
            VG = vol_surface(gold_alpha_coefs, gold_rho_coefs, gold_nu_coefs, T, M)
        else:
            VG = local_vol_surface(gold_alpha_coefs, gold_rho_coefs, gold_nu_coefs, T, M, r, R)
        VG_max = int(10. * np.max(VG)) / 10.
        DV = V - VG

    fig = plt.figure(figsize=(12*2, 10))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0., hspace=0.)
    #fig.subplots_adjust(wspace=0.25)

    if surface_type in ['gold']:
        plt.title('%s, Implied Volatility Surface on %s.' % (
            config.DATA_TYPE.replace('&', '\&'), day.strftime('%Y-%m-%d') ), fontsize=40, pad=20)
    elif surface_type in ['model', 'diff']:
        plt.title('%s, Implied Volatility Surface by Model on %s.' % (
            config.DATA_TYPE.replace('&', '\&'), day.strftime('%Y-%m-%d') ), fontsize=40, pad=20)
    elif surface_type in ['local_gold']:
        plt.title('%s, Local Volatility Surface on %s.' % (
            config.DATA_TYPE.replace('&', '\&'), day.strftime('%Y-%m-%d') ), fontsize=40, pad=20)
    else:
        assert(surface_type in ['local_model', 'local_diff'])
        plt.title('%s, Local Volatility Surface by Model on %s.' % (
            config.DATA_TYPE.replace('&', '\&'), day.strftime('%Y-%m-%d') ), fontsize=40, pad=20)
    plt.axis('off')

    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        #ax = Axes3D(fig)
        if i == 0:
            ax.set_xlabel('Maturity (year)', fontsize=30, labelpad=30)
            ax.set_ylabel('Moneyness', fontsize=30, labelpad=20)
            #ax.set_zlabel('Volatility', fontsize=30, labelpad=40)
        else:
            ax.set_xlabel('Maturity (year)', fontsize=30, labelpad=20)
            ax.set_ylabel('Moneyness', fontsize=30, labelpad=35)
            #ax.set_zlabel('Volatility', fontsize=30, labelpad=40)
        if i == 0:
            surf = ax.plot_surface(T, M, V, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.plot_wireframe(T, M, V, color='black')
        else:
            if surface_type in ['diff', 'local_diff']:
                assert(DV is not None)
                surf = ax.plot_surface(T, M, DV, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                ax.plot_wireframe(T, M, DV, color='black')
            elif surface_type in ['gold', 'local_gold']:
                surf = ax.plot_surface(T, M, VG, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                ax.plot_wireframe(T, M, VG, color='black')
            else:
                assert(surface_type in ['model', 'local_model'])
                surf = ax.plot_surface(T, M, V, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                ax.plot_wireframe(T, M, V, color='black')
        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # rotate the axes and update
        #for angle in range(0, 360):
        if surface_type in ['diff']:
            if i == 0:
                ax.set_title(r'$\Sigma(K, T; \widetilde{{\bf p}^*}, \widetilde{{\bf q}^*}, \widetilde{{\bf r}^*})$',
                             fontsize=30, pad=20)
            else:
                ax.set_title(r'$\Sigma(K, T; \widetilde{{\bf p}^*}, \widetilde{{\bf q}^*}, \widetilde{{\bf r}^*}) -' +
                             r'\Sigma(K, T; {\bf p}^*, {\bf q}^*, {\bf r}^*)$',
                             fontsize=30, pad=20)
        elif surface_type in ['local_diff']:
            if i == 0:
                ax.set_title(r'$\Sigma_{loc}(K, T; \widetilde{{\bf p}^*}, \widetilde{{\bf q}^*}, \widetilde{{\bf r}^*})$',
                             fontsize=30, pad=20)
            else:
                ax.set_title(r'$\Sigma_{loc}(K, T; \widetilde{{\bf p}^*}, \widetilde{{\bf q}^*}, \widetilde{{\bf r}^*}) -' +
                             r'\Sigma_{loc}(K, T; {\bf p}^*, {\bf q}^*, {\bf r}^*)$',
                             fontsize=30, pad=20)
        ax.set_xlim([0., 3.])
        ax.set_ylim([M_min, M_max])
        if i == 0:
            if surface_type in ['gold', 'local_gold']:
                ax.set_zlim([0., max(0.4, VG_max)])
            else:
                ax.set_zlim([0., max(0.4, V_max)])
        else:
            if surface_type in ['diff', 'local_diff']:
                ax.set_zlim([-0.04, 0.04])
            elif surface_type in ['gold', 'local_gold']:
                ax.set_zlim([0., max(0.4, VG_max)])
            else:
                assert(surface_type in ['model', 'local_model'])
                ax.set_zlim([0., max(0.4, V_max)])
        if i == 0:
            ax.view_init(35, -10)
            ax.tick_params(axis='x', labelsize=30, pad=10)
            ax.tick_params(axis='y', labelsize=30, pad=0)
            ax.tick_params(axis='z', labelsize=30, pad=20)
        else:
            ax.tick_params(axis='x', labelsize=30, pad=0)
            ax.tick_params(axis='y', labelsize=30, pad=10)
            ax.tick_params(axis='z', labelsize=30, pad=25)

    #plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if plot_result:
        plt.show()


def create_param_curve(logger, daily_train_data, loss_model, candidate_model, parameter='Alpha'):
    try:
        assert(loss_model is not None or candidate_model is not None)
    except Exception as e:
        logger.fatal('Need at least one %s model.' % parameter)
        raise e

    if loss_model is not None:
        loss_model.eval()
    if candidate_model is not None:
        candidate_model.eval()

    daily_train_data = np.copy(daily_train_data)
    _, initial_state, _, _, _ = daily_train_data

    if parameter == 'Alpha':
        overflow_max = ALPHA_OVERFLOW_MAX
        overflow_min = ALPHA_OVERFLOW_MIN
    elif parameter == 'Rho':
        overflow_max = RHO_OVERFLOW_MAX
        overflow_min = RHO_OVERFLOW_MIN
    else:
        assert(parameter == 'Nu')
        overflow_max = NU_OVERFLOW_MAX
        overflow_min = NU_OVERFLOW_MIN
    T = np.linspace(0., 3., num=1081)[1:]

    monte_carlo_iter = 500

    # 1. Compute point scores.
    if candidate_model is None:
        point_scores = None
    else:
        point_scores = candidate_model.predict(initial_state)
    logger.debug('Point scores: ' + ', '.join('%.3f' % score for score in point_scores))

    # 2. Compute candidate probabilities.
    candidate_probs, point_probs, candidate_point_nums = compute_candidate_probs(daily_train_data, point_scores,
                                                                                 point_score_thresh=POINT_SCORE_THRESH)
    assert(np.sum(candidate_probs) > 0)
    logger.debug('Total # of candidates: %d.' % np.sum(candidate_probs > 0))
    logger.debug('Probabilities based on point scores')
    for point_num in range(len(initial_state)+1):
        logger.debug(f'# of points: {point_num}. Probability: ' +
                     f'{np.sum(candidate_probs[candidate_point_nums == point_num])}')
    onehot_probs, _, _ = compute_candidate_probs(daily_train_data, point_scores,
                                                 point_score_thresh=POINT_SCORE_THRESH,
                                                 onehot=True)
    assert(np.sum(onehot_probs) > 0)
    logger.debug('One-hot probabilities')
    for point_num in range(len(initial_state)+1):
        logger.debug(f'# of points: {point_num}. Probability: ' +
                     f'{np.sum(onehot_probs[candidate_point_nums == point_num])}')

    # 3. Find best candidate.
    if loss_model is None:
        best_value = 0.
        best_daily_train_data = randomly_select_daily_train_data(daily_train_data, onehot_probs,
                                                                 parameter=parameter)
        _, _, _, _, pred_coefs = best_daily_train_data
        params = compute_params_from_coefs(pred_coefs, T, parameter)
    else:
        # Monte-Carlo approach
        best_value = -1e100
        params = None
        best_daily_train_data = daily_train_data
        start_time = time.time()
        for it in range(monte_carlo_iter):
            value = None
            if it == 0:
                temp_daily_train_data = randomly_select_daily_train_data(daily_train_data, onehot_probs,
                                                                         parameter=parameter)
                _, state, _, _, pred_coefs = temp_daily_train_data
                value = loss_model.predict(state)  # negative loss.
            else:
                while True:
                    temp_daily_train_data = randomly_select_daily_train_data(daily_train_data, candidate_probs,
                                                                             parameter=parameter)
                    _, state, _, _, pred_coefs = temp_daily_train_data
                    pred_curve_params = state[:, 4]
                    if np.max(pred_curve_params) >= overflow_max or np.min(pred_curve_params) <= overflow_min:  # overflow
                        continue
                    value = loss_model.predict(state)  # negative loss.
                    break
            if value > best_value:
                # Further overflow if needed.
                params = compute_params_from_coefs(pred_coefs, T, parameter)
                if np.max(params) >= overflow_max or np.min(params) <= overflow_min:  # overflow
                    continue
                best_value = value
                best_daily_train_data = temp_daily_train_data

    # 5. Compute predicted coefficients.
    print(T)
    print(params)
    print(best_value)
    _, best_state, _, yesterday_coefs, pred_coefs = best_daily_train_data
    return best_daily_train_data, point_scores, best_value


"""
    min_candidate_number = 8
    #max_candidate_number = 14
    #candidate_score_threshold = -2.
    max_candidate_number = 200
    candidate_score_threshold = -1.
    monte_carlo_num = 10000
    do_monte = False

    if candidate_model is None:
        candidate_scores = None
        candidate_argsort = np.arange(len(state))
    else:
        # 1. Compute candidate scores.
        candidate_scores = candidate_model.predict(state)

        #if parameter == 'Rho':  # Both are important!
        # Way 1
        candidate_argsort = np.argsort(candidate_scores)[::-1]
        candidate_argsort = candidate_argsort[
            candidate_scores[candidate_argsort] > candidate_score_threshold]  # remove outliers.
        if len(candidate_argsort) > max_candidate_number:
            candidate_argsort = candidate_argsort[:max_candidate_number]
        #else:  # Yesterday is important!
        #    # Way 2
        #    candidate_raw_argsort = np.argsort(
        #        candidate_scores[NUM_STANDARD_TENOR:])[::-1] + NUM_STANDARD_TENOR
        #    candidate_raw_argsort = candidate_raw_argsort[
        #        candidate_scores[candidate_raw_argsort] > candidate_score_threshold]  # remove outliers.
        #    if len(candidate_raw_argsort) > max_candidate_number - NUM_STANDARD_TENOR:
        #        candidate_raw_argsort = candidate_raw_argsort[:max_candidate_number - NUM_STANDARD_TENOR]
        #    candidate_argsort = np.concatenate(
        #        [candidate_raw_argsort, np.arange(NUM_STANDARD_TENOR)], axis=0)

    # 2. Compute loss values.
    T = np.linspace(1./12., 3., num=1080)
    if loss_model is None:
        candidate_is_selected = np.zeros((len(state),), dtype=bool)
        candidate_is_selected[candidate_argsort] = True
        update = True
        augmented_daily_train_data = select_daily_train_data(
            augmented_daily_train_data, candidate_is_selected, update, parameter)
        day, state, coefs, yesterday_coefs = augmented_daily_train_data
        max_value = None
        max_state = np.copy(state)
    else:
        max_value = -1e100
        max_state = None
        if do_monte or candidate_model is None:
            num_iter = monte_carlo_num
            do_monte = True
        else:
            assert(len(candidate_argsort) <= max_candidate_number)
            num_iter = 2 ** len(candidate_argsort)
        for i in range(num_iter):
            if not do_monte:
                candidate_is_selected = np.zeros((len(state),), dtype=bool)
                # If considering alpha, always adding some points.
                if parameter == 'Alpha':
                    candidate_is_selected[NUM_STANDARD_TENOR-1] = True
                    candidate_is_selected[NUM_STANDARD_TENOR] = True
                it = i
                for j in range(len(candidate_argsort)):
                    if it % 2 == 1:
                        candidate_is_selected[candidate_argsort[j]] = True
                    it = it >> 1
                if np.sum(candidate_is_selected) < min_candidate_number:
                    continue
                augmented_daily_train_data = select_daily_train_data(
                    augmented_daily_train_data, candidate_is_selected,
                    update=True, parameter=parameter)
            else:
                augmented_daily_train_data = randomly_select_daily_train_data(
                    augmented_daily_train_data, update=True,
                    selection_range=(min_candidate_number, max_candidate_number),
                    parameter=parameter)
            day, state, coefs, yesterday_coefs = augmented_daily_train_data
            if np.max(state[:, 4]) >= overflow_max or np.min(state[:, 4]) <= overflow_min:  # overflow
                continue
            # Further overflow
            params = compute_params_from_coefs(coefs, T, parameter=parameter)
            if np.max(params) >= overflow_max or np.min(params) <= overflow_min:  # overflow
                continue
            value = loss_model.predict(state)  # minus loss.
            if value > max_value:
                max_value = value
                max_state = np.copy(state)

    augmented_daily_train_data = day, max_state, coefs, yesterday_coefs
    candidate_tenors = max_state[:, 0]
    candidate_params = max_state[:, 1]
    candidate_is_selected = (max_state[:, 3] == 1.)
    assert(np.sum(candidate_is_selected) > 0)
    pred_coefs = fit_param_curve(candidate_params[candidate_is_selected],
                                 candidate_tenors[candidate_is_selected],
                                 yesterday_coefs, parameter)

    return augmented_daily_train_data, pred_coefs, candidate_scores, max_value
"""


def create_param_curve_at_day(logger, day, yesterday, market_data, raw_param_data, coef_data,
                              loss_model, candidate_model, parameter='Alpha'):
    daily_param_data = acquire_daily_param_data(
        day, yesterday, market_data, raw_param_data, coef_data, parameter)
    daily_train_data = acquire_daily_train_data(daily_param_data, parameter)
    return create_param_curve(logger, daily_train_data, loss_model, candidate_model, parameter)


def create_vol_surface_at_day(logger, config, day, yesterday, market_data, raw_param_data, coef_data,
                              alpha_loss_model, alpha_candidate_model,
                              nu_loss_model, nu_candidate_model,
                              rho_loss_model, rho_candidate_model,
                              save_base_name='', plot_result=True,
                              plot_curve=True, plot_smile=True, plot_surface=True):
    logger.info('')
    logger.info('Now creating vol surface.')
    logger.info('')

    start_time = time.time()
    pred_alpha_coefs, pred_rho_coefs, pred_nu_coefs = None, None, None
    yesterday_alpha_coefs, yesterday_rho_coefs, yesterday_nu_coefs = None, None, None
    gold_alpha_coefs, gold_rho_coefs, gold_nu_coefs = None, None, None
    daily_alpha_data, daily_rho_data, daily_nu_data = None, None, None
    daily_alpha_train_data, daily_rho_train_data, daily_nu_train_data = None, None, None
    for parameter in ['Alpha', 'Rho', 'Nu']:
        daily_param_data = acquire_daily_param_data(
            day, yesterday, market_data, raw_param_data, coef_data, parameter)
        daily_train_data = acquire_daily_train_data(daily_param_data, parameter)
        if parameter == 'Alpha':
            daily_alpha_data = daily_param_data
            daily_alpha_train_data, _, _ = create_param_curve(
                logger, daily_train_data, alpha_loss_model, alpha_candidate_model, parameter)
            _, _, gold_alpha_coefs, yesterday_alpha_coefs, pred_alpha_coefs = daily_alpha_train_data
            logger.info('Computed Alpha Coefficients: ' + ', '.join('%.6f' % p for p in pred_alpha_coefs))
            elapsed_time = time.time() - start_time
            logger.info('Elapsed time: %5u seconds.' % elapsed_time)
            logger.info('')
            save_path = save_base_name + '_Alpha_curve.png' if save_base_name else ''
            if save_path or (plot_result and plot_curve):
                plot_daily_train_data(config, daily_alpha_train_data, save_path, plot_curve, parameter)
        elif parameter == 'Rho':
            daily_rho_data = daily_param_data
            daily_rho_train_data, _, _ = create_param_curve(
                logger, daily_train_data, rho_loss_model, rho_candidate_model, parameter)
            _, _, gold_alpha_coefs, yesterday_alpha_coefs, pred_rho_coefs = daily_rho_train_data
            logger.info('Computed Rho Coefficients: ' + ', '.join('%.6f' % p for p in pred_rho_coefs))
            elapsed_time = time.time() - start_time
            logger.info('Elapsed time: %5u seconds.' % elapsed_time)
            logger.info('')
            save_path = save_base_name + '_Rho_curve.png' if save_base_name else ''
            if save_path or (plot_result and plot_curve):
                plot_daily_train_data(config, daily_rho_train_data, save_path, plot_curve, parameter)
        else:
            assert(parameter == 'Nu')
            daily_nu_data = daily_param_data
            daily_nu_train_data, _, _ = create_param_curve(
                logger, daily_train_data, nu_loss_model, nu_candidate_model, parameter)
            _, _, gold_alpha_coefs, yesterday_alpha_coefs, pred_nu_coefs = daily_nu_train_data
            logger.info('Computed Nu Coefficients: ' + ', '.join('%.6f' % p for p in pred_nu_coefs))
            elapsed_time = time.time() - start_time
            logger.info('Elapsed time: %5u seconds.' % elapsed_time)
            logger.info('')
            save_path = save_base_name + '_Nu_curve.png' if save_base_name else ''
            if save_path or (plot_result and plot_curve):
                plot_daily_train_data(config, daily_nu_train_data, save_path, plot_curve, parameter)

    try:
        daily_smile_data = acquire_daily_smile_data(day, market_data, raw_param_data, coef_data)
        save_path = save_base_name + '_smile_curves.png' if save_base_name else ''
        if daily_smile_data is not None and (save_path or (plot_result and plot_smile)):
            logger.info('Plotting smile curves for each option.')
            logger.info('')
            plot_daily_smile_data(config, daily_smile_data, daily_alpha_data, daily_rho_data, daily_nu_data,
                                  pred_alpha_coefs, pred_rho_coefs, pred_nu_coefs, save_path, plot_smile)
    except Exception as e:
        logger.warning(e)

    elapsed_time = time.time() - start_time
    logger.info('Vol surface is created.')
    logger.info('Elapsed time: %5u seconds.' % elapsed_time)
    logger.info('')

    save_path = save_base_name + '_vol_surface.png' if save_base_name else ''
    if save_path or (plot_result and plot_surface):
        plot_vol_surface(config, day, pred_alpha_coefs, pred_rho_coefs, pred_nu_coefs,
                         yesterday_alpha_coefs, yesterday_rho_coefs, yesterday_nu_coefs,
                         gold_alpha_coefs, gold_rho_coefs, gold_nu_coefs,
                         save_path)

    return pred_alpha_coefs, pred_rho_coefs, pred_nu_coefs, \
        daily_alpha_train_data, daily_rho_train_data, daily_nu_train_data
