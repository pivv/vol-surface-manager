import sys
import os

import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

import time

import pickle

from collections import defaultdict

from .kap_constants import *
from .kap_utils import *
from .kap_parse_args import parse_args
from .kap_logger import acquire_logger
from .kap_data_loader import load_data_from_config


def acquire_daily_coefs(day, coef_data, parameter='Alpha'):
    if day not in coef_data.index:  # New day to compute!
        return None
    if parameter == 'Alpha':
        coefs = np.array([
            coef_data.loc[day]['Alpha_p1'], coef_data.loc[day]['Alpha_p2'],
            coef_data.loc[day]['Alpha_p3'], coef_data.loc[day]['Alpha_p4'],
            coef_data.loc[day]['Alpha_p5']])
    elif parameter == 'Rho':
        coefs = np.array([
            coef_data.loc[day]['Rho_p1'], coef_data.loc[day]['Rho_p2'],
            coef_data.loc[day]['Rho_p3']])
    else:
        assert(parameter == 'Nu')
        coefs = np.array([
            coef_data.loc[day]['Nu_p1'], coef_data.loc[day]['Nu_p2'],
            coef_data.loc[day]['Nu_p3'], coef_data.loc[day]['Nu_p4']])
    return coefs


def coef_data_to_daily_coefs(day, coef_data):
    alpha_coefs = acquire_daily_coefs(day, coef_data, 'Alpha')
    rho_coefs = acquire_daily_coefs(day, coef_data, 'Rho')
    nu_coefs = acquire_daily_coefs(day, coef_data, 'Nu')
    return alpha_coefs, rho_coefs, nu_coefs


def daily_coefs_to_coef_data(day, alpha_coefs, rho_coefs, nu_coefs, coef_data=None):
    columns = ['Realday', 'Alpha_p1', 'Alpha_p2', 'Alpha_p3', 'Alpha_p4', 'Alpha_p5',
               'Rho_p1', 'Rho_p2', 'Rho_p3', 'Nu_p1', 'Nu_p2', 'Nu_p3', 'Nu_p4']
    if coef_data is None:
        coef_data = pd.DataFrame(columns=columns)
    else:
        coef_data = coef_data.reset_index().reindex(columns=columns)
    coef_data.loc[len(coef_data)] = [day] + alpha_coefs.tolist() + rho_coefs.tolist() + nu_coefs.tolist()
    coef_data = coef_data.set_index('Realday')
    return coef_data


def irs_data_to_daily_irs(day, irs_data):
    if day not in irs_data.index:
        return None
    return np.array(irs_data.loc[day, IRS_COLUMNS])


def dividend_data_to_daily_dividend(day, dividend_data, market_data):
    if day not in dividend_data.index:
        return None
    s = float(market_data.loc[day, 'SpotPrice'].iloc[0])
    tq = np.array(dividend_data.loc[day, 'T']) / float(DAYS_PER_YEAR)
    q = np.log(np.array(dividend_data.loc[day, 'Dividend']) / s)
    return tq, q


def acquire_daily_param_data(day, yesterday,
                             market_data, raw_param_data, coef_data,
                             parameter='Alpha'):
    assert(day in raw_param_data.index)
    coefs = acquire_daily_coefs(day, coef_data, parameter)
    yesterday_coefs = acquire_daily_coefs(
        yesterday, coef_data, parameter)
    assert(yesterday_coefs is not None)

    tenor_indices = np.argsort(raw_param_data.loc[day]['Tenor'].to_numpy())

    raw_tenors = (raw_param_data.loc[day]['Tenor'] / float(DAYS_PER_YEAR)).to_numpy()[tenor_indices]
    raw_params = raw_param_data.loc[day][parameter].to_numpy()[tenor_indices]
    assert(np.all(raw_tenors[:-1] <= raw_tenors[1:]))

    yesterday_tenors = np.copy(STANDARD_TENORS)
    yesterday_params = compute_params_from_coefs(
        yesterday_coefs, yesterday_tenors, parameter)

    daily_param_data = (day, coefs, yesterday_coefs,
                        raw_tenors, raw_params, yesterday_tenors, yesterday_params)

    return daily_param_data


def acquire_daily_smile_data(day, market_data, raw_param_data, coef_data):
    if market_data is None or day not in market_data.index:  # New day to compute!
        return None

    daily_market_data = market_data.loc[day].set_index('Tenor')

    tenor_indices = np.argsort(raw_param_data.loc[day]['Tenor'].to_numpy())
    raw_tenors = (raw_param_data.loc[day]['Tenor'] / float(DAYS_PER_YEAR)).to_numpy()[tenor_indices]
    moneynesses = []
    vols = []
    positions = []

    for tenor in raw_param_data.loc[day]['Tenor'][tenor_indices]:
        if tenor not in daily_market_data.index:
            moneynesses.append(np.array([]))
            vols.append(np.array([]))
            positions.append(np.array([]))
            continue
        option_data = daily_market_data.loc[[tenor]]
        option_moneynesses = option_data['Moneyness'].to_numpy()
        option_vols = option_data['ImpVol'].to_numpy()
        option_positions = option_data['Position'].to_numpy()

        moneynesses.append(option_moneynesses)
        vols.append(option_vols)
        positions.append(option_positions)

    daily_smile_data = (raw_tenors, moneynesses, vols, positions)
    return daily_smile_data


def plot_daily_param_data(config, daily_param_data,
                          save_path='', plot_result=True, parameter='Alpha'):
    if isinstance(parameter, list):
        daily_param_datas = daily_param_data
        parameters = parameter
    else:
        assert(isinstance(parameter, str))
        daily_param_datas = [daily_param_data]
        parameters = [parameter]

    fontsize = 18
    fig = plt.figure(figsize=(7.5*len(parameters), 5))
    fig.subplots_adjust(wspace=0.25)
    for iparam, (daily_param_data, parameter) in enumerate(zip(daily_param_datas, parameters)):
        (day, coefs, yesterday_coefs,
         raw_tenors, raw_params, yesterday_tenors, yesterday_params) = daily_param_data
        curve_tenors = np.linspace(0., 3., num=361)[1:]
        if parameter == 'Alpha':
            parameter_atm = 'alpha_{atm}'
            coef_name = 'p'
        elif parameter == 'Rho':
            parameter_atm = 'rho'
            coef_name = 'q'
        else:
            assert(parameter == 'Nu')
            parameter_atm = 'nu'
            coef_name = 'r'

        ax = fig.add_subplot(1, len(parameters), iparam+1)
        ax.set_title(r'%s, Parameteric Curve of $\%s$ at %s.' % (
            config.DATA_TYPE.replace('&', '\&'), parameter_atm, day.strftime('%Y-%m-%d') ), fontsize=fontsize)
        ax.set_xlabel('Maturity $t$ (year)', fontsize=fontsize)
        ax.set_ylabel(r'$\%s$' % parameter_atm, fontsize=fontsize)
        if coefs is not None:
            curve_params = compute_params_from_coefs(
                coefs, curve_tenors, parameter)
            ax.plot(curve_tenors, curve_params, 'g', label=r'$f_{\%s}(t; {\bf{%s}}^{*})$' %
                                                           (parameter.lower(), coef_name))
        yesterday_curve_params = compute_params_from_coefs(
            yesterday_coefs, curve_tenors, parameter)
        #ax.plot(curve_tenors, yesterday_curve_params, 'b--', label='yesterday')
        ax.plot(yesterday_tenors, yesterday_params, 'bx', label=r'$\{\%s^{c, t} ~ | ~ t \in \mathcal{T}_C\}$' %
                                                                parameter.lower())
        ax.plot(raw_tenors, raw_params, 'ro', label=r'$\{\%s^{r, t} ~ | ~ t \in \mathcal{T}_R\}$' %
                                                    parameter.lower())
        ax.set_xlim([-0.1, 3.1])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        #ax.set_ylim([0.1, 0.2])
        ax.legend(fontsize=15)
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    if plot_result:
        plt.show()


def plot_daily_smile_data(config, daily_smile_data,
                          daily_alpha_data, daily_rho_data, daily_nu_data,
                          pred_alpha_coefs=None, pred_rho_coefs=None,
                          pred_nu_coefs=None, save_path='', plot_result=True, plot_tenor=None):
    tenors, moneynesses, vols, positions = daily_smile_data
    (day, alpha_coefs, yesterday_alpha_coefs, _, raw_alphas, _, _) = daily_alpha_data
    (day, rho_coefs, yesterday_rho_coefs, _, raw_rhos, _, _) = daily_rho_data
    (day, nu_coefs, yesterday_nu_coefs, _, raw_nus, _, _) = daily_nu_data

    if alpha_coefs is not None:
        assert(rho_coefs is not None and nu_coefs is not None)
        curve_alphas = compute_params_from_coefs(alpha_coefs, tenors, 'Alpha')
        curve_rhos = compute_params_from_coefs(rho_coefs, tenors, 'Rho')
        curve_nus = compute_params_from_coefs(nu_coefs, tenors, 'Nu')
    else:
        curve_alphas = None
        curve_rhos = None
        curve_nus = None

    if pred_alpha_coefs is not None:
        assert(pred_rho_coefs is not None and pred_nu_coefs is not None)
        pred_curve_alphas = compute_params_from_coefs(pred_alpha_coefs, tenors, 'Alpha')
        pred_curve_rhos = compute_params_from_coefs(pred_rho_coefs, tenors, 'Rho')
        pred_curve_nus = compute_params_from_coefs(pred_nu_coefs, tenors, 'Nu')
    else:
        pred_curve_alphas = None
        pred_curve_rhos = None
        pred_curve_nus = None

    fontsize = 18
    if plot_tenor is None:
        cols = 4
        rows = (len(tenors)-1) // cols + 1
        fig = plt.figure(figsize=(7.5*cols, 5*rows))
        plt.title('%s, Smile Curves on %s.' % (
            config.DATA_TYPE.replace('&', '\&'), day.strftime('%Y-%m-%d') ), fontsize=20, pad=40)
        #plt.title('%s, Smile Curves by Practitioner on %s.' % (
        #    config.DATA_TYPE.replace('&', '\&'), day.strftime('%Y-%m-%d') ), fontsize=20, pad=40)
        plt.axis('off')
        fig.subplots_adjust(hspace=0.35)
    else:
        assert(np.any(np.round(tenors * DAYS_PER_YEAR) == plot_tenor))
        fig = plt.figure(figsize=(7.5, 5))
        plt.title('%s, Smile Curve on %s.' % (
            config.DATA_TYPE.replace('&', '\&'), day.strftime('%Y-%m-%d') ), fontsize=fontsize)
        #plt.title('%s, Smile Curve by Practitioner on %s.' % (
        #    config.DATA_TYPE.replace('&', '\&'), day.strftime('%Y-%m-%d') ), fontsize=fontsize)

    for tenor_index, tenor in enumerate(tenors):
        if plot_tenor is None or int(np.round(tenor * DAYS_PER_YEAR)) == plot_tenor:
            option_moneynesses = moneynesses[tenor_index]
            option_vols = vols[tenor_index]
            option_positions = positions[tenor_index]

            if len(option_positions) > 0:
                option_is_call = (option_positions == 'C')
                option_is_put = (option_positions == 'P')
            else:
                option_is_call = np.array([])
                option_is_put = np.array([])

            if len(option_moneynesses) > 0:
                M_min = min(-0.3, np.min(option_moneynesses))
                M_max = max(0.3, np.max(option_moneynesses))
            else:
                M_min = -0.3
                M_max = 0.3

            T = tenor
            M = np.linspace(M_min, M_max, num=500)

            raw_alpha = raw_alphas[tenor_index]
            raw_rho = raw_rhos[tenor_index]
            raw_nu = raw_nus[tenor_index]

            if plot_tenor is None:
                ax = fig.add_subplot(rows, cols, tenor_index+1)
                ax.set_title('Maturity $T = %.2f$ year.' % tenor, fontsize=fontsize)
            else:
                ax = fig.add_subplot(1, 1, 1)
            #ax.set_xlabel('Moneyness $\log (K/f)$, Maturity $T = %.2f$.' % tenor, fontsize=fontsize)
            ax.set_xlabel('Moneyness $\log (K/f)$', fontsize=fontsize)
            ax.set_ylabel('Volatility', fontsize=fontsize)
            ax.plot(M, smile(raw_alpha, raw_rho, raw_nu, T, M), 'k', label=r'$\sigma_{model}(K, T; \alpha^{r,T}, \rho^{r,T}, \nu^{r,T})$')
            #if curve_alphas is not None:
            #    curve_alpha = curve_alphas[tenor_index]
            #    curve_rho = curve_rhos[tenor_index]
            #    curve_nu = curve_nus[tenor_index]
            #    ax.plot(M, smile(curve_alpha, curve_rho, curve_nu, T, M), 'g',
            #            label=r'$\sigma_{model}\big(K, T; f_{\alpha}(T;\widetilde{{\bf p}} ^ *), f_{\rho}(T;\widetilde{{\bf q}} ^ *), f_{\nu}(T;\widetilde{{\bf r}} ^ *)\big)$')
            #if pred_curve_alphas is not None:
            #    pred_curve_alpha = pred_curve_alphas[tenor_index]
            #    pred_curve_rho = pred_curve_rhos[tenor_index]
            #    pred_curve_nu = pred_curve_nus[tenor_index]
            #    ax.plot(M, smile(pred_curve_alpha, pred_curve_rho, pred_curve_nu, T, M), 'tab:orange', label='prediction')
            if len(option_moneynesses) > 0:
                ax.plot(option_moneynesses[option_is_call],
                         option_vols[option_is_call], 'r.', label='call option')
                ax.plot(option_moneynesses[option_is_put],
                         option_vols[option_is_put], 'b.', label='put option')
            ax.set_xlim([M_min, M_max])
            #plt.ylim([0.1, 0.2])
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.legend(fontsize=15)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if plot_result:
        plt.show()


def acquire_daily_train_data(daily_param_data, parameter='Alpha'):
    # state
    # tenor param today selected current_curve yesterday_curve

    (day, coefs, yesterday_coefs,
     raw_tenors, raw_params, yesterday_tenors, yesterday_params) = daily_param_data

    candidate_tenors = np.concatenate(
        [yesterday_tenors, raw_tenors], axis=0)
    candidate_params = np.concatenate(
        [yesterday_params, raw_params], axis=0)

    candidate_is_today = np.zeros_like(candidate_params)
    candidate_is_today[-len(raw_params):] = 1.

    if parameter == 'Alpha':
        initial_select = []
    elif parameter == 'Rho':
        initial_select = []
    else:
        assert(parameter == 'Nu')
        initial_select = []

    candidate_is_selected = np.zeros_like(candidate_params)
    if len(initial_select) > 0:
        candidate_is_selected[initial_select] = 1.

    pred_curve_params = np.zeros_like(candidate_params)  # updated later.
    yesterday_curve_params = compute_params_from_coefs(
        yesterday_coefs, candidate_tenors, parameter)

    state = np.stack([candidate_tenors, candidate_params, candidate_is_today, candidate_is_selected,
                      pred_curve_params, yesterday_curve_params], axis=-1)

    assert(state.shape[1] == STATE_DIM)
    pred_coefs = yesterday_coefs
    daily_train_data = (day, state, coefs, yesterday_coefs, pred_coefs)
    daily_train_data = update_daily_train_data(daily_train_data, parameter)  # update

    return daily_train_data


def daily_train_data_to_selection_data(daily_train_data, selection_data=None):
    day, state, coefs, yesterday_coefs, pred_coefs = daily_train_data
    candidate_tenors = state[:, 0]
    candidate_params = state[:, 1]
    candidate_is_today = (state[:, 2] == 1.)
    candidate_is_selected = (state[:, 3] == 1.)
    columns = ['REALDAY', 'STATUS', 'TTM', 'VALUE', 'SELECTED']
    if selection_data is None:
        selection_data = pd.DataFrame(columns=columns)
    else:
        selection_data = selection_data.reset_index().reindex(columns=columns)
    for tenor, param, today, selected in zip(
            candidate_tenors, candidate_params, candidate_is_today, candidate_is_selected):
        status = 'RAW' if today == 1. else 'YESTERDAY'
        tenor_day = int(np.round(tenor * DAYS_PER_YEAR))
        selected_str = 'O' if selected == 1. else 'X'
        selection_data.loc[len(selection_data)] = [day, status, tenor_day, param, selected_str]
    selection_data = selection_data.set_index('REALDAY')
    return selection_data


def selection_data_to_daily_train_data(selection_data, daily_train_data, parameter='Alpha'):
    day, state, coefs, yesterday_coefs, pred_coefs = daily_train_data
    daily_selection_data = selection_data.loc[day].set_index('TTM')
    assert(np.all(np.array(daily_selection_data.index) == np.round(state[:, 0] * DAYS_PER_YEAR)))
    candidate_is_selected = np.array(daily_selection_data.loc[:, 'SELECTED'] == 'O')
    daily_train_data = select_daily_train_data(daily_train_data, candidate_is_selected, parameter)
    return daily_train_data


def select_daily_train_data(daily_train_data, candidate_is_selected, parameter='Alpha'):
    # state
    # tenor param today selected current_curve yesterday_curve
    day, state, coefs, yesterday_coefs, pred_coefs = daily_train_data
    state[candidate_is_selected, 3] = 1.
    state[~candidate_is_selected, 3] = 0.
    daily_train_data = (day, state, coefs, yesterday_coefs, pred_coefs)
    daily_train_data = update_daily_train_data(daily_train_data, parameter)  # update
    return daily_train_data


def compute_candidate_probs(daily_train_data, point_scores=None, point_score_thresh=POINT_SCORE_THRESH,
                            onehot=False):
    # state
    # tenor param today selected current_curve yesterday_curve
    day, state, coefs, yesterday_coefs, pred_coefs = daily_train_data

    if point_scores is None:
        point_probs = np.zeros((len(state),), dtype='float')
        point_probs[:] = 0.5
        min_candidate_point_num = len(state) - MAX_DELTA_SELECTION - 3
    else:
        selection_univ = (point_scores >= point_score_thresh)
        point_probs = scipy.stats.norm.cdf(point_scores)
        point_probs[~selection_univ] = 0.
        #point_probs[NUM_STANDARD_TENOR-1] = 1.
        #point_probs[NUM_STANDARD_TENOR] = 1.
        min_candidate_point_num = np.sum(selection_univ) - MAX_DELTA_SELECTION
    if onehot:
        point_probs[point_probs > 0.] = 1.

    candidate_probs = np.ones((2**len(state),), dtype='float')
    candidate_point_nums = np.zeros((2**len(state),), dtype='int')
    candidate_indices = np.arange(2**len(state))
    for ipoint in range(len(state)):
        candidate_probs[candidate_indices % 2 == 1] *= point_probs[ipoint]
        candidate_probs[candidate_indices % 2 == 0] *= 1. - point_probs[ipoint]
        candidate_point_nums[candidate_indices % 2 == 1] += 1
        candidate_indices = candidate_indices // 2
    candidate_probs[candidate_point_nums < min_candidate_point_num] = 0.
    if np.sum(candidate_probs) > 0:
        candidate_probs /= np.sum(candidate_probs)
    return candidate_probs, point_probs, candidate_point_nums


def randomly_select_daily_train_data(daily_train_data, candidate_probs, fixed_point=None, parameter='Alpha'):
    # state
    # tenor param today selected current_curve yesterday_curve
    day, state, coefs, yesterday_coefs, pred_coefs = daily_train_data

    candidate_is_selected = np.zeros((len(state),), dtype=bool)
    candidate_index = np.random.choice(2**len(state), p=candidate_probs)
    for ipoint in range(len(state)):
        if candidate_index % 2 == 1:
            candidate_is_selected[ipoint] = True
        candidate_index = candidate_index // 2
    assert(candidate_index == 0)
    if fixed_point is not None:
        candidate_is_selected[fixed_point] = True
    return select_daily_train_data(daily_train_data, candidate_is_selected, parameter)


def update_daily_train_data(daily_train_data, parameter='Alpha'):
    # state
    # tenor param today selected current_curve yesterday_curve
    day, state, coefs, yesterday_coefs, pred_coefs = daily_train_data

    # update train data from current selection.
    candidate_tenors = state[:, 0]
    candidate_params = state[:, 1]
    candidate_is_selected = (state[:, 3] == 1.)

    if np.sum(candidate_is_selected) == 0:
        pred_coefs = np.copy(yesterday_coefs)
    else:
        pred_coefs = fit_param_curve(candidate_params[candidate_is_selected],
                                     candidate_tenors[candidate_is_selected],
                                     yesterday_coefs, parameter)
    pred_curve_params = compute_params_from_coefs(
        pred_coefs, candidate_tenors, parameter)

    state[:, 4] = pred_curve_params
    daily_train_data = (day, state, coefs, yesterday_coefs, pred_coefs)
    return daily_train_data


def plot_daily_train_data(config, daily_train_data, save_path='', plot_result=True, parameter='Alpha'):
    if isinstance(parameter, list):
        daily_train_datas = daily_train_data
        parameters = parameter
    else:
        assert(isinstance(parameter, str))
        daily_train_datas = [daily_train_data]
        parameters = [parameter]

    fontsize = 18
    fig = plt.figure(figsize=(7.5*len(parameters), 5))
    fig.subplots_adjust(wspace=0.25)
    for iparam, (daily_train_data, parameter) in enumerate(zip(daily_train_datas, parameters)):
        # state
        # tenor param today selected current_curve yesterday_curve

        day, state, coefs, yesterday_coefs, pred_coefs = daily_train_data
        curve_tenors = np.linspace(0., 3., num=361)[1:]
        yesterday_curve_params = compute_params_from_coefs(
            yesterday_coefs, curve_tenors, parameter)
        if parameter == 'Alpha':
            parameter_atm = 'alpha_{atm}'
            coef_name = 'p'
        elif parameter == 'Rho':
            parameter_atm = 'rho'
            coef_name = 'q'
        else:
            assert(parameter == 'Nu')
            parameter_atm = 'nu'
            coef_name = 'r'

        candidate_is_today = (state[:, 2] == 1.)
        candidate_is_selected = (state[:, 3] == 1.)

        pred_curve_params = compute_params_from_coefs(
            pred_coefs, curve_tenors, parameter)

        selected_tenors = np.sort(state[candidate_is_selected, 0])

        raw_selected_tenors = state[
            np.logical_and(candidate_is_today, candidate_is_selected), 0]
        raw_selected_params = state[
            np.logical_and(candidate_is_today, candidate_is_selected), 1]

        raw_not_selected_tenors = state[
            np.logical_and(candidate_is_today, ~candidate_is_selected), 0]
        raw_not_selected_params = state[
            np.logical_and(candidate_is_today, ~candidate_is_selected), 1]

        yesterday_selected_tenors = state[
            np.logical_and(~candidate_is_today, candidate_is_selected), 0]
        yesterday_selected_params = state[
            np.logical_and(~candidate_is_today, candidate_is_selected), 1]

        yesterday_not_selected_tenors = state[
            np.logical_and(~candidate_is_today, ~candidate_is_selected), 0]
        yesterday_not_selected_params = state[
            np.logical_and(~candidate_is_today, ~candidate_is_selected), 1]

        ax = fig.add_subplot(1, len(parameters), iparam+1)
        ax.set_title(r'%s, Parameteric Curve of $\%s$ at %s.' % (
            config.DATA_TYPE.replace('&', '\&'), parameter_atm, day.strftime('%Y-%m-%d') ), fontsize=fontsize)
        ax.set_xlabel('Maturity $t$ (year)', fontsize=fontsize)
        ax.set_ylabel(r'$\%s$' % parameter_atm, fontsize=fontsize)
        if coefs is not None:
            curve_params = compute_params_from_coefs(
                coefs, curve_tenors, parameter)
            ax.plot(curve_tenors, curve_params, 'g', label=r'$f_{\%s}(t; {\bf{%s}}^{*})$' %
                                                           (parameter.lower(), coef_name))
        ax.plot(curve_tenors, yesterday_curve_params, '--', color=(0.8, 0.8, 1), label=r'$f_{\%s}(t; {\bf{%s}}^{y})$' %
                                                                                       (parameter.lower(), coef_name))
        ax.plot(curve_tenors, pred_curve_params, 'k', label=r'$f_{\%s}(t; \widetilde{{\bf{%s}}^{*}})$' %
                                                            (parameter.lower(), coef_name))

        #ax.plot(yesterday_selected_tenors, yesterday_selected_params, 'bx',
        #        label=r'$\{\%s^{c, t} ~ | ~ t \in \mathcal{T}_C\} ~ \cap ~ \mathcal{V}_{\%s}$' %
        #              (parameter.lower(), parameter.lower()))
        #ax.plot(yesterday_not_selected_tenors, yesterday_not_selected_params, 'x', color=(0.8, 0.8, 1),
        #        label=r'$\{\%s^{c, t} ~ | ~ t \in \mathcal{T}_C\} ~ - ~ \mathcal{V}_{\%s}$' %
        #              (parameter.lower(), parameter.lower()))
        #ax.plot(raw_selected_tenors, raw_selected_params, 'ro',
        #        label=r'$\{\%s^{r, t} ~ | ~ t \in \mathcal{T}_R\} ~ \cap ~ \mathcal{V}_{\%s}$' %
        #              (parameter.lower(), parameter.lower()))
        #ax.plot(raw_not_selected_tenors, raw_not_selected_params, 'o', color=(1, 0.8, 0.8),
        #        label=r'$\{\%s^{r, t} ~ | ~ t \in \mathcal{T}_R\} ~ - ~ \mathcal{V}_{\%s}$' %
        #              (parameter.lower(), parameter.lower()))

        ax.plot(yesterday_selected_tenors, yesterday_selected_params, 'bx',
                label=r'$\{\%s^{c, t} ~ | ~ t \in \mathcal{T}_C\} ~ \cap ~ \widetilde{\mathcal{S}}_{\%s}^*$' %
                      (parameter.lower(), parameter.lower()))
        ax.plot(yesterday_not_selected_tenors, yesterday_not_selected_params, 'x', color=(0.8, 0.8, 1),
                label=r'$\{\%s^{c, t} ~ | ~ t \in \mathcal{T}_C\} ~ - ~ \widetilde{\mathcal{S}}_{\%s}^*$' %
                      (parameter.lower(), parameter.lower()))
        ax.plot(raw_selected_tenors, raw_selected_params, 'ro',
                label=r'$\{\%s^{r, t} ~ | ~ t \in \mathcal{T}_R\} ~ \cap ~ \widetilde{\mathcal{S}}_{\%s}^*$' %
                      (parameter.lower(), parameter.lower()))
        ax.plot(raw_not_selected_tenors, raw_not_selected_params, 'o', color=(1, 0.8, 0.8),
                label=r'$\{\%s^{r, t} ~ | ~ t \in \mathcal{T}_R\} ~ - ~ \widetilde{\mathcal{S}}_{\%s}^*$' %
                      (parameter.lower(), parameter.lower()))

        #ax.plot(yesterday_selected_tenors, yesterday_selected_params, 'bx',
        #        label=r'$\{\%s^{c, t} ~ | ~ t \in \mathcal{T}_C\}$' %
        #              (parameter.lower()))
        #ax.plot(yesterday_not_selected_tenors, yesterday_not_selected_params, 'x', color=(0.8, 0.8, 1),
        #        label=r'$\{\%s^{c, t} ~ | ~ t \in \mathcal{T}_C\}$' %
        #              (parameter.lower()))
        #ax.plot(raw_selected_tenors, raw_selected_params, 'ro',
        #        label=r'$\{\%s^{r, t} ~ | ~ t \in \mathcal{T}_R\}$' %
        #              (parameter.lower()))
        ax.set_xlim([-0.1, 3.1])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.legend(fontsize=12)
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    if plot_result:
        plt.show()


def augment_daily_train_data(daily_train_data, translate_amount, scale_amount, parameter='Alpha'):
    # state
    # tenor param today selected current_curve yesterday_curve

    day, state, coefs, yesterday_coefs, pred_coefs = daily_train_data
    state = np.copy(state)
    coefs = scale_coefs(coefs, scale_amount, parameter)
    coefs = translate_coefs(coefs, translate_amount, parameter)
    yesterday_coefs = scale_coefs(yesterday_coefs, scale_amount, parameter)
    yesterday_coefs = translate_coefs(yesterday_coefs, translate_amount, parameter)
    pred_coefs = scale_coefs(pred_coefs, scale_amount, parameter)
    pred_coefs = translate_coefs(pred_coefs, translate_amount, parameter)
    params = state[:, 1]
    params = scale_params(params, scale_amount)
    params = translate_params(params, translate_amount)
    state[:, 1] = params
    params = state[:, 4]
    params = scale_params(params, scale_amount)
    params = translate_params(params, translate_amount)
    state[:, 4] = params
    params = state[:, 5]
    params = scale_params(params, scale_amount)
    params = translate_params(params, translate_amount)
    state[:, 5] = params
    new_daily_train_data = (day, state, coefs, yesterday_coefs, pred_coefs)
    return new_daily_train_data


def randomly_augment_daily_train_data(daily_train_data, translate_range, scale_range, parameter='Alpha'):
    translate_amount = np.random.uniform(low=translate_range[0], high=translate_range[1])
    scale_amount = np.random.uniform(low=scale_range[0], high=scale_range[1])
    augmented_daily_train_data = augment_daily_train_data(
        daily_train_data, translate_amount, scale_amount, parameter)
    return augmented_daily_train_data


def normalize_state(state, parameter='Alpha'):
    # state
    # tenor param today selected current_curve yesterday_curve

    x_one = np.copy(state)
    if parameter == 'Alpha':
        x_one[:, 0] = (x_one[:, 0] - 0.8527) / 0.8208
        x_one[:, 1] = (x_one[:, 1] - 0.1403) / 0.03802
        x_one[:, 2] = (x_one[:, 2] - 0.5) / 0.5
        x_one[:, 3] = (x_one[:, 3] - 0.5) / 0.5
        x_one[:, 4] = (x_one[:, 4] - 0.1403) / 0.03802
        x_one[:, 5] = (x_one[:, 5] - 0.1403) / 0.03802
    elif parameter == 'Rho':
        x_one[:, 0] = (x_one[:, 0] - 0.8527) / 0.8208
        x_one[:, 1] = (x_one[:, 1] + 0.4213) / 0.1172
        x_one[:, 2] = (x_one[:, 2] - 0.5) / 0.5
        x_one[:, 3] = (x_one[:, 3] - 0.5) / 0.5
        x_one[:, 4] = (x_one[:, 4] + 0.4213) / 0.1172
        x_one[:, 5] = (x_one[:, 5] + 0.4213) / 0.1172
    else:
        assert(parameter == 'Nu')
        x_one[:, 0] = (x_one[:, 0] - 0.8527) / 0.8208
        x_one[:, 1] = (x_one[:, 1] - 1.0951) / 0.6418
        x_one[:, 2] = (x_one[:, 2] - 0.5) / 0.5
        x_one[:, 3] = (x_one[:, 3] - 0.5) / 0.5
        x_one[:, 4] = (x_one[:, 4] - 1.0951) / 0.6418
        x_one[:, 5] = (x_one[:, 5] - 1.0951) / 0.6418
    return x_one


def acquire_max_point_num(train_data):
    max_point_num = 0
    for daily_train_data in train_data:
        max_point_num = max(max_point_num, len(daily_train_data[1]))
    return max_point_num


def acquire_point_candidate_scores(logger, daily_train_data, score_count, parameter='Alpha'):
    _, initial_state, _, _, _ = daily_train_data
    candidate_probs, _, _ = compute_candidate_probs(daily_train_data, point_scores=None, onehot=False)

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

    point_scores = []
    for ipoint in range(len(initial_state)):
        point_score = 0.
        for i in range(score_count):
            while True:
                temp_daily_train_data = randomly_select_daily_train_data(daily_train_data, candidate_probs,
                                                                         fixed_point=ipoint, parameter=parameter)
                day, state, coefs, yesterday_coefs, pred_coefs = temp_daily_train_data
                pred_curve_params = state[:, 4]
                if np.max(pred_curve_params) < overflow_max and np.min(pred_curve_params) > overflow_min:  # not overflow
                    break

            point_score -= compute_l1_loss(pred_coefs, coefs, parameter=parameter)
        logger.info("   Done for [%d/%d]'th point." % (ipoint+1, len(state)))
        point_score /= score_count
        point_scores.append(point_score)
    point_scores = stats.zscore(np.array(point_scores))

    return point_scores


def acquire_all_point_candidate_scores(logger, all_data, score_count, parameter='Alpha'):
    # this may take a long time.
    logger.info('')
    logger.info('Computing point scores. This may take long time. Using score count: %d.' % score_count)
    logger.info('')
    start_time = time.time()
    all_point_scores = []
    for index in range(len(all_data)):
        daily_train_data = all_data[index]
        point_scores = acquire_point_candidate_scores(logger, daily_train_data, score_count, parameter)
        elapsed_time = time.time() - start_time
        logger.info("Process [%d/%d]. Elapsed time: %5u seconds." % (index + 1, len(all_data), elapsed_time))
        logger.debug('Computed point scores: ' + ', '.join('%.6f' % s for s in point_scores))
        all_point_scores.append(point_scores)
    logger.info('')
    logger.info('Computing ended.')
    logger.info('')
    return all_point_scores


def normalized_state_to_sparse_state(state, point_scores=None):
    # state
    # tenor param today selected current_curve yesterday_curve
    sparse_state = np.zeros((3 * DAYS_PER_YEAR, STATE_DIM - 1), dtype='float')
    point_locs = np.zeros((3 * DAYS_PER_YEAR,), dtype='bool')
    tenors = np.round(((state[:, 0] * 0.8208) + 0.8527) * DAYS_PER_YEAR).astype('int')
    tenor_count = defaultdict(int)
    for itenor, tenor in enumerate(tenors):
        if tenor_count[tenor] > 0:
            tenor += 1
            assert(tenor_count[tenor] == 0)
            tenors[itenor] = tenor
        tenor_count[tenor] += 1
    sparse_state[tenors-1, :] = state[:, 1:]
    point_locs[tenors-1] = True
    sparse_point_scores = None
    if point_scores is not None:
        assert(len(point_scores) == len(state))
        sparse_point_scores = np.zeros((3 * DAYS_PER_YEAR,), dtype='float')
        sparse_point_scores[tenors-1] = point_scores[:]
    if point_scores is None:
        return sparse_state, point_locs, tenors
    else:
        return sparse_state, sparse_point_scores, point_locs, tenors


def acquire_random_loss_batch(train_data, train_candidate_probs, parameter='Alpha'):
    # state
    # tenor param today selected current_curve yesterday_curve

    max_point_num = acquire_max_point_num(train_data)
    assert(max_point_num <= MAX_POINT_NUM)

    criterion = compute_l1_loss
    if parameter == 'Alpha':
        translate_range = ALPHA_TRANSLATE_RANGE
        scale_range = ALPHA_SCALE_RANGE
        overflow_max = ALPHA_OVERFLOW_MAX
        overflow_min = ALPHA_OVERFLOW_MIN
    elif parameter == 'Rho':
        translate_range = RHO_TRANSLATE_RANGE
        scale_range = RHO_SCALE_RANGE
        overflow_max = RHO_OVERFLOW_MAX
        overflow_min = RHO_OVERFLOW_MIN
    else:
        assert(parameter == 'Nu')
        translate_range = NU_TRANSLATE_RANGE
        scale_range = NU_SCALE_RANGE
        overflow_max = NU_OVERFLOW_MAX
        overflow_min = NU_OVERFLOW_MIN

    x = []
    pos = []
    y = []
    lengths = []
    for index in range(len(train_data)):
        daily_train_data = train_data[index]
        candidate_probs = train_candidate_probs[index]
        assert(np.sum(candidate_probs) > 0)
        count = 0
        while True:
            augmented_daily_train_data = randomly_augment_daily_train_data(
                daily_train_data, translate_range, scale_range, parameter)
            augmented_daily_train_data = randomly_select_daily_train_data(
                augmented_daily_train_data, candidate_probs, parameter=parameter)

            day, state, coefs, yesterday_coefs, pred_coefs = augmented_daily_train_data
            pred_curve_params = state[:, 4]
            if np.max(pred_curve_params) < overflow_max and np.min(pred_curve_params) > overflow_min:  # not overflow
                break
            else:
                count += 1
                if count == 10:
                    break
        if count == 10:
            continue
        # yesterday_loss = criterion(yesterday_coefs, coefs, parameter)
        pred_loss = criterion(pred_coefs, coefs, parameter=parameter)
        if parameter == 'Alpha':
            value = - pred_loss * ALPHA_LOSS_SCALE
        elif parameter == 'Rho':
            value = - pred_loss * RHO_LOSS_SCALE
        else:
            assert(parameter == 'Nu')
            value = - pred_loss * NU_LOSS_SCALE
        if np.fabs(value) > LOSS_THRESH_VALUE:
            value = float(np.sign(value) * LOSS_THRESH_VALUE)
        x_one = normalize_state(state, parameter)
        x.append(np.pad(x_one, ((0, max_point_num-len(x_one)), (0, 0)), mode='constant', constant_values=0))
        pos.append(np.pad(np.arange(len(x_one)), (0, max_point_num-len(x_one)),
                          mode='constant', constant_values=-1))
        y.append(value)
        lengths.append(len(x_one))
    x = np.stack(x, axis=0)
    pos = np.stack(pos, axis=0)
    y = np.stack(y, axis=0)
    lengths = np.stack(lengths, axis=0)

    return x, pos, y, lengths


def acquire_random_loss_sparse_batch(train_data, train_candidate_probs, parameter='Alpha'):
    # state
    # tenor param today selected current_curve yesterday_curve

    max_point_num = acquire_max_point_num(train_data)
    assert(max_point_num <= MAX_POINT_NUM)

    criterion = compute_l1_loss
    if parameter == 'Alpha':
        translate_range = ALPHA_TRANSLATE_RANGE
        scale_range = ALPHA_SCALE_RANGE
        overflow_max = ALPHA_OVERFLOW_MAX
        overflow_min = ALPHA_OVERFLOW_MIN
    elif parameter == 'Rho':
        translate_range = RHO_TRANSLATE_RANGE
        scale_range = RHO_SCALE_RANGE
        overflow_max = RHO_OVERFLOW_MAX
        overflow_min = RHO_OVERFLOW_MIN
    else:
        assert(parameter == 'Nu')
        translate_range = NU_TRANSLATE_RANGE
        scale_range = NU_SCALE_RANGE
        overflow_max = NU_OVERFLOW_MAX
        overflow_min = NU_OVERFLOW_MIN

    x = []
    y = []
    point_locs = []
    for index in range(len(train_data)):
        daily_train_data = train_data[index]
        candidate_probs = train_candidate_probs[index]
        assert(np.sum(candidate_probs) > 0)
        count = 0
        while True:
            augmented_daily_train_data = randomly_augment_daily_train_data(
                daily_train_data, translate_range, scale_range, parameter)
            augmented_daily_train_data = randomly_select_daily_train_data(
                augmented_daily_train_data, candidate_probs, parameter=parameter)

            day, state, coefs, yesterday_coefs, pred_coefs = augmented_daily_train_data
            pred_curve_params = state[:, 4]
            if np.max(pred_curve_params) < overflow_max and np.min(pred_curve_params) > overflow_min:  # not overflow
                break
            else:
                count += 1
                if count == 10:
                    break
        if count == 10:
            continue
        # yesterday_loss = criterion(yesterday_coefs, coefs, parameter)
        pred_loss = criterion(pred_coefs, coefs, parameter=parameter)
        if parameter == 'Alpha':
            value = - pred_loss * ALPHA_LOSS_SCALE
        elif parameter == 'Rho':
            value = - pred_loss * RHO_LOSS_SCALE
        else:
            assert(parameter == 'Nu')
            value = - pred_loss * NU_LOSS_SCALE
        if np.fabs(value) > LOSS_THRESH_VALUE:
            value = float(np.sign(value) * LOSS_THRESH_VALUE)
        x_one, point_locs_one, _ = normalized_state_to_sparse_state(normalize_state(state, parameter))
        x.append(x_one)
        y.append(value)
        point_locs.append(point_locs_one)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    point_locs = np.stack(point_locs, axis=0)

    return x, y, point_locs


def acquire_random_candidate_batch(train_data, train_point_scores, parameter='Alpha'):
    # state
    # tenor param today selected current_curve yesterday_curve

    max_point_num = acquire_max_point_num(train_data)
    assert(max_point_num <= MAX_POINT_NUM)

    if parameter == 'Alpha':
        translate_range = ALPHA_TRANSLATE_RANGE
        scale_range = ALPHA_SCALE_RANGE
    elif parameter == 'Rho':
        translate_range = RHO_TRANSLATE_RANGE
        scale_range = RHO_SCALE_RANGE
    else:
        assert(parameter == 'Nu')
        translate_range = NU_TRANSLATE_RANGE
        scale_range = NU_SCALE_RANGE

    x = []
    pos = []
    y = []
    lengths = []
    for index in range(len(train_data)):
        daily_train_data = train_data[index]
        augmented_daily_train_data = randomly_augment_daily_train_data(
            daily_train_data, translate_range, scale_range, parameter)
        day, state, coefs, yesterday_coefs, pred_coefs = augmented_daily_train_data
        x_one = normalize_state(state, parameter)
        x.append(np.pad(x_one, ((0, max_point_num-len(x_one)), (0, 0)), mode='constant', constant_values=0))
        pos.append(np.pad(np.arange(len(x_one)), (0, max_point_num-len(x_one)),
                          mode='constant', constant_values=-1))
        point_scores = train_point_scores[index]
        assert(len(point_scores) == len(x_one))
        y.append(np.pad(point_scores, (0, max_point_num-len(point_scores)), mode='constant', constant_values=0))
        lengths.append(len(x_one))
    x = np.stack(x, axis=0)
    pos = np.stack(pos, axis=0)
    y = np.stack(y, axis=0)
    lengths = np.stack(lengths, axis=0)

    return x, pos, y, lengths


def acquire_random_candidate_sparse_batch(train_data, train_point_scores, parameter='Alpha'):
    # state
    # tenor param today selected current_curve yesterday_curve

    max_point_num = acquire_max_point_num(train_data)
    assert(max_point_num <= MAX_POINT_NUM)

    if parameter == 'Alpha':
        translate_range = ALPHA_TRANSLATE_RANGE
        scale_range = ALPHA_SCALE_RANGE
    elif parameter == 'Rho':
        translate_range = RHO_TRANSLATE_RANGE
        scale_range = RHO_SCALE_RANGE
    else:
        assert(parameter == 'Nu')
        translate_range = NU_TRANSLATE_RANGE
        scale_range = NU_SCALE_RANGE

    x = []
    y = []
    point_locs = []
    for index in range(len(train_data)):
        daily_train_data = train_data[index]
        augmented_daily_train_data = randomly_augment_daily_train_data(
            daily_train_data, translate_range, scale_range, parameter)
        day, state, coefs, yesterday_coefs, pred_coefs = augmented_daily_train_data
        point_scores = train_point_scores[index]
        assert(len(point_scores) == len(state))
        x_one, y_one, point_locs_one, _ = normalized_state_to_sparse_state(
            normalize_state(state, parameter), point_scores)
        x.append(x_one)
        y.append(y_one)
        point_locs.append(point_locs_one)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    point_locs = np.stack(point_locs, axis=0)

    return x, y, point_locs

