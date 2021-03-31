import sys
import os

import numpy as np
import pandas as pd

import scipy
from scipy import interpolate
from scipy import optimize
from scipy import stats

from .kap_constants import *


def _alpha(p, t):
    return (p[0] + (p[3]/p[4]) * (1 - np.exp(-p[4]*t)) /
            (p[4]*t) + (p[1]/p[2]) * np.exp(-p[2]*t))


def _rho(p, t):
    return p[0] + p[1] * np.exp(-t*p[2])


def _nu(p, t):
    return p[0] + p[1] * np.power(t, p[2]) * np.exp(p[3]*t)


def _translate_alpha(p, v):
    p[0] += v
    return p


def _translate_rho(p, v):
    p[0] += v
    return p


def _translate_nu(p, v):
    p[0] += v
    return p


def translate_coefs(coefs, v, parameter='Alpha'):
    if coefs is None:  # No coefs
        return None
    p = np.copy(coefs)
    if parameter == 'Alpha':
        return _translate_alpha(p, v)
    elif parameter == 'Rho':
        return _translate_rho(p, v)
    else:
        assert(parameter == 'Nu')
        return _translate_nu(p, v)


def _scale_alpha(p, v):
    p[0] *= v
    p[1] *= v
    p[3] *= v
    return p


def _scale_rho(p, v):
    p[0] *= v
    p[1] *= v
    return p


def _scale_nu(p, v):
    p[0] *= v
    p[1] *= v
    return p


def scale_coefs(coefs, v, parameter='Alpha'):
    if coefs is None:  # No coefs
        return None
    p = np.copy(coefs)
    if parameter == 'Alpha':
        return _scale_alpha(p, v)
    elif parameter == 'Rho':
        return _scale_rho(p, v)
    else:
        assert(parameter == 'Nu')
        return _scale_nu(p, v)


def translate_params(params, v):
    return params + v


def scale_params(params, v):
    return params * v


def compute_params_from_coefs(coefs, tenors, parameter='Alpha'):
    assert(coefs is not None)
    p = coefs
    t = tenors
    assert(len(p.shape) == 1)
    if parameter == 'Alpha':
        return _alpha(p, t)
    elif parameter == 'Rho':
        return _rho(p, t)
    else:
        assert(parameter == 'Nu')
        return _nu(p, t)


def _l2_loss(pred, gold):
    return np.mean((pred - gold)**2)


def fit_param_curve(params, tenors, init_coefs, parameter='Alpha'):
    assert(init_coefs is not None)
    assert(len(params.shape) == 1 and len(tenors.shape) == 1
           and len(params) == len(tenors))
    loss = (lambda x: _l2_loss(
        compute_params_from_coefs(x, tenors, parameter), params))
    y = optimize.minimize(loss, init_coefs)
    return y.x


def compute_irs(irs, t):
    tck = interpolate.splrep(IRS_TENORS, irs, k=3, s=0)
    R = interpolate.splev(t, tck, der=0)
    dR = interpolate.splev(t, tck, der=1)
    r = dR * t + R
    return r, R


def compute_dividend(dividend, t):
    dt = 1. / DAYS_PER_YEAR
    n = int(np.round(t / dt))
    tq, q = dividend
    iq = np.round(tq / dt).astype(int)
    assert(((iq[1:] - iq[:-1]) > 0).all())
    q_exp_sum = np.sum(np.exp(q[iq < n]))
    return np.log(1. + q_exp_sum) / t


def smile(alpha, rho, nu, t, m):  # m : Moneyness, log(K/F)
    eps = 1e-6
    m_thresh = 5.
    m = np.clip(m, -m_thresh, m_thresh)
    y = nu/alpha * (-m)
    fs = np.sqrt(1.-2.*rho*y+y**2)
    fl = np.log((fs + y - rho) / (1 - rho))
    f1 = y/fl
    f2 = 1. - 1./2.*rho*y + (-1./4.*rho**2 + 1./6.)*y**2
    if isinstance(m, float):
        f = f1 if np.fabs(m) >= eps else f2
    else:
        assert(isinstance(m, np.ndarray))
        f1[np.fabs(m) < eps] = 0.
        f2[np.fabs(m) >= eps] = 0.
        f = f1 + f2
    c = 1. + (1./4.*rho*nu*alpha + (2.-3.*rho**2)/24.*nu**2) * t
    return alpha*f*c


def smile_derivatives(alpha, rho, nu, t, m):  # m : Moneyness, log(K/F)
    eps = 1e-6
    m_thresh = 5.
    m = np.clip(m, -m_thresh, m_thresh)
    y = nu/alpha * (-m)
    fs = np.sqrt(1.-2.*rho*y+y**2)
    fl = np.log((fs + y - rho) / (1 - rho))

    f1 = y/fl
    f2 = 1. - 1./2.*rho*y + (-1./4.*rho**2 + 1./6.)*y**2
    df1 = (fl*fs - y) / (fl*fl*fs)
    df2 = -1./2.*rho + 2.*(-1./4.*rho**2 + 1./6.)*y - 1./8.*(6.*rho**2 - 5.)*rho*y**2
    ddf1 = (fl*(3*rho*y-y**2-2) + 2*fs*y) / (fl*fs)**3
    ddf2 = (2.*(-1./4.*rho**2 + 1./6.) - 1./4.*(6.*rho**2 - 5.)*rho*y +
            12.*(-5./16.*rho**4 + 1./3.*rho**2 - 17./360.)*y**2)
    if isinstance(m, float):
        f = f1 if np.fabs(m) >= eps else f2
        df = df1 if np.fabs(m) >= eps else df2
        ddf = ddf1 if np.fabs(m) >= eps else ddf2
    else:
        assert(isinstance(m, np.ndarray))
        f1[np.fabs(m) < eps] = 0.
        f2[np.fabs(m) >= eps] = 0.
        f = f1 + f2
        df1[np.fabs(m) < eps] = 0.
        df2[np.fabs(m) >= eps] = 0.
        df = df1 + df2
        ddf1[np.fabs(m) < eps] = 0.
        ddf2[np.fabs(m) >= eps] = 0.
        ddf = ddf1 + ddf2

    c = 1. + (1./4.*rho*nu*alpha + (2.-3.*rho**2)/24.*nu**2) * t
    S = alpha*f*c
    dS = -nu*df*c
    ddS = (nu*df + nu*nu*ddf/alpha) * c
    if isinstance(m, float):
        dS = dS if np.fabs(m) < m_thresh else 0.
        ddS = ddS if np.fabs(m) < m_thresh else 0.
    else:
        dS[np.fabs(m) >= m_thresh] = 0.
        ddS[np.fabs(m) >= m_thresh] = 0.
    return S, dS, ddS


def vol_surface(alpha_coefs, rho_coefs, nu_coefs, t, m):  # m : Moneyness, log(K/F)
    alpha = _alpha(alpha_coefs, t)
    rho = _rho(rho_coefs, t)
    nu = _nu(nu_coefs, t)
    return smile(alpha, rho, nu, t, m)


def implied_vol(t, m, c, r, R, position):  # m : Moneyness, log(K/F), c: Option Price, log(C/F)
    y = -m
    alpha = np.exp(c) / np.exp(-R*t)

    Ry_C = 2.*alpha - np.exp(y) + 1.
    Ry_P = 2.*alpha + np.exp(y) - 1.
    if isinstance(m, float):
        Ry = Ry_C if position == 'C' else Ry_P
    else:
        assert(isinstance(m, np.ndarray))
        Ry_C[position == 'P'] = 0.
        Ry_P[position == 'C'] = 0.
        Ry = Ry_C + Ry_P

    A = (np.exp((1. - 2./np.pi)*y) - np.exp(-(1. - 2./np.pi)*y))**2
    B = 4.*(np.exp(2.*y/np.pi) + np.exp(-2.*y/np.pi)) - 2.*np.exp(-y)*(
            np.exp((1. - 2./np.pi)*y) + np.exp(-(1. - 2./np.pi)*y))*(np.exp(2.*y) + 1. - Ry**2)
    C = np.exp(-2.*y)*(Ry**2 - (np.exp(y)-1.)**2)*((np.exp(y)+1.)**2-Ry**2)
    beta = 2.*C/(B+np.sqrt(np.clip(B**2 + 4.*A*C, 0, None)))
    gamma = -(np.pi/2.)*np.log(beta)
    #print(alpha[C < 0])
    #print(Ry[C < 0])
    #print(t[C < 0])
    #print(m[C < 0])
    #print(np.sort(B)[:30])
    #print(np.sort(beta)[:30])
    #assert((~np.isnan(Ry)).all())
    #assert((~np.isnan(A)).all())
    #assert((~np.isnan(B)).all())
    #assert((~np.isnan(C)).all())
    #assert((~np.isnan(beta)).all())
    #assert((beta > 0).all())
    #assert((~np.isnan(gamma)).all())
    #assert((~np.isnan(y)).all())

    A2y_C1 = 1./2. + 1./2. * np.sqrt(1. - np.exp(-2.*2.*y/np.pi))
    c0_C1 = -R*t + np.log(np.exp(y)*A2y_C1 - 1./2.)

    A2y_C2 = 1./2. - 1./2. * np.sqrt(1. - np.exp(2.*2.*y/np.pi))
    c0_C2 = -R*t + np.log(np.exp(y)/2. - A2y_C2)

    A2y_P1 = 1./2. - 1./2. * np.sqrt(1. - np.exp(-2.*2.*y/np.pi))
    c0_P1 = -R*t + np.log(1./2. - np.exp(y)*A2y_P1)

    A2y_P2 = 1./2. + 1./2. * np.sqrt(1. - np.exp(2.*2.*y/np.pi))
    c0_P2 = -R*t + np.log(A2y_P2 - np.exp(y)/2.)

    sigma1 = 1./np.sqrt(t) * (np.sqrt(gamma+y) + np.sqrt(gamma-y))
    sigma2 = 1./np.sqrt(t) * (np.sqrt(gamma+y) - np.sqrt(gamma-y))
    sigma3 = 1./np.sqrt(t) * (-np.sqrt(gamma+y) + np.sqrt(gamma-y))
    if isinstance(m, float):
        if position == 'C':
            if y >= 0:
                sigma = sigma2 if c <= c0_C1 else sigma1
            else:
                sigma = sigma3 if c <= c0_C2 else sigma1
        else:
            assert(position == 'P')
            if y >= 0:
                sigma = sigma2 if c <= c0_P1 else sigma1
            else:
                sigma = sigma3 if c <= c0_P2 else sigma1
    else:
        assert(isinstance(m, np.ndarray))
        sigma = np.zeros_like(sigma1)
        sigma[np.logical_and.reduce([position == 'C', y >= 0, c <= c0_C1])] = sigma2[
            np.logical_and.reduce([position == 'C', y >= 0, c <= c0_C1])]
        sigma[np.logical_and.reduce([position == 'C', y >= 0, c > c0_C1])] = sigma1[
            np.logical_and.reduce([position == 'C', y >= 0, c > c0_C1])]
        sigma[np.logical_and.reduce([position == 'C', y < 0, c <= c0_C2])] = sigma3[
            np.logical_and.reduce([position == 'C', y < 0, c <= c0_C2])]
        sigma[np.logical_and.reduce([position == 'C', y < 0, c > c0_C2])] = sigma1[
            np.logical_and.reduce([position == 'C', y < 0, c > c0_C2])]
        sigma[np.logical_and.reduce([position == 'P', y >= 0, c <= c0_P1])] = sigma2[
            np.logical_and.reduce([position == 'P', y >= 0, c <= c0_P1])]
        sigma[np.logical_and.reduce([position == 'P', y >= 0, c > c0_P1])] = sigma1[
            np.logical_and.reduce([position == 'P', y >= 0, c > c0_P1])]
        sigma[np.logical_and.reduce([position == 'P', y < 0, c <= c0_P2])] = sigma3[
            np.logical_and.reduce([position == 'P', y < 0, c <= c0_P2])]
        sigma[np.logical_and.reduce([position == 'P', y < 0, c > c0_P2])] = sigma1[
            np.logical_and.reduce([position == 'P', y < 0, c > c0_P2])]
    return sigma


def local_vol_surface(alpha_coefs, rho_coefs, nu_coefs, t, m, r, R):  # m : Moneyness, log(K/F)
    alpha = _alpha(alpha_coefs, t)
    rho = _rho(rho_coefs, t)
    nu = _nu(nu_coefs, t)
    y = m - t*R
    S, dS, ddS = smile_derivatives(alpha, rho, nu, t, m)
    if isinstance(m, float):
        y = y if np.isfinite(m) else 0.
    else:
        y[~np.isfinite(m)] = 0.

    eps = 1e-6 / float(DAYS_PER_YEAR)
    alpha_eps = _alpha(alpha_coefs, t+eps)
    rho_eps = _rho(rho_coefs, t+eps)
    nu_eps = _nu(nu_coefs, t+eps)
    S_eps = smile(alpha_eps, rho_eps, nu_eps, t+eps, m)
    alpha_meps = _alpha(alpha_coefs, t-eps)
    rho_meps = _rho(rho_coefs, t-eps)
    nu_meps = _nu(nu_coefs, t-eps)
    S_meps = smile(alpha_meps, rho_meps, nu_meps, t-eps, m)
    dSdT = (S_eps**2 - S_meps**2) / (4.*eps*S)
    #dSdT = (S_eps**2 - S**2) / (2.*eps*S)

    f = S**2 + 2*S*t*(dSdT + r*dS)
    g = (1. - y*dS/S)**2 + S*t*(dS - 1./4.*S*t*dS**2 + ddS)
    return np.sqrt(np.clip(f / g, 0., None))


def monte_carlo(t0, m0, position,
                alpha_coefs=None, rho_coefs=None, nu_coefs=None, irs=None, dividend=None,
                sigma=None, num_sample=10000):  # m : Moneyness, log(K/F)
    assert(isinstance(t0, float) and isinstance(m0, float))  # No array input
    dt = 1. / DAYS_PER_YEAR
    n = int(np.round(t0 / dt))
    m = np.array([0] * num_sample)
    dt_sqrt = np.sqrt(dt)
    if sigma is None:
        assert(alpha_coefs is not None and rho_coefs is not None and nu_coefs is not None)
        sigma = vol_surface(alpha_coefs, rho_coefs, nu_coefs, t0, m0)
    #print(sigma)
    #print(n)
    if irs is not None:
        _, R0 = compute_irs(irs, t0)
    else:
        R0 = 0.
    iq, q = None, None
    if dividend is not None:
        tq, q = dividend
        iq = np.round(tq / dt).astype(int)
        assert(((iq[1:] - iq[:-1]) > 0).all())
        iq = iq.tolist()
    for i in range(n):
        t = float(i+1) * dt  # Using Future T for stability.
        if irs is not None:
            r, _ = compute_irs(irs, t)
        else:
            r = 0.
        w = np.random.randn(num_sample//2)
        w = np.concatenate([w, -w], axis=0)
        assert(len(w) == num_sample)
        if iq is not None and i in iq:
            dem = 1. + R0*dt + sigma*dt_sqrt*w - np.exp(q[iq.index(i)] - m)
        else:
            dem = 1. + R0*dt + sigma*dt_sqrt*w
        dem[dem <= 0] = 0.
        m = m + np.log(dem)
    #print(m)
    #print(np.sort(m))
    if position == 'C':
        C = np.mean(np.exp(-R0*t0) * (np.exp(m) - np.exp(m0)).clip(min=0))
    else:
        assert(position == 'P')
        C = np.mean(np.exp(-R0*t0) * (np.exp(m0) - np.exp(m)).clip(min=0))
    #print(C)
    return np.log(C)  # log(C/S0)


def local_monte_carlo(t0, m0, position,
                      alpha_coefs, rho_coefs, nu_coefs, irs=None, dividend=None,
                      num_sample=10000):  # m : Moneyness, log(K/F)
    assert(isinstance(t0, float) and isinstance(m0, float))  # No array input
    n = int(np.round(t0 * DAYS_PER_YEAR))
    m = np.array([0] * num_sample)
    dt = 1. / DAYS_PER_YEAR
    dt_sqrt = np.sqrt(dt)
    #print(n)
    if irs is not None:
        _, R0 = compute_irs(irs, t0)
    else:
        R0 = 0.
    iq, q = None, None
    if dividend is not None:
        tq, q = dividend
        iq = np.round(tq * DAYS_PER_YEAR).astype(int).tolist()
    for i in range(n):
        t = float(i+1) / DAYS_PER_YEAR  # Using Future T for stability.
        if irs is not None:
            r, R = compute_irs(irs, t)
        else:
            r, R = 0., 0.
        sigma = local_vol_surface(alpha_coefs, rho_coefs, nu_coefs, t, m, r, R)
        w = np.random.randn(num_sample//2)
        w = np.concatenate([w, -w], axis=0)
        assert(len(w) == num_sample)
        if iq is not None and i in iq:
            dem = 1. + r*dt + sigma*dt_sqrt*w - np.exp(q[iq.index(i)] - m)
        else:
            dem = 1. + r*dt + sigma*dt_sqrt*w
        dem[dem <= 0] = 0.
        m = m + np.log(dem)
    #print(m)
    #print(np.sort(m))
    if position == 'C':
        C = np.mean(np.exp(-R0*t0) * (np.exp(m) - np.exp(m0)).clip(min=0))
    else:
        assert(position == 'P')
        C = np.mean(np.exp(-R0*t0) * (np.exp(m0) - np.exp(m)).clip(min=0))
    #print(C)
    return np.log(C)  # log(C/S0)


def local_vix(t0,
              alpha_coefs, rho_coefs, nu_coefs, irs=None, dividend=None,
              num_sample=10000):  # m : Moneyness, log(K/F)
    assert(isinstance(t0, float))  # No array input
    n = int(np.round(t0 * DAYS_PER_YEAR))
    m = np.array([0.] * num_sample)
    vix = np.array([0.] * num_sample)
    dt = 1. / DAYS_PER_YEAR
    dt_sqrt = np.sqrt(dt)
    #print(n)
    iq, q = None, None
    if dividend is not None:
        tq, q = dividend
        iq = np.round(tq * DAYS_PER_YEAR).astype(int).tolist()
    for i in range(n):
        t = float(i+1) / DAYS_PER_YEAR  # Using Future T for stability.
        if irs is not None:
            r, R = compute_irs(irs, t)
        else:
            r, R = 0., 0.
        sigma = local_vol_surface(alpha_coefs, rho_coefs, nu_coefs, t, m, r, R)
        vix = vix + sigma**2 * dt
        w = np.random.randn(num_sample//2)
        w = np.concatenate([w, -w], axis=0)
        assert(len(w) == num_sample)
        if iq is not None and i in iq:
            dem = 1. + r*dt + sigma*dt_sqrt*w - np.exp(q[iq.index(i)] - m)
        else:
            dem = 1. + r*dt + sigma*dt_sqrt*w
        dem[dem <= 0] = 0.
        m = m + np.log(dem)
    vix = vix / t0
    return 100. * np.sqrt(vix.mean())


def compute_l1_loss(pred_coefs, gold_coefs, curve_tenors=STANDARD_TENORS, parameter='Alpha'):
    #curve_tenors = np.linspace(0., 3., num=37)[1:]
    #curve_tenors = STANDARD_TENORS
    pred_curve_params = compute_params_from_coefs(
        pred_coefs, curve_tenors, parameter)
    gold_curve_params = compute_params_from_coefs(
        gold_coefs, curve_tenors, parameter)
    return np.mean(np.fabs(pred_curve_params - gold_curve_params))


def compute_l2_loss(pred_coefs, gold_coefs, curve_tenors=STANDARD_TENORS, parameter='Alpha'):
    #curve_tenors = np.linspace(0., 3., num=37)[1:]
    #curve_tenors = STANDARD_TENORS
    pred_curve_params = compute_params_from_coefs(
        pred_coefs, curve_tenors, parameter)
    gold_curve_params = compute_params_from_coefs(
        gold_coefs, curve_tenors, parameter)
    return np.sqrt(np.mean((pred_curve_params - gold_curve_params) ** 2))


def compute_smile_l1_loss(pred_alpha_coefs, pred_rho_coefs, pred_nu_coefs,
                          gold_alpha_coefs, gold_rho_coefs, gold_nu_coefs,
                          t, m_min, m_max):
    pred_alpha = _alpha(pred_alpha_coefs, t)
    pred_rho = _rho(pred_rho_coefs, t)
    pred_nu = _nu(pred_nu_coefs, t)
    gold_alpha = _alpha(gold_alpha_coefs, t)
    gold_rho = _rho(gold_rho_coefs, t)
    gold_nu = _nu(gold_nu_coefs, t)
    M = np.linspace(m_min, m_max, 100)
    pred_smile_values = smile(pred_alpha, pred_rho, pred_nu, t, M)
    gold_smile_values = smile(gold_alpha, gold_rho, gold_nu, t, M)
    return np.mean(np.fabs(pred_smile_values - gold_smile_values))


def compute_smile_l2_loss(pred_alpha_coefs, pred_rho_coefs, pred_nu_coefs,
                          gold_alpha_coefs, gold_rho_coefs, gold_nu_coefs,
                          t, m_min, m_max):
    pred_alpha = _alpha(pred_alpha_coefs, t)
    pred_rho = _rho(pred_rho_coefs, t)
    pred_nu = _nu(pred_nu_coefs, t)
    gold_alpha = _alpha(gold_alpha_coefs, t)
    gold_rho = _rho(gold_rho_coefs, t)
    gold_nu = _nu(gold_nu_coefs, t)
    M = np.linspace(m_min, m_max, 100)
    #pred_smile_values = smile(pred_alpha, pred_rho, pred_nu, t, M)
    pred_smile_values = smile(pred_alpha, pred_rho, pred_nu, t, M)
    gold_smile_values = smile(gold_alpha, gold_rho, gold_nu, t, M)
    #print(t, m_min, m_max)
    #print(pred_alpha, pred_rho, pred_nu)
    #print(gold_alpha, gold_rho, gold_nu)
    #print(pred_smile_values)
    #print(gold_smile_values)
    #print(np.mean((pred_smile_values - gold_smile_values) ** 2))
    #assert(0 == 1)
    #return (pred_alpha - gold_alpha) ** 2
    return np.sqrt(np.mean((pred_smile_values - gold_smile_values) ** 2))
