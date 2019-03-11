"""
Set of functions used in the topology analysis
"""

import numpy as np
from scipy import integrate as integrate

from . histo_functions import exp, gauss, expgauss

### Partial derivative of gaussian and exponential functions,
### with respect to their parameters
def der_g_amp(x, mu, sigma, amp):
    return 1/(2*np.pi)**.5/sigma * np.exp(-0.5*(x-mu)**2./sigma**2.)
def der_g_mu(x, mu, sigma, amp):
    return amp/(2*np.pi)**.5 * np.exp(-0.5*(x-mu)**2./sigma**2.) * (x-mu)/sigma**3
def der_g_sigma(x, mu, sigma, amp):
    return amp * np.exp(-0.5*(x-mu)**2./sigma**2.)*(x-mu)**2/(2*np.pi)**.5/sigma**4 - \
           amp*np.exp(-0.5*(x-mu)**2./sigma**2.)/(2*np.pi)**.5/sigma**2

def der_e_a0(x, tau, a0):
    return np.exp(x/tau)
def der_e_tau(x, tau, a0):
    return - a0 * np.exp(x/tau) * x/tau**2


def var_on_signal_events(f, x):
    """
    This function calculates the variance of the number of signal events of one bin
    """
    d_g_amp = der_g_amp(x, f.values[3], f.values[4], f.values[2])
    d_g_mu = der_g_mu(x, f.values[3], f.values[4], f.values[2])
    d_g_sigma = der_g_sigma(x, f.values[3], f.values[4], f.values[2])

    sigma2_Ns = d_g_amp**2 * f.cov[2, 2] + d_g_mu**2 * f.cov[3, 3] + d_g_sigma**2 * f.cov[4, 4]
    sigma2_Ns += 2 * d_g_amp * d_g_mu * f.cov[2,3]
    sigma2_Ns += 2 * d_g_amp * d_g_sigma * f.cov[2,4]
    sigma2_Ns += 2 * d_g_mu * d_g_sigma * f.cov[3,4]

    return sigma2_Ns


def var_on_background_events(f, x):
    """
    This function calculates the variance of the number of background events of one bin
    """
    d_e_a0 = der_e_a0(x, f.values[1], f.values[0])
    d_e_tau = der_e_tau(x, f.values[1], f.values[0])

    sigma2_Nb = d_e_a0**2 * f.cov[0,0] + d_e_tau**2 * f.cov[1,1]
    sigma2_Nb += 2 * d_e_a0 * d_e_tau * f.cov[0,1]

    return sigma2_Nb


def find_fractions(x, fit_result, e_min, e_max, e_min_plot, e_max_plot, nbins_plot):
    """
    This function returns the fraction of signal and background events and their errors
    calculated bin by bin. It assumes that the number of signal and backgrund events
    are not correlated (which is not correct), and that the total number of events is
    given by the sum of the two.
    """
    low_bin = np.digitize(e_min, x)
    high_bin = np.digitize(e_max, x)

    bin_width = (e_max_plot - e_min_plot) / nbins_plot

    s = b = 0.
    var_s = var_b = 0
    for i in range(low_bin, high_bin+1):
        centre_value = e_min_plot + i * bin_width + bin_width/2
        s += gauss(centre_value, fit_result.values[2], fit_result.values[3], fit_result.values[4])
        b += exp(centre_value, fit_result.values[0], fit_result.values[1])

        var_s += var_on_signal_events(fit_result, centre_value)
        var_b += var_on_background_events(fit_result, centre_value)

    tot = s+b
    fs = s/(s+b)
    fb = b/(s+b)
    err_fs = np.sqrt((b/(s+b)**2)**2*var_s + (s/(s+b)**2)**2*var_b)
    err_fb = np.sqrt((s/(s+b)**2)**2*var_b + (b/(s+b)**2)**2*var_s)

    return(tot, fs, fb, err_fs, err_fb)


def find_fractions_n_tot_no_error(x, y, fit_result, e_min, e_max, e_min_plot, e_max_plot, nbins_plot):
    """
    This function returns the fraction of signal and background events and their errors
    calculated bin by bin. It assumes that the total number of events has no error.
    """
    low_bin = np.digitize(e_min, x)
    high_bin = np.digitize(e_max, x)

    n_tot = sum(y[low_bin:high_bin+1])

    bin_width = (e_max_plot - e_min_plot) / nbins_plot

    s = b = 0.
    var_s = var_b = 0
    for i in range(low_bin, high_bin+1):
        centre_value = e_min_plot + i * bin_width + bin_width/2
        s += gauss(centre_value, fit_result.values[2], fit_result.values[3], fit_result.values[4])
        b += exp(centre_value, fit_result.values[0], fit_result.values[1])

        var_s += var_on_signal_events(fit_result, centre_value)
        var_b += var_on_background_events(fit_result, centre_value)

    fs = s/n_tot
    fb = b/n_tot
    err_fs = np.sqrt(1/n_tot**2*var_s)
    err_fb = np.sqrt(1/n_tot**2*var_b)

    return(n_tot, fs, fb, err_fs, err_fb)


def find_number_of_events(x, fit_result, e_min, e_max, e_min_plot, e_max_plot, nbins_plot):
    """
    This function returns the number of events of signal and background and their errors
    calculated bin by bin.
    """
    low_bin = np.digitize(e_min, x)
    high_bin = np.digitize(e_max, x)

    bin_width = (e_max_plot - e_min_plot) / nbins_plot

    s = b = 0.
    var_s = var_b = 0
    for i in range(low_bin, high_bin+1):
        centre_value = e_min_plot + i * bin_width + bin_width/2
        s += myhf.gauss(centre_value, fit_result.values[2], fit_result.values[3], fit_result.values[4])
        b += myhf.exp(centre_value, fit_result.values[0], fit_result.values[1])

        var_s += var_on_signal_events(fit_result, centre_value)
        var_b += var_on_background_events(fit_result, centre_value)

    tot = s+b

    return(tot, s, b, np.sqrt(var_s), np.sqrt(var_b))


def find_number_of_events_integrals(f, e_min, e_max, bin_width):
    """
    This function returns the number of events of signal and background and their errors
    calculated using integrals.
    """
    integral_tot   = integrate.quad(expgauss, e_min, e_max, args=(f.values[0], f.values[1], f.values[2], f.values[3], f.values[4]))
    integral_exp   = integrate.quad(exp, e_min, e_max, args=(f.values[0], f.values[1]))
    integral_gauss = integrate.quad(gauss, e_min, e_max, args=(f.values[2], f.values[3], f.values[4]))

    tot    = integral_tot[0]/bin_width
    n_bckg = integral_exp[0]/bin_width
    n_sig  = integral_gauss[0]/bin_width

    i_der_amp   = integrate.quad(der_g_amp,   e_min, e_max, args=(f.values[3], f.values[4], f.values[2]))
    i_der_mu    = integrate.quad(der_g_mu,    e_min, e_max, args=(f.values[3], f.values[4], f.values[2]))
    i_der_sigma = integrate.quad(der_g_sigma, e_min, e_max, args=(f.values[3], f.values[4], f.values[2]))

    var_n_sig = (i_der_amp[0]*f.cov[2,2]/bin_width)**2 + (i_der_mu[0]*f.cov[3,3]/bin_width)**2 + (i_der_sigma[0]*f.cov[4,4]/bin_width)**2 + 2/bin_width**2*i_der_mu[0]*i_der_sigma[0]*f.cov[3,4] + 2/bin_width**2*i_der_mu[0]*i_der_amp[0]*f.cov[2,3] + 2/bin_width**2*i_der_sigma[0]*i_der_amp[0]*f.cov[2,4]

    i_der_a0  = integrate.quad(der_e_a0,  e_min, e_max, args=(f.values[1], f.values[0]))
    i_der_tau = integrate.quad(der_e_tau, e_min, e_max, args=(f.values[1], f.values[0]))

    var_n_bckg = (i_der_a0[0]*f.cov[0,0]/bin_width)**2 + (i_der_tau[0]*f.cov[1,1]/bin_width)**2 + 2/bin_width**2*i_der_tau[0]*i_der_a0[0]*f.cov[0,1]

    return tot, n_sig, n_bckg, np.sqrt(var_n_sig), np.sqrt(var_n_bckg)


### Fractions and errors from Extended Maximum Likelihood unbinned fit
### with the range of the fit equal to the range in which we want to know
### the number of events
def find_fractions_ml_unbinned(fit_result):

    signal = 'Ns'
    background = 'Nb'
    s = fit_result.values[signal]
    b = fit_result.values[background]
    err_s = fit_result.errors[signal]
    err_b = fit_result.errors[background]
    cov_sb = fit_result.covariance[signal, background]

    tot = s+b
    fs = s/(s+b)
    fb = b/(s+b)

    err_fs = np.sqrt((b/(s+b)**2)**2*err_s**2 + (s/(s+b)**2)**2*err_b**2 - 2*b*s/(s+b)**4*cov_sb)
    err_fb = np.sqrt((s/(s+b)**2)**2*err_b**2 + (b/(s+b)**2)**2*err_s**2 - 2*b*s/(s+b)**4*cov_sb)

    return (tot, fs, fb, err_fs, err_fb)
