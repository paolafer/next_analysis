"""
Set of functions used in the topology analysis
"""

import numpy as np
from . histo_functions import exp, gauss

### Errors from the fit, bin by bin

def var_on_signal_events(f, x):
    #g = f.values[2]/(2*np.pi)**.5/f.values[4] * np.exp(-0.5*(x-f.values[3])**2./f.values[4]**2.)
    ### derivative of gaussian with respect to the three variables: amplitude, mean and sigma
    der_g_amp = 1/(2*np.pi)**.5/f.values[4] * np.exp(-0.5*(x-f.values[3])**2./f.values[4]**2.)
    der_g_mu = f.values[2]/(2*np.pi)**.5 * np.exp(-0.5*(x-f.values[3])**2./f.values[4]**2.) * (x-f.values[3])/f.values[4]**3
    der_g_sigma = f.values[2] * np.exp(-0.5*(x-f.values[3])**2./f.values[4]**2.)*(x-f.values[3])**2/(2*np.pi)**.5/f.values[4]**4 - f.values[2] * np.exp(-0.5*(x-f.values[3])**2./f.values[4]**2.)/(2*np.pi)**.5/f.values[4]**2

    sigma2_Ns = der_g_amp**2 * f.cov[2, 2] + der_g_mu**2 * f.cov[3, 3] + der_g_sigma**2 * f.cov[4, 4]
    sigma2_Ns += 2 * der_g_amp * der_g_mu * f.cov[2,3]
    sigma2_Ns += 2 * der_g_amp * der_g_sigma * f.cov[2,4]
    sigma2_Ns += 2 * der_g_mu * der_g_sigma * f.cov[3,4]

    return sigma2_Ns


def var_on_background_events(f, x):
    #e = f.values[0] * np.exp(x/f.values[1])
    ### derivative of exponential with respect to the two variables: amplitude and decay constant
    der_e_a0 = np.exp(x/f.values[1])
    der_e_tau = - f.values[0] * np.exp(x/f.values[1]) * x/f.values[1]**2

    sigma2_Nb = der_e_a0**2 * f.cov[0,0] + der_e_tau**2 * f.cov[1,1]
    sigma2_Nb += 2 * der_e_a0 * der_e_tau * f.cov[0,1]

    return sigma2_Nb


def find_fractions(x, fit_result, e_min, e_max, e_min_plot, e_max_plot, nbins_plot):
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

    #tot = s+b
    fs = s/n_tot
    fb = b/n_tot
    err_fs = np.sqrt(1/n_tot**2*var_s)
    err_fb = np.sqrt(1/n_tot**2*var_b)

    return(n_tot, fs, fb, err_fs, err_fb)


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
