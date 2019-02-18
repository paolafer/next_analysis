"""
Set of functions used in the topology analysis
"""

import numpy as np
from . histo_functions import exp, gauss

### Errors from the fit, bin by bin

def var_on_signal1_events(f, x):
    #g = f.values[2]/(2*np.pi)**.5/f.values[4] * np.exp(-0.5*(x-f.values[3])**2./f.values[4]**2.)
    ### derivative of gaussian with respect to the three variable: amplitude, mean and sigma
    der_g_amp = 1/(2*np.pi)**.5/f.values[4] * np.exp(-0.5*(x-f.values[3])**2./f.values[4]**2.)
    der_g_mu = f.values[2]/(2*np.pi)**.5 * np.exp(-0.5*(x-f.values[3])**2./f.values[4]**2.) * (x-f.values[3])/f.values[4]**3
    der_g_sigma = f.values[2] * np.exp(-0.5*(x-f.values[3])**2./f.values[4]**2.)*(x-f.values[3])**2/(2*np.pi)**.5/f.values[4]**4 - f.values[2] * np.exp(-0.5*(x-f.values[3])**2./f.values[4]**2.)/(2*np.pi)**.5/f.values[4]**2

    sigma2_Ns = der_g_amp**2 * f.cov[2, 2] + der_g_mu**2 * f.cov[3, 3] + der_g_sigma**2 * f.cov[4, 4]
    sigma2_Ns += 2 * der_g_amp * der_g_mu * f.cov[2,3]
    sigma2_Ns += 2 * der_g_amp * der_g_sigma * f.cov[2,4]
    sigma2_Ns += 2 * der_g_mu * der_g_sigma * f.cov[3,4]

    return sigma2_Ns


def var_on_signal2_events(f, x):
    #g = f.values[5]/(2*np.pi)**.5/f.values[7] * np.exp(-0.5*(x-f.values[6])**2./f.values[7]**2.)
    ### derivative of gaussian with respect to the three variable: amplitude, mean and sigma
    der_g_amp = 1/(2*np.pi)**.5/f.values[7] * np.exp(-0.5*(x-f.values[6])**2./f.values[7]**2.)
    der_g_mu = f.values[5]/(2*np.pi)**.5 * np.exp(-0.5*(x-f.values[6])**2./f.values[7]**2.) * (x-f.values[6])/f.values[7]**3
    der_g_sigma = f.values[5] * np.exp(-0.5*(x-f.values[6])**2./f.values[7]**2.)*(x-f.values[6])**2/(2*np.pi)**.5/f.values[7]**4 - f.values[5] * np.exp(-0.5*(x-f.values[6])**2./f.values[7]**2.)/(2*np.pi)**.5/f.values[7]**2

    sigma2_Ns = der_g_amp**2 * f.cov[5, 5] + der_g_mu**2 * f.cov[6, 6] + der_g_sigma**2 * f.cov[7, 7]
    sigma2_Ns += 2 * der_g_amp * der_g_mu * f.cov[5,6]
    sigma2_Ns += 2 * der_g_amp * der_g_sigma * f.cov[5,7]
    sigma2_Ns += 2 * der_g_mu * der_g_sigma * f.cov[6,7]

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

    s1 = s2 = b = 0.
    var_s1 = var_s2 = var_b = 0
    for i in range(low_bin, high_bin+1):
        centre_value = e_min_plot + i * bin_width + bin_width/2
        s1 += gauss(centre_value, fit_result.values[2], fit_result.values[3], fit_result.values[4])
        s2 += gauss(centre_value, fit_result.values[5], fit_result.values[6], fit_result.values[7])
        b += exp(centre_value, fit_result.values[0], fit_result.values[1])

        var_s1 += var_on_signal1_events(fit_result, centre_value)
        var_s2 += var_on_signal2_events(fit_result, centre_value)
        var_b += var_on_background_events(fit_result, centre_value)

    tot = s1+s2+b
    fs = (s1+s2)/(s1+s2+b)
    fb = b/(s1+s2+b)
    err_fs = np.sqrt((b/(s1+s2+b)**2)**2*var_s1 + (b/(s1+s2+b)**2)**2*var_s2 + ((s1+s2)/(s1+s2+b)**2)**2*var_b)
    err_fb = np.sqrt(((s1+s2)/(s1+s2+b)**2)**2*var_b + (b/(s1+s2+b)**2)**2*(var_s1+var_s2))

    return(tot, fs, fb, err_fs, err_fb)
