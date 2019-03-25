"""
Set of functions used in the topology analysis
"""

import numpy as np
import math
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
        s += gauss(centre_value, fit_result.values[2], fit_result.values[3], fit_result.values[4])
        b += exp(centre_value, fit_result.values[0], fit_result.values[1])

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

def find_number_of_events_integrals_analytic_binned_fit(f, Nsigma, bin_size):
    a0, tau, amp, mu, sigma = f.values
    e_int       = a0*tau*(math.exp((Nsigma*sigma-mu)/tau)-math.exp((-Nsigma*sigma-mu)/tau))/bin_size
    e_der_a0    = tau*(math.exp((Nsigma*sigma-mu)/tau)-math.exp((-Nsigma*sigma-mu)/tau))/bin_size
    e_der_tau   = a0/tau* ( (mu-Nsigma*sigma+tau)*math.exp((-mu+Nsigma*sigma)/tau)-(mu+Nsigma*sigma+tau)*math.exp((-mu-Nsigma*sigma)/tau))/bin_size
    e_der_amp   = 0
    e_der_mu    = -a0*(math.exp((Nsigma*sigma-mu)/tau)-math.exp((-Nsigma*sigma-mu)/tau))/bin_size
    e_der_sigma = a0*Nsigma*(math.exp((Nsigma*sigma-mu)/tau)+math.exp((-Nsigma*sigma-mu)/tau))/bin_size 
    var_bck = e_der_a0**2* f.cov[0,0]\
            + e_der_tau**2*f.cov[1,1]\
            + e_der_amp**2*f.cov[2,2]\
            + e_der_mu**2*f.cov[3,3] \
            + 2*e_der_sigma**2+f.cov[4,4]\
            + 2*e_der_a0*e_der_tau*f.cov[0,1]\
            + 2*e_der_a0*e_der_amp*f.cov[0,2]\
            + 2*e_der_a0*e_der_mu*f.cov[0,3]\
            + 2*e_der_a0*e_der_sigma*f.cov[0,4]\
            + 2*e_der_tau*e_der_amp*f.cov[1,2]\
            + 2*e_der_tau*e_der_mu*f.cov[1,3]\
            + 2*e_der_tau*e_der_sigma*f.cov[1,4]\
            + 2*e_der_amp*e_der_mu*f.cov[2,3]\
            + 2*e_der_amp*e_der_sigma*f.cov[2,4]\
            + 2*e_der_mu*e_der_sigma*f.cov[3,4]
    g_int       = param_amp*math.erf(Nsigma/math.sqrt(2))/bin_size
    g_der_a0    = 0
    g_der_tau   = 0
    g_der_amp   = math.erf(Nsigma/math.sqrt(2))/bin_size
    g_der_mu    = 0
    g_der_sigma = 0
    var_sig = g_der_a0**2* f.cov[0,0]\
            + g_der_tau**2*f.cov[1,1]\
            + g_der_amp**2*f.cov[2,2]\
            + g_der_mu**2*f.cov[3,3] \
            + g_der_sigma**2*f.cov[4,4]\
            + 2*g_der_a0*g_der_tau*f.cov[0,1]\
            + 2*g_der_a0*g_der_amp*f.cov[0,2]\
            + 2*g_der_a0*g_der_mu*f.cov[0,3]\
            + 2*g_der_a0*g_der_sigma*f.cov[0,4]\
            + 2*g_der_tau*g_der_amp*f.cov[1,2]\
            + 2*g_der_tau*g_der_mu*f.cov[1,3]\
            + 2*g_der_tau*g_der_sigma*f.cov[1,4]\
            + 2*g_der_amp*g_der_mu*f.cov[2,3]\
            + 2*g_der_amp*g_der_sigma*f.cov[2,4]\
            + 2*g_der_mu*g_der_sigma*f.cov[3,4]
    return g_int+e_int, g_int, e_int, np.sqrt(var_sig), np.sqrt(var_bck)

def find_number_of_events_integrals_analytic_fixed_borders_binned_fit(f, Emin, Emax, bin_size):
    a0, tau, amp, mu, sigma = f.values
    exp_help = math.exp(-Emin/tau)-math.exp(-Emax/tau)
    e_int       = a0*tau*exp_help/bin_size
    e_der_a0    = tau*(math.exp((Nsigma*sigma-mu)/tau)-math.exp((-Nsigma*sigma-mu)/tau))/bin_size
    e_der_tau   = a0/tau* (math.exp(-Emin/tau)*(Emin+tau)-math.exp(-Emax/tau)*(Emax+tau))/bin_size
    e_der_amp   = 0
    e_der_mu    = 0
    var_bck = e_der_a0**2* f.cov[0,0]\
            + e_der_tau**2*f.cov[1,1]\
            + e_der_amp**2*f.cov[2,2]\
            + e_der_mu**2*f.cov[3,3] \
            + 2*e_der_sigma**2+f.cov[4,4]\
            + 2*e_der_a0*e_der_tau*f.cov[0,1]\
            + 2*e_der_a0*e_der_amp*f.cov[0,2]\
            + 2*e_der_a0*e_der_mu*f.cov[0,3]\
            + 2*e_der_a0*e_der_sigma*f.cov[0,4]\
            + 2*e_der_tau*e_der_amp*f.cov[1,2]\
            + 2*e_der_tau*e_der_mu*f.cov[1,3]\
            + 2*e_der_tau*e_der_sigma*f.cov[1,4]\
            + 2*e_der_amp*e_der_mu*f.cov[2,3]\
            + 2*e_der_amp*e_der_sigma*f.cov[2,4]\
            + 2*e_der_mu*e_der_sigma*f.cov[3,4]
    g_int       = 0.5*amp*(-math.erf((-Emax+mu)/math.sqrt(2)/sigma)+math.erf((-Emin+mu)/math.sqrt(2)/sigma))/bin_size
    g_der_a0    = 0
    g_der_tau   = 0
    g_der_amp   = 0.5*(-math.erf((-Emax+mu)/math.sqrt(2)/sigma)+math.erf((-Emin+mu)/math.sqrt(2)/sigma))/bin_size
    g_der_mu    = amp/math.sqrt(2*math.pi)/sigma*(-math.exp(-(Emax-mu)**2/2/sigma**2)+math.exp(-(Emin-mu)**2/2/sigma**2))/bin_size
    g_der_sigma = amp/math.sqrt(2*math.pi)/sigma**2*((-Emax+mu)*math.exp(-(Emax-mu)**2/2/sigma**2)+(Emin-mu)*math.exp(-(Emin-mu)**2/2/sigma**2))/bin_size
    var_sig = g_der_a0**2* f.cov[0,0]\
            + g_der_tau**2*f.cov[1,1]\
            + g_der_amp**2*f.cov[2,2]\
            + g_der_mu**2*f.cov[3,3] \
            + g_der_sigma**2*f.cov[4,4]\
            + 2*g_der_a0*g_der_tau*f.cov[0,1]\
            + 2*g_der_a0*g_der_amp*f.cov[0,2]\
            + 2*g_der_a0*g_der_mu*f.cov[0,3]\
            + 2*g_der_a0*g_der_sigma*f.cov[0,4]\
            + 2*g_der_tau*g_der_amp*f.cov[1,2]\
            + 2*g_der_tau*g_der_mu*f.cov[1,3]\
            + 2*g_der_tau*g_der_sigma*f.cov[1,4]\
            + 2*g_der_amp*g_der_mu*f.cov[2,3]\
            + 2*g_der_amp*g_der_sigma*f.cov[2,4]\
            + 2*g_der_mu*g_der_sigma*f.cov[3,4]
    return g_int+e_int, g_int, e_int, np.sqrt(var_sig), np.sqrt(var_bck)


def find_number_of_events_integrals_analytic_unbinned_fit(fit_result, Nsigma, efmin, efmax):
    Ns         = fit_result.values['Ns']
    err_Ns     = fit_result.errors['Ns']
    mu         = fit_result.values['mu']
    err_mu     = fit_result.errors['mu']
    sigma      = fit_result.values['sigma']
    err_sigma  = fit_result.errors['sigma']
    Nb         = fit_result.values['Nb']
    err_Nb     = fit_result.errors['Nb']
    tau        = fit_result.values['tau']
    err_tau    = fit_result.errors['tau']
    cov_Ns_Nb     = fit_result.covariance['Ns', 'Nb']
    cov_Ns_mu     = fit_result.covariance['Ns', 'mu']
    cov_Ns_sigma  = fit_result.covariance['Ns', 'sigma']
    cov_Ns_tau    = fit_result.covariance['Ns', 'tau']
    cov_mu_sigma  = fit_result.covariance['mu', 'sigma']
    cov_mu_Nb     = fit_result.covariance['mu', 'Nb']
    cov_mu_tau    = fit_result.covariance['mu', 'tau']
    cov_sigma_Nb  = fit_result.covariance['sigma', 'Nb']
    cov_sigma_tau = fit_result.covariance['sigma', 'tau']
    cov_Nb_tau    = fit_result.covariance['Nb', 'tau']

    int_g        = Ns * math.erf(Nsigma/math.sqrt(2))
    g_der_Ns     = math.erf(Nsigma/math.sqrt(2))
    g_der_mu     = 0
    g_der_sigma  = 0
    g_der_Nb     = 0
    g_der_tau    = 0

    exp_help     = math.exp(-efmin/tau)-math.exp(-efmax/tau)
    int_e        = Nb*math.exp(-(mu+Nsigma*sigma)/tau)*(-1+math.exp(2*Nsigma*sigma/tau))/exp_help
    e_der_Ns     = 0
    e_der_mu     = -Nb*math.exp(-(mu+Nsigma*sigma)/tau)*(-1+math.exp(2*Nsigma*sigma/tau))/exp_help/tau
    e_der_sigma  = Nb*math.exp(-(mu+Nsigma*sigma)/tau)*(1+math.exp(2*Nsigma*sigma/tau))/exp_help/tau*Nsigma
    e_der_Nb     = math.exp(-(mu+Nsigma*sigma)/tau)*(-1+math.exp(2*Nsigma*sigma/tau))/exp_help  
    e_der_tau    = Nb/exp_help**2/tau**2*(\
                   (efmin-mu-sigma*Nsigma)*math.exp(-(efmin+mu+Nsigma*sigma)/tau)+\
                   (efmax-mu+sigma*Nsigma)*math.exp(-(efmax+mu-Nsigma*sigma)/tau)-\
                   (efmin-mu+sigma*Nsigma)*math.exp(-(efmin+mu-Nsigma*sigma)/tau)-\
                   (efmax-mu-sigma*Nsigma)*math.exp(-(efmax+mu+Nsigma*sigma)/tau))


    var_g   =     g_der_Ns    **2           * err_Ns**2    \
            +     g_der_mu    **2           * err_mu**2    \
            +     g_der_sigma **2           * err_sigma**2 \
            +     g_der_Nb    **2           * err_Nb**2    \
            +     g_der_tau   **2           * err_tau**2   \
            + 2 * g_der_Ns    * g_der_mu    * cov_Ns_mu    \
            + 2 * g_der_Ns    * g_der_sigma * cov_Ns_sigma \
            + 2 * g_der_Ns    * g_der_Nb    * cov_Ns_Nb    \
            + 2 * g_der_Ns    * g_der_tau   * cov_Ns_tau   \
            + 2 * g_der_mu    * g_der_sigma * cov_mu_sigma \
            + 2 * g_der_mu    * g_der_Nb    * cov_mu_Nb    \
            + 2 * g_der_mu    * g_der_tau   * cov_mu_tau   \
            + 2 * g_der_sigma * g_der_Nb    * cov_sigma_Nb \
            + 2 * g_der_sigma * g_der_tau   * cov_sigma_tau\
            + 2 * g_der_Nb    * g_der_tau   * cov_Nb_tau   \

    var_e   =     e_der_Ns    **2           * err_Ns**2    \
            +     e_der_mu    **2           * err_mu**2    \
            +     e_der_sigma **2           * err_sigma**2 \
            +     e_der_Nb    **2           * err_Nb**2    \
            +     e_der_tau   **2           * err_tau**2   \
            + 2 * e_der_Ns    * e_der_mu    * cov_Ns_mu    \
            + 2 * e_der_Ns    * e_der_sigma * cov_Ns_sigma \
            + 2 * e_der_Ns    * e_der_Nb    * cov_Ns_Nb    \
            + 2 * e_der_Ns    * e_der_tau   * cov_Ns_tau   \
            + 2 * e_der_mu    * e_der_sigma * cov_mu_sigma \
            + 2 * e_der_mu    * e_der_Nb    * cov_mu_Nb    \
            + 2 * e_der_mu    * e_der_tau   * cov_mu_tau   \
            + 2 * e_der_sigma * e_der_Nb    * cov_sigma_Nb \
            + 2 * e_der_sigma * e_der_tau   * cov_sigma_tau\
            + 2 * e_der_Nb    * e_der_tau   * cov_Nb_tau

    return int_g+int_e, int_g, int_e, np.sqrt(var_g), np.sqrt(var_e)

def find_number_of_events_integrals_analytic_fixed_borders_unbinned_fit(fit_result, emin, emax, efmin, efmax):
    Ns         = fit_result.values['Ns']
    err_Ns     = fit_result.errors['Ns']
    mu         = fit_result.values['mu']
    err_mu     = fit_result.errors['mu']
    sigma      = fit_result.values['sigma']
    err_sigma  = fit_result.errors['sigma']
    Nb         = fit_result.values['Nb']
    err_Nb     = fit_result.errors['Nb']
    tau        = fit_result.values['tau']
    err_tau    = fit_result.errors['tau']

    cov_Ns_Nb     = fit_result.covariance['Ns', 'Nb']
    cov_Ns_mu     = fit_result.covariance['Ns', 'mu']
    cov_Ns_sigma  = fit_result.covariance['Ns', 'sigma']
    cov_Ns_tau    = fit_result.covariance['Ns', 'tau']
    cov_mu_sigma  = fit_result.covariance['mu', 'sigma']
    cov_mu_Nb     = fit_result.covariance['mu', 'Nb']
    cov_mu_tau    = fit_result.covariance['mu', 'tau']
    cov_sigma_Nb  = fit_result.covariance['sigma', 'Nb']
    cov_sigma_tau = fit_result.covariance['sigma', 'tau']
    cov_Nb_tau    = fit_result.covariance['Nb', 'tau']

    int_g       = 0.5*Ns*(-math.erf((-emax+mu)/math.sqrt(2)/sigma)+math.erf((-emin+mu)/math.sqrt(2)/sigma))
    g_der_Ns    = 0.5*(-math.erf((-emax+mu)/math.sqrt(2)/sigma)+math.erf((-emin+mu)/math.sqrt(2)/sigma))
    g_der_mu    = Ns/math.sqrt(2*math.pi)/sigma*(-math.exp(-(emax-mu)**2/2/sigma**2)+math.exp(-(emin-mu)**2/2/sigma**2))
    g_der_sigma = Ns/math.sqrt(2*math.pi)/sigma**2*((-emax+mu)*math.exp(-(emax-mu)**2/2/sigma**2)+(emin-mu)*math.exp(-(emin-mu)**2/2/sigma**2))
    g_der_tau   = 0
    g_der_Nb   = 0

    exp_help     = math.exp(-efmin/tau)-math.exp(-efmax/tau)
    int_e        = Nb*(-math.exp(-emax/tau)+math.exp(-emin/tau))/exp_help
    e_der_Ns     = 0
    e_der_mu     = 0
    e_der_sigma  = 0
    e_der_Nb     = (-math.exp(-emax/tau)+math.exp(-emin/tau))/exp_help
    e_der_tau    = Nb/exp_help**2/tau**2*(\
                   (efmin-emax)*math.exp(-(efmin+emax)/tau)+\
                   (efmax-emin)*math.exp(-(efmax+emin)/tau)-\
                   (efmin-emin)*math.exp(-(efmin+emin)/tau)-\
                   (efmax-emax)*math.exp(-(efmax+emax)/tau))

    var_g   =     g_der_Ns    **2           * err_Ns**2    \
            +     g_der_mu    **2           * err_mu**2    \
            +     g_der_sigma **2           * err_sigma**2 \
            +     g_der_Nb    **2           * err_Nb**2    \
            +     g_der_tau   **2           * err_tau**2   \
            + 2 * g_der_Ns    * g_der_mu    * cov_Ns_mu    \
            + 2 * g_der_Ns    * g_der_sigma * cov_Ns_sigma \
            + 2 * g_der_Ns    * g_der_Nb    * cov_Ns_Nb    \
            + 2 * g_der_Ns    * g_der_tau   * cov_Ns_tau   \
            + 2 * g_der_mu    * g_der_sigma * cov_mu_sigma \
            + 2 * g_der_mu    * g_der_Nb    * cov_mu_Nb    \
            + 2 * g_der_mu    * g_der_tau   * cov_mu_tau   \
            + 2 * g_der_sigma * g_der_Nb    * cov_sigma_Nb \
            + 2 * g_der_sigma * g_der_tau   * cov_sigma_tau\
            + 2 * g_der_Nb    * g_der_tau   * cov_Nb_tau   \

    var_e   =     e_der_Ns    **2           * err_Ns**2    \
            +     e_der_mu    **2           * err_mu**2    \
            +     e_der_sigma **2           * err_sigma**2 \
            +     e_der_Nb    **2           * err_Nb**2    \
            +     e_der_tau   **2           * err_tau**2   \
            + 2 * e_der_Ns    * e_der_mu    * cov_Ns_mu    \
            + 2 * e_der_Ns    * e_der_sigma * cov_Ns_sigma \
            + 2 * e_der_Ns    * e_der_Nb    * cov_Ns_Nb    \
            + 2 * e_der_Ns    * e_der_tau   * cov_Ns_tau   \
            + 2 * e_der_mu    * e_der_sigma * cov_mu_sigma \
            + 2 * e_der_mu    * e_der_Nb    * cov_mu_Nb    \
            + 2 * e_der_mu    * e_der_tau   * cov_mu_tau   \
            + 2 * e_der_sigma * e_der_Nb    * cov_sigma_Nb \
            + 2 * e_der_sigma * e_der_tau   * cov_sigma_tau\
            + 2 * e_der_Nb    * e_der_tau   * cov_Nb_tau   

    return int_g+int_e, int_g, int_e, np.sqrt(var_g), np.sqrt(var_e)
