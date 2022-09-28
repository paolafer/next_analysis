"""
Set of functions for histogramming and data fitting
"""

import numpy as np
import matplotlib.pyplot as plt
import textwrap

def twogauss(x, amp0, mu0, sigma0, amp, mu, sigma):
    return  amp0/(2*np.pi)**.5/sigma0 * np.exp(-0.5*(x-mu0)**2./sigma0**2.) + amp/(2*np.pi)**.5/sigma * np.exp(-0.5*(x-mu)**2./sigma**2.)

def gauss(x, amp, mu, sigma):
    return amp/(2*np.pi)**.5/sigma * np.exp(-0.5*(x-mu)**2./sigma**2.)

def exp(x, a0, tau):
    return a0 * np.exp(x/tau)

def expgauss(x, a0, tau, amp, mu, sigma):
    return a0 * np.exp(x/tau) + amp/(2*np.pi)**.5/sigma * np.exp(-0.5*(x-mu)**2./sigma**2.)

def exp2gauss(x, a0, tau, amp, mu, sigma, amp1, mu1, sigma1):
    return a0 * np.exp(x/tau) + amp/(2*np.pi)**.5/sigma * np.exp(-0.5*(x-mu)**2./sigma**2.) + amp1/(2*np.pi)**.5/sigma1 * np.exp(-0.5*(x-mu1)**2./sigma1**2.)

def polygauss(x, a0, a1, a2, amp, mu, sigma):
    return a0 + x*a1 + x*x*a2 + amp/(2*np.pi)**.5/sigma * np.exp(-0.5*(x-mu)**2./sigma**2.)

def gtext_res(values, errors, chi2, un='pe'):
    """
    Build a string to be displayed within a matplotlib plot.
    Show mean, sigma, chi2 and resolution.
    """
    return textwrap.dedent("""
        $\mu$ = {:.2f} $\pm$ {:.2f} {}
        $\sigma$ = {:.2f} $\pm$ {:.2f} {}
        R (%) = {:.2f} $\pm$ {:.2f}
        R - FWHM (%) = {:.2f} $\pm$ {:.2f}
        $\chi^2$/N$_\mathrm{{dof}}$  = {:.2f}
        """.format(values[0] , errors[0], un,
             values[1] , errors[1], un,
             100*values[1]/values[0], 100*np.sqrt((errors[1]/values[0])**2+values[1]/values[0]**2*errors[0]**2),
             2.35*100*values[1]/values[0], 2.35*100*np.sqrt((errors[1]/values[0])**2+values[1]/values[0]**2*errors[0]**2), chi2))

def gtext(values, errors, chi2, un='ps'):
    """
    Build a string to be displayed within a matplotlib plot.
    Show mean, FWHM and chi2.
    """
    return textwrap.dedent("""
        $\mu$ = {:.2f} $\pm$ {:.2f} {}
        FWHM = {:.2f} $\pm$ {:.2f} {}
        $\chi^2$/N$_\mathrm{{dof}}$  = {:.2f}
        """.format(values[0], errors[0], un,
                   2.35*values[1], 2.35*errors[1], un, chi2))


def gtext_angle(values, errors, chi2, min_r, max_r):
    """
    Build a string to be displayed within a matplotlib plot.
    Shows mean, FWHM in deg, FWHM in mm, corresponding to min and max radius and chi2.
    """
    return textwrap.dedent("""
        $\mu$ = {:.2f} $\pm$ {:.2f} deg
        FWHM = {:.2f} $\pm$ {:.2f} deg
        FWHM (arc) = {:.2f}-{:.2f} mm
        $\chi^2$/N$_\mathrm{{dof}}$  = {:.2f}
        """.format(values[0], errors[0],
             2.35*values[1], 2.35*errors[1], 2.35*values[1]/180*np.pi*min_r, 2.35*values[1]/180*np.pi*max_r, chi2))

def gtext_2gaus(values, errors, chi2, un='ps'):
    """
    Build a string to be displayed within a matplotlib plot.
    Show two gaussians.
    """
    return textwrap.dedent("""
        $\mu_1$ = {:.2f} $\pm$ {:.2f} {:}
        FWHM$_1$ = {:.2f} $\pm$ {:.2f} {:}
        $\mu_2$ = {:.2f} $\pm$ {:.2f} {:}
        FWHM$_2$ = {:.2f} $\pm$ {:.2f} {:}
        $\chi^2$/N$_\mathrm{{dof}}$  = {:.2f}
        """.format(values[1], errors[1], un,
             2.35*values[2], 2.35*errors[2], un,
             values[4], errors[4], un,
             2.35*values[5], 2.35*errors[5], un,
             chi2))

def gtext_2gaus_res(values, errors, chi2, un='ps'):
    """
    Build a string to be displayed within a matplotlib plot.
    Show two gaussians.
    """
    return textwrap.dedent("""
        $\mu_1$ = {:.2f} $\pm$ {:.2f} {:}
        FWHM$_1$ = {:.2f} $\pm$ {:.2f} {:}
        res$_1$ = {:.2f} $\pm$ {:.2f}
        $\mu_2$ = {:.2f} $\pm$ {:.2f} {:}
        FWHM$_2$ = {:.2f} $\pm$ {:.2f} {:}
        res$_2$ = {:.2f} $\pm$ {:.2f}
        $\chi^2$/N$_\mathrm{{dof}}$  = {:.2f}
        """.format(values[1], errors[1], un,
             2.35*values[2], 2.35*errors[2], un,
             2.35*values[2]/values[1], 100*np.sqrt((errors[2]/values[1])**2+values[2]/values[1]**2*errors[1]**2),
             values[4], errors[4], un,
             2.35*values[5], 2.35*errors[5], un,
             2.35*values[5]/values[4], 100*np.sqrt((errors[5]/values[4])**2+values[5]/values[4]**2*errors[4]**2),
             chi2))

def gtext_2gaus_angle(values, errors, chi2, min_r, max_r):
    """
    Build a string to be displayed within a matplotlib plot.
    """
    return textwrap.dedent("""
        $\mu_1$ = {:.2f} $\pm$ {:.2f} deg
        FWHM$_1$ = {:.2f} $\pm$ {:.2f} deg
        $\mu_2$ = {:.2f} $\pm$ {:.2f} deg
        FWHM$_2$ = {:.2f} $\pm$ {:.2f} deg
        FWHM$_1$ (arc) = {:.2f}-{:.2f} mm
        FWHM$_2$ (arc) = {:.2f}-{:.2f} mm
        $\chi^2$/N$_\mathrm{{dof}}$  = {:.2f}
        """.format(values[1], errors[1],
             2.35*values[2], 2.35*errors[2],
             values[4], errors[4],
             2.35*values[5], 2.35*errors[5],
             2.35*values[2]/180*np.pi*min_r, 2.35*values[2]/180*np.pi*max_r,
             2.35*values[5]/180*np.pi*min_r, 2.35*values[5]/180*np.pi*max_r,
             chi2))

## style
# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

#blue = tableau20[0]
#green = tableau20[4]
#fucsia = tableau20[6]


# useful to normalize histograms
def get_weights(data, norm):
    if norm:
        return np.repeat(1.0/len(data), len(data))
    else:
        return np.repeat(1.0, len(data))


def plot_comparison(data1, data2, therange, nbins, norm=True, color1=tableau20[6], color2=tableau20[4], xlabel='', ylabel='', plt_lbl1='', plt_lbl2='', legend_loc=1):

    bins = np.histogram(np.hstack((data1, data2)),
                        nbins, therange)[1] #get the bin edges

    weights1 = get_weights(data1, norm)
    weights2 = get_weights(data2, norm)

    plt.hist(data1, label=plt_lbl1, weights=weights1, color=color1,
             bins=bins, histtype='step', stacked=True, fill=False, linewidth=4.0, linestyle=':')
    plt.hist(data2, label=plt_lbl2, weights=weights2, color=color2,
             bins=bins, histtype='step', stacked=True, fill=False, linewidth=3.0)

    lnd = plt.legend(loc=legend_loc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

