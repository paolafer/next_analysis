"""
Set of functions for histogramming and data fitting
"""

import numpy as np
import matplotlib.pyplot as plt

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

blue = tableau20[0]
green = tableau20[4]
fucsia = tableau20[6]


# useful to normalize histograms
def get_weights(data):
    return np.repeat(1.0/len(data), len(data))


def plot_comparison(data_one, data_two, nbins, color_one=fucsia, color_two=blue, xlabel='', ylabel='', legend_loc=1):

    bins = np.histogram(np.hstack((data_two, data_one)),
                        bins=nbins)[1] #get the bin edges
    weights_one = get_weights(data_one)
    weights_two = get_weights(data_two)

    plt.hist(data_one, label='MC', weights=weights_one, color=color_one,
             bins=bins, histtype='step', stacked=True, fill=False, linewidth=4.0, linestyle=':')
    plt.hist(data_two, label='Data', weights=weights_two, color=color_two,
             bins=bins, histtype='step', stacked=True, fill=False, linewidth=3.0)

    lnd = plt.legend(loc=legend_loc)
    #plt.scatter(z_spec,e_spec,marker='.')
    #plt.xlim([0.,2000])
    #plt.ylim([0,0.02])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

