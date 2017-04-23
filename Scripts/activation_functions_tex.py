import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

x = np.arange(-2,2,0.01)
sigmoid = np.exp(x)/(1 + np.exp(x))
tanh = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
relu = np.maximum(x-0.015, 0)
cut_off = 0.1

a = 0.333 * x[x<0]
b = x[x>=0] + 0.015
lrelu = np.append(a, b)

plt.plot(x,sigmoid, x, tanh, x, relu, x, lrelu, linewidth='2')
plt.legend(['sigmoid', "hyperbolic tangent", "rectifier", "leaky rectifier ($\\alpha=0.333$)"], loc=2)

tikz_save('../plots/activations_plot.tex', figureheight = '\\figureheight', figurewidth = '\\figurewidth')

print("DONE")

