import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-2,2,0.01)
sigmoid = np.exp(x)/(1 + np.exp(x))
tanh = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
relu = np.maximum(x-0.015, 0)
cut_off = 0.1

a = 0.333 * x[x<0]
b = x[x>=0] + 0.015
lrelu = np.append(a, b)

plt.rc('text', usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})

#plt.rc('font', family='serif')

plt.plot(x,sigmoid, x, tanh, x, relu, x, lrelu, linewidth='2')
plt.xlabel(r'x')
plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
plt.title(r'Activation functions')
plt.legend([r'sigmoid', "hyperbolic tangent", "rectifier", "leaky rectifier ($\\alpha=0.333$)"], loc=2)

plt.savefig('../figs/activations.png')

plt.show()

print("DONE")

