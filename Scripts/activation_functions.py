import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-2,2,0.01)
sigmoid = np.exp(x)/(1 + np.exp(x))
tanh = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
relu = np.maximum(x, 0)
lrelu = np.append(0.333 * x[x<0], x[x>=0])

plt.rc('text', usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})

#plt.rc('font', family='serif')

plt.plot(x,sigmoid, x, tanh, x, relu, x, lrelu)
plt.xlabel(r'x')
plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
plt.title(r'Activation functions')
plt.legend([r'sigmoid', "hyperbolic tangent", "rectifier", "leaky rectifier"], loc=2)

plt.savefig('../figs/activations.pdf')

plt.show()

print("DONE")

