import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

x = np.arange(1,21,1)

dices = np.array(
        [[0.6917, 0.6776,0.6113], #1
         [0.7285, 0.6539, 0.5439],
         [0.7345, 0.6179, 0.4915],
         [0.7441, 0.5955, 0.4880],
         [0.7638, 0.6021, 0.4772], #5
         [0.7661, 0.6063, 0.4877],
         [0.7739, 0.6110, 0.4713],
         [0.7869, 0.6155, 0.4829],
         [0,0,0],
         [0.7885, 0.6318, 0.5151],#10
         [0,0,0],
         [0,0,0],
         [0,0,0],
         [0,0,0],
         [0.7922, 0.6235, 0.4784], #15
         [0.7877, 0.5985, 0.4615], 
         [0.7932, 0.6274, 0.4877],
         [0.7905, 0.6166, 0.4910],
         [0.7935, 0.6266, 0.4980],
         [0.7922, 0.6228, 0.4915]]) #20
       
plt.plot(x, dices[:,0], '', dices[:,1], '', dices[:,2], '')
plt.xlabel('Epoch')
plt.ylabel('Dice score',fontsize=16)
plt.legend(['Complete region', 'Core', 'Enhancing'], loc=2)


tikz_save('../plots/dice_training.tex', figureheight = '\\figureheight', figurewidth = '\\figurewidth')
