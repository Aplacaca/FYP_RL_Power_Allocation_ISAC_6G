import numpy as np

def f_Theta(a1,a2,a3,c):
    Theta = a3 - a1*1/(np.power(c, a2))
    return Theta