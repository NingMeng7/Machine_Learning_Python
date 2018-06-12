import numpy as np
from numpy import *
import scipy.optimize as opt
import matplotlib.pyplot as plt
from module import GetData
from module import grad_ascent
from module import plotBoundary


dataArr, labelMat = GetData()
weights = grad_ascent(dataArr, labelMat)
print(weights)
plotBoundary(array(weights))
