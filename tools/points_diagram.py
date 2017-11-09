#!/usr/bin/python3

import math

# Импортируем один из пакетов Matplotlib
import pylab

# Импортируем пакет со вспомогательными функциями
from matplotlib import mlab

from matplotlib import pyplot as mpl
import matplotlib
#matplotlib.use('Agg')

def show_graph(xlist=None, ylist=None):
    pylab.ion()

    if xlist is None:
        xlist = mlab.frange (0, 1, 0.1)
    # Данные для очередного кадра
    if ylist is None:
        ylist = [math.sin (x) for x in xlist]

    # !!! Очистим график
    pylab.clf()

    # Выведем новые данные
    pylab.plot (xlist, ylist, ':')

    # !!! Нарисуем их
    # !!! Обратите внимание, что здесь используется функция draw(), а не show()
    pylab.draw()
    mpl.pause(0.001)

