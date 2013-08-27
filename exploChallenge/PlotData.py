__author__ = 'ftruzzi'

import sys
from pylab import *

def plotData(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            if "click" not in line:
                data.append(line.strip().split()[3])
    plot(data)
    ylabel("CTR")
    xlabel("lines x 10.000")
    ylim(0.01, 0.1)
    grid(True)
    show()


if __name__ == '__main__':
    plotData(sys.argv[1])


