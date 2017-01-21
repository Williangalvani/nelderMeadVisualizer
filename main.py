from nelderMead import NelderMead
from plotter import PlotFunctionAndTriangle


def function(x, y):
    return (x-2) **2 + (y-2) ** 2

dataPointsList = NelderMead(function, 2).solve()
PlotFunctionAndTriangle(function, dataPointsList)
