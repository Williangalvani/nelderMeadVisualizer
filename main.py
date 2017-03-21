from neldermead import NelderMead
from plotter import PlotFunctionAndTriangle, plotF


def function(x, y):
    """
    Parabola centered at (2, 2)
    :param x: x values
    :param y: y values
    :return: z values
    """
    return (x-2) ** 2 + (y-2) ** 2

vertices, evals = NelderMead(function, 2).solve()
PlotFunctionAndTriangle(function, vertices)
plotF(evals)
