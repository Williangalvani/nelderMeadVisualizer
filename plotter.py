import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.animation as animation


class PlotFunctionAndTriangle:

    def __init__(self, function, datapoints):

        self.dataPoints = datapoints
        dx, dy = 0.05, 0.05

        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[slice(0, 5 + dy, dy),
                        slice(0, 5 + dx, dx)]

        z = function(x, y)

        # x and y are bounds, so z should be the value *inside* those bounds.
        # Therefore, remove the last value from the z array.
        z = z[:-1, :-1]
        levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

        # pick the desired colormap, sensible levels, and define a normalization
        # instance which takes data values and translates those into levels.
        cmap = plt.get_cmap('PiYG')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        self.fig, ax1 = plt.subplots(nrows=1)

        # contours are *point* based plots, so convert our bound into point
        # centers
        cf = ax1.contourf(x[:-1, :-1] + dx / 2.,
                          y[:-1, :-1] + dy / 2., z, levels=levels,
                          cmap=cmap)
        self.fig.colorbar(cf, ax=ax1)
        ax1.set_title('Nelder-Mead on a parabolic function')

        initial = [[0, 1, 2], [0, 1, 2]]
        self.line, = ax1.plot(*initial)

        # adjust spacing between subplots so `ax1` title and `ax0` tick labels
        # don't overlap
        self.fig.tight_layout()

        ani = animation.FuncAnimation(self.fig, self.animate, len(self.dataPoints),
                                      interval=300, blit=True)
        ani.save('animation.gif', writer='imagemagick', fps=3)
        plt.show()

    def animate(self, i):
        points = np.append(self.dataPoints[i], [self.dataPoints[i][0]], axis=0)
        self.line.set_data(points.transpose())  # update the data
        return self.line,