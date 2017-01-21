import numpy as np


class CachedFunction:
    def __init__(self, function):
        self.real_function = function
        self.cache = {}

    def eval(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            z = self.real_function(*args)
            self.cache[args] = z
            return z


class NelderMead:
    """
    Nelder-Mead Derivative Free optimization method implementation
    """

    def __init__(self, function, dimensions, alpha=1.0, gama=2.0, rho=0.5, sigma=0.5):
        """
        :param function: Function to find the minimum
        :param dimensions: Input dimensions of "function"
        :param alpha: Reflection coefficient
        :param gama: Expansion coefficient
        :param rho: Contraction coefficient
        :param sigma: Shrink Coefficient
        """
        self.history = []
        self.vertices = np.random.rand(dimensions + 1, dimensions)*5
        self.function = CachedFunction(function).eval
        self.alpha = alpha
        self.gama = gama
        self.rho = rho
        self.sigma = sigma
        self.reflected = None
        self.expanded = None
        self.contracted = None

    def get_centroid(self):
        """
        :return: The centroid of the n-1 first vertices
        """
        arr = self.vertices[:-1]
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        centroid = np.array([sum_x / length, sum_y / length])
        return centroid

    def order_vertices(self):
        """
        Orders self.vertices from the better function value to the worst
        """
        y = [self.function(*x) for x in self.vertices]
        order = np.argsort(y)
        vertices = np.array([self.vertices[order[i]] for i in range(len(self.vertices))])
        self.vertices = vertices

    def get_reflected_point(self):
        """
        :return: last vertex reflected on the centroid position
        """
        self.reflected = self.vertices[-1] + self.alpha*2*(self.get_centroid() - self.vertices[-1])
        return self.reflected

    def reflect(self):
        """
        Replaces the last vertex with the reflected vertex.
        """
        self.vertices[-1, :] = self.reflected

    def reflected_is_best(self):
        """
        :return: BOOL, returns if the reflected vertex is better than all the current ones.
        """
        return self.function(*self.reflected) < self.function(*self.vertices[0])

    def get_expanded_point(self):
        """
        :return: Returns the previously reflected vertex expanded around the centroid.
        """
        centroid = self.get_centroid()
        self.expanded = centroid + self.gama*(self.reflected - centroid)
        return self.expanded

    def expand(self):
        """
        Replaces the last vertex with the expanded vertex
        """
        self.vertices[-1, :] = self.expanded

    def get_contracted_point(self):
        """
        :return: The last vertex contracted around the centroid.
        """
        centroid = self.get_centroid()
        self.contracted = centroid + self.rho*(self.vertices[-1] - centroid)
        return self.contracted

    def contract(self):
        """
        Replaces the last vertex with the contracted vertex
        """
        self.vertices[-1, :] = self.contracted

    def shrink(self):
        """
        Shrinks all vertices around the one with the best value
        """
        for i in range(1, len(self.vertices)):
            self.vertices[i] = self.vertices[0] + self.sigma*(self.vertices[i]-self.vertices[0])

    def converged(self):
        """
        check for convergence
        """
        converged = np.sqrt(np.sum((self.vertices[0]-self.vertices[1])**2)) < 0.0001
        return converged

    def solve(self):
        """
        High-level steps of the Nelder-Mead method
        :return: History of vertex positions.
        """
        f = self.function
        while not self.converged():
            self.history.append(self.vertices)

            #step 1: sort
            self.order_vertices()

            #step 3: reflect
            reflected = self.get_reflected_point()
            if f(*self.vertices[0]) < f(*reflected) < f(*self.vertices[-1]):
                self.reflect()
                continue

            #step 4: expand
            if self.reflected_is_best():
                expanded = self.get_expanded_point()
                if f(*expanded) < f(*reflected):
                    self.expand()
                else:
                    self.reflect()
                continue

            #step 5: contract
            contracted = self.get_contracted_point()
            if f(*contracted) < f(*self.vertices[-1]):
                self.contract()
                continue

            #step 6: shrink
            self.shrink()

        return self.history
