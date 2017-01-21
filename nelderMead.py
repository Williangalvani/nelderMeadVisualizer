from random import random
import numpy as np

class NelderMead:

    def __init__(self, function, dimensions, alpha=1.0, gama=2.0, rho=0.5, sigma=0.5):
        self.history = []
        self.vertices = np.random.rand(dimensions + 1, dimensions)*5
        self.function = function
        self.alpha = alpha
        self.gama = gama
        self.rho = rho
        self.sigma = sigma
        self.reflected = None
        self.expanded = None
        self.contracted = None

    def get_centroid(self):
        arr = self.vertices[:-1]
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        centroid = np.array([sum_x / length, sum_y / length])
        return centroid

    def order_vertices(self):
        y = [self.function(*x) for x in self.vertices]
        order = np.argsort(y)
        vertices = np.array([self.vertices[order[i]] for i in range(len(self.vertices))])
        self.vertices = vertices

    def get_reflected_point(self):
        self.reflected = self.vertices[-1] + self.alpha*2*(self.get_centroid() - self.vertices[-1])
        return self.reflected

    def reflect(self):
        self.vertices[-1.0, :] = self.reflected

    def reflected_is_best(self):
        return self.function(*self.reflected) < self.function(*self.vertices[0])

    def get_expanded_point(self):
        centroid = self.get_centroid()
        self.expanded = centroid + self.gama*(self.reflected - centroid)
        return self.expanded

    def expand(self):
        self.vertices[-1, :] = self.expanded

    def get_contracted_point(self):
        centroid = self.get_centroid()
        self.contracted = centroid + self.rho*(self.vertices[-1] - centroid)
        return self.contracted

    def contract(self):
        self.vertices[-1, :] = self.contracted

    def shrink(self):
        for i in range(1, len(self.vertices)):
            self.vertices[i] = self.vertices[0] + self.sigma*(self.vertices[i]-self.vertices[0])

    def converged(self):
        converged = np.sqrt(np.sum((self.vertices[0]-self.vertices[1])**2)) < 0.0001
        return converged

    def solve(self):
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