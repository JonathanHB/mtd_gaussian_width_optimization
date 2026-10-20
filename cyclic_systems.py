import numpy as np
import matplotlib.pyplot as plt

#these should in principle have a superclass since their methods are the same except for 

class F_sine:
    def __init__(self, npts, A):
        self.npts = npts
        self.x_min = 0
        self.x_max = 2*np.pi
        self.x = np.linspace(self.x_min, self.x_max, self.npts, endpoint=False)
        self.F = A*np.sin(self.x)  # Example function; replace with actual function as needed

    def evaluate(self):
        return self.x, self.F

    def plot(self):
        x, F = self.evaluate()
        plt.plot(x, F)
        plt.title("sin(x)")
        plt.xlabel("x (radians)")
        plt.ylabel("F (energy)")
        plt.xlim(self.x_min, self.x_max)
        plt.show()


class F_2sine:
    def __init__(self, npts, A):
        self.npts = npts
        self.x_min = 0
        self.x_max = 2*np.pi
        self.x = np.linspace(self.x_min, self.x_max, self.npts, endpoint=False)
        self.F = A*np.sin(2*self.x)  # Example function; replace with actual function as needed

    def evaluate(self):
        return self.x, self.F

    def plot(self):
        x, F = self.evaluate()
        plt.plot(x, F)
        plt.title("sin(2x)")
        plt.xlabel("x (radians)")
        plt.ylabel("F (energy)")
        plt.xlim(self.x_min, self.x_max)
        plt.show()

