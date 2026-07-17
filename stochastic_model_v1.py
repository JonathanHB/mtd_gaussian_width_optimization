import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


#written by copilot
def fit_cubic_minimum(x, y, plot=False):
    """Fit a cubic polynomial to data and return its minimum within the data range."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 4:
        raise ValueError("At least 4 data points are required to fit a cubic polynomial")

    coeffs = np.polyfit(x, y, 3)
    cubic = np.poly1d(coeffs)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    critical_points = np.roots(np.polyder(cubic))
    candidates = []
    for point in critical_points:
        if np.isreal(point) and np.isfinite(point):
            point = float(np.real(point))
            if x_min <= point <= x_max:
                candidates.append(point)

    candidates.extend([x_min, x_max])

    values = [cubic(point) for point in candidates]
    best_idx = int(np.argmin(values))

    if plot:
        x_fit = np.linspace(x_min, x_max, 400)
        plt.figure()
        plt.plot(x, y, "o", label="data")
        plt.plot(x_fit, cubic(x_fit), "-", label="cubic fit")
        plt.scatter(candidates[best_idx], values[best_idx], color="red", label="minimum")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    return float(candidates[best_idx]), float(values[best_idx])


#system
class gaussian_well_1:
    def __init__(self, A,B):
        self.A = A
        self.B = B

    def f1(self, x):
        return -self.A*(1/np.sqrt(2*np.pi*self.B**2))*np.exp(-0.5 * (x / self.B)**2)

    def stdev(self):
        return self.B

    def well_volume(self,x):
        volume = 0
        for Fi in self.f1(x):
            if Fi<0:
                volume -= Fi
        return volume

    def xlim(self):
        return -5*self.stdev()


#system
class double_well_1:
    def __init__(self, A,B):
        self.A = A
        self.B = B

    def f1(self, x):
        return self.A * (x**4 - self.B*x**2)

    def stdev(self):
        return np.sqrt(self.B/2)/2

    def well_volume(self,x):
        volume = 0
        for Fi in self.f1(x):
            if Fi<0:
                volume -= Fi
        return volume
    
    def xlim(self):
        return -3*self.stdev()


def simulate(system, T, sigma, omega, deltaT, x, nrounds, threshold, plot=False):

    plot_mode = "a"

    # #construct MTD gaussian
    # gaussian = norm.pdf(x, loc=0, scale=sigma)
    # gaussian /= np.sum(gaussian)

    #simulate

    well_vol = system.well_volume(x)
    #print(f"well volume = {well_vol}")

    #plt.ylim(-30,50)

    lim_ind = np.where(x>system.xlim())[0][0]

    F = system.f1(x)
    V = np.zeros(len(F))

    rmse = []

    #nrounds = 20*round(well_vol/omega)
    #nrounds = 99999
    conv = False
    #print(f"{nrounds} rounds")
    for t in range(nrounds):
        #for an even number of frames the convolution is shifted off by half a bin. 
        # For an odd number of frames it should already be symmetrical so the symmetrization should do nothing
        #convolution_halfshifted = np.convolve(gaussian, np.exp(-V/deltaT)*np.exp(-(F+V)/T)/np.sum(np.exp(-(F+V)/T)),mode="same")
        #convolution_symmetrized = (convolution_halfshifted+convolution_halfshifted[::-1])/2
        
        probability_density = np.exp(-(F+V)/T)/np.sum(np.exp(-(F+V)/T))
        bin = np.random.choice(range(len(x)), p=probability_density)
        #print(bin)
        gaussian = norm.pdf(x, loc=x[bin], scale=sigma)
        gaussian /= np.sum(gaussian)
        V += omega*np.exp(-V[bin]/deltaT)*gaussian
        
        F_norm = F[lim_ind:-lim_ind]+T*np.log(np.sum(np.exp(-F[lim_ind:-lim_ind]/T)))
        fV = -V*(T+deltaT)/deltaT
        fV_norm = fV[lim_ind:-lim_ind]+T*np.log(np.sum(np.exp(-fV[lim_ind:-lim_ind]/T)))

        rmse.append(np.sqrt(np.mean((F_norm - fV_norm)**2)))

        if t%int(nrounds/100)==0:
            if plot:
                if plot_mode=="a":
                    plt.plot(x, F+V*(T+deltaT)/deltaT, color=(t/nrounds,0.8,0))
                else:
                    plt.plot(x[lim_ind:-lim_ind], fV_norm, color=(t/nrounds,0.8,0))

    # if not conv:
    #     print(f"did not converge after {nrounds} rounds")

    if plot:
        if plot_mode=="a":
            plt.plot(x,F, color="black")
            plt.axvline(x[lim_ind], linestyle="dotted", color="green")
        else:
            plt.plot(x[lim_ind:-lim_ind], F_norm, color="black", linestyle="dashed")
        plt.xlabel("x")
        plt.ylabel("Free energy")
        plt.show()

        plt.plot(rmse)
        plt.ylim(0,max(rmse))
        plt.xlabel("Number of independent gaussians deposited")
        plt.ylabel("RMSE")
        plt.show()

    return rmse, t, nrounds