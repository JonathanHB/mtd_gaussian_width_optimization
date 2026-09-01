import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
import time
from scipy.stats import norm


#written by copilot
def fit_cubic_minimum(x, y, plot=False):
    """Fit a polynomial to the lower half of the data and return its minimum within the data range."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 4:
        raise ValueError("At least 4 data points are required to fit a cubic polynomial")

    uselowerhalf = False
    if uselowerhalf:
        sorted_idx = np.argsort(y)
        n_keep = x.size // 2
        if n_keep < 4:
            raise ValueError("At least 4 data points are required in the lower half of y values to fit a cubic polynomial")

        lower_idx = sorted_idx[:n_keep]
        x = x[lower_idx]
        y = y[lower_idx]

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
    
    if len(candidates) == 0:
        candidates.extend([x_min, x_max]) #include endpoints in case the minimum is at the boundary

    values = [cubic(point) for point in candidates]
    best_idx = int(np.argmin(values))

    if plot:
        x_fit = np.linspace(x_min, x_max, 400)
        plt.figure()
        plt.plot(x, y, "o", label="data")
        plt.plot(x_fit, cubic(x_fit), "-", label="cubic fit")
        plt.scatter(candidates[best_idx], values[best_idx], color="red", label="minimum")
        plt.xlabel("gaussian width (sigma)")
        plt.ylabel("RMSE (kT)")
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
    
    def sec_deriv_bottom(self):
        return self.A/(self.B**2)


#system
class sine_well_1:
    def __init__(self, A,B,C):
        self.A = A
        self.B = B
        self.C = C

    def f1(self, x):
        return self.A * x**4 - self.B*np.cos(self.C*x)

    def x_minimum(self):
        return 0
    
    def depth(self):
        return 2*abs(self.B)
    
    def stdev(self):
        return 15

    # def well_volume(self,x):
    #     volume = 0
    #     for Fi in self.f1(x):
    #         if Fi<0:
    #             volume -= Fi
    #     return volume
    
    def sec_deriv_bottom(self):
        #f'' = 12Ax**2 - 2AB
        return self.B*self.C**2

    def xlim(self):
        return -10

#system
class double_well_1:
    def __init__(self, A,B):
        self.A = A
        self.B = B

    def f1(self, x):
        return self.A * (x**4 - self.B*x**2)

    def x_minimum(self):
        return np.sqrt(self.B/2)
    
    def depth(self):
        return self.A*(self.B**2)/4
    
    def stdev(self):
        return np.sqrt(self.B/2)/2

    def well_volume(self,x):
        volume = 0
        for Fi in self.f1(x):
            if Fi<0:
                volume -= Fi
        return volume
    
    def sec_deriv_bottom(self):
        #f'' = 12Ax**2 - 2AB
        return 4*self.A*self.B

    def xlim(self):
        return -np.sqrt(self.B)


@njit
def _weighted_choice(weights):
    n = weights.shape[0]
    cumsum = np.empty(n, dtype=np.float64)
    total = 0.0
    for i in range(n):
        total += weights[i]
        cumsum[i] = total
    if total <= 0.0:
        return 0
    r = np.random.random() * total
    for i in range(n):
        if r < cumsum[i]:
            return i
    return n - 1


@njit
def _simulate_numba_core(F, x, V, T, sigma, omega, deltaT, lim_ind, nrounds, F_norm):
    npts = x.shape[0]
    mse = np.empty(nrounds, dtype=np.float64)
    gaussian = np.empty(npts, dtype=np.float64)
    exp_term = np.empty(npts, dtype=np.float64)

    for t in range(nrounds):
        total = 0.0
        for i in range(npts):
            exp_term[i] = np.exp(-(F[i] + V[i]) / T)
            total += exp_term[i]
        if total == 0.0:
            total = 1.0
        for i in range(npts):
            exp_term[i] /= total

        bin_idx = _weighted_choice(exp_term)

        factor = np.exp(-V[bin_idx] / deltaT)
        gaussian_sum = 0.0
        for i in range(npts):
            dx = x[i] - x[bin_idx]
            gaussian[i] = np.exp(-0.5 * (dx / sigma) ** 2) #can this be factored out and then accessed via array indexing?
            gaussian_sum += gaussian[i]
        if gaussian_sum == 0.0:
            gaussian_sum = 1.0
        for i in range(npts):
            V[i] += omega * factor * gaussian[i] / gaussian_sum

        log_norm = 0.0
        for i in range(lim_ind, npts - lim_ind):
            fV_val = -V[i] * (T + deltaT) / deltaT
            log_norm += np.exp(-fV_val / T)
        if log_norm == 0.0:
            log_norm = 1.0
        log_norm = T * np.log(log_norm)

        n_slice = npts - 2 * lim_ind
        mse_val = 0.0
        for idx in range(n_slice):
            i = lim_ind + idx
            fV_norm = -V[i] * (T + deltaT) / deltaT + log_norm
            diff = F_norm[idx] - fV_norm
            mse_val += diff * diff
        mse[t] = mse_val / n_slice

    return mse, V


def simulate(system, T, sigma, omega, deltaT, x, nrounds, threshold, plot=False):

    plot_mode = "a"
    npts = len(x)

    lim_ind = np.where(x > system.xlim())[0][0]

    F = system.f1(x)
    F_norm = F[lim_ind:-lim_ind] + T * np.log(np.sum(np.exp(-F[lim_ind:-lim_ind] / T)))
    V = np.zeros(npts)

    mse, V = _simulate_numba_core(F, x, V, T, sigma, omega, deltaT, lim_ind, nrounds, F_norm)

    rmse = np.sqrt(mse)

    if plot:
        if plot_mode == "a":
            plt.plot(x, F, color="black", label="G(x)")
            #fV_val = -V * (T + deltaT) / deltaT
            #log_norm += np.exp(-fV_val / T)
            plt.plot(x, F+V*(T+deltaT)/deltaT, color="lime", label="G(x)+V(x)")
            #plt.axvline(x[lim_ind], linestyle="dotted", color="green")
        else:
            plt.plot(x[lim_ind:-lim_ind], F_norm, color="black", linestyle="dashed")

        plt.xlabel("x")
        plt.ylabel("Free energy")
        plt.xlim(x[lim_ind], x[-lim_ind])
        plt.ylim(-5, 15)
        plt.legend()
        plt.title(f"n_gaussians={nrounds}, sigma={sigma}")
        plt.savefig(f"figures/error_vs_sigma_ngaussians={nrounds}_sigma={sigma}.png")
        plt.show()

        plt.plot(rmse)
        plt.ylim(0, max(rmse))
        plt.xlabel("Number of independent gaussians deposited")
        plt.ylabel("RMSE")
        plt.show()

    return rmse, nrounds - 1, nrounds



#do not use this method; it is retained as a reference and is a slower version of copilot-refactored simulate()
def simulate_old(system, T, sigma, omega, deltaT, x, nrounds, threshold, plot=False):

    t01 = time.time()

    plot_mode = "a"

    # #construct MTD gaussian
    #TODO BEWARE THIS MAY FAIL FOR ODD len(x)
    ref_gaussian_ = norm.pdf(x, loc=x[int(len(x)/2)], scale=sigma)
    ref_gaussian_ /= np.sum(ref_gaussian_)
    ref_gaussian = np.zeros(2*len(ref_gaussian_))
    halfx = int(np.ceil(len(ref_gaussian_)/2))
    ref_gaussian[halfx:halfx+len(x)] = ref_gaussian_
    # plt.plot(ref_gaussian)
    # plt.show()
    # plt.plot(ref_gaussian_)
    # plt.show()
    #simulate

    #well_vol = system.well_volume(x)
    #print(f"well volume = {well_vol}")

    #plt.ylim(-30,50)

    lim_ind = np.where(x>system.xlim())[0][0]
    # print(np.where(x>system.xlim()))
    # print(lim_ind)
    
    F = system.f1(x)
    F_norm = F[lim_ind:-lim_ind]+T*np.log(np.sum(np.exp(-F[lim_ind:-lim_ind]/T)))
    V = np.zeros(len(F))

    mse = []

    t03 = time.time()
    #print(f"pre-simulation: {t03-t01}")

    #nrounds = 20*round(well_vol/omega)
    #nrounds = 99999
    #conv = False
    #print(f"{nrounds} rounds")
    for t in range(nrounds):
        t1 = time.time()
        #for an even number of frames the convolution is shifted off by half a bin. 
        # For an odd number of frames it should already be symmetrical so the symmetrization should do nothing
        #convolution_halfshifted = np.convolve(gaussian, np.exp(-V/deltaT)*np.exp(-(F+V)/T)/np.sum(np.exp(-(F+V)/T)),mode="same")
        #convolution_symmetrized = (convolution_halfshifted+convolution_halfshifted[::-1])/2
        tp0 = time.time()
        probability_density = np.exp(-(F+V)/T)/np.sum(np.exp(-(F+V)/T))
        tp1 = time.time()
        #print(f"prob: {tp1-tp0}")

        tr0 = time.time()
        bin = np.random.choice(range(len(x)), p=probability_density) #probabilities must sum to 1
        tr1 = time.time()
        #print(f"random: {tr1-tr0}")

        tg0 = time.time()
        #print(bin)
        #print(x[bin])

        gaussian = ref_gaussian[-bin-len(x):-bin]
        # plt.plot(gaussian)
        # g2 = norm.pdf(x, loc=x[bin], scale=sigma)
        # g2/= np.sum(g2)
        # plt.plot(g2)
        # plt.show()
        # sys.exit(0)
        tg1 = time.time()
        #print(f"gaussian: {tg1-tg0}")

        #print(np.sum(gaussian)) #not 1
        #gaussian /= np.sum(gaussian)
        V += omega*np.exp(-V[bin]/deltaT)*gaussian


        te0 = time.time()

        fV = -V*(T+deltaT)/deltaT
        fV_norm = fV[lim_ind:-lim_ind]+T*np.log(np.sum(np.exp(-fV[lim_ind:-lim_ind]/T)))

        mse.append(np.mean((F_norm - fV_norm)**2))
        te1 = time.time()
        #print(f"error: {te1-te0}")

        t2 = time.time()
        #print("simulation round: ", t2-t1)
        
        if plot:
            if t%int(nrounds/100)==0: #t==nrounds-1:#
                print(f"round {t}")
                if plot_mode=="a":
                    plt.plot(x, F+V*(T+deltaT)/deltaT, color=(t/nrounds,0.8,0))
                else:
                    plt.plot(x[lim_ind:-lim_ind], fV_norm, color=(t/nrounds,0.8,0))

    t04 = time.time()
    #print(f"simulation: {t04-t03}")

    # if not conv:
    #     print(f"did not converge after {nrounds} rounds")

    rmse = np.sqrt(mse)

    if plot:
        if plot_mode=="a":
            plt.plot(x,F, color="black")
            plt.axvline(x[lim_ind], linestyle="dotted", color="green")
        else:
            plt.plot(x[lim_ind:-lim_ind], F_norm, color="black", linestyle="dashed")

        plt.xlabel("x")
        plt.ylabel("Free energy")
        plt.ylim(-10,15)
        plt.show()

        plt.plot(rmse)
        plt.ylim(0,max(rmse))
        plt.xlabel("Number of independent gaussians deposited")
        plt.ylabel("RMSE")
        plt.show()

    t02 = time.time()
    #print(f"total = {t02-t01}")
    
    #sys.exit(0)
    
    return rmse, t, nrounds