import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


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


def simulate(system, T, sigma, omega, deltaT, x, threshold, plot=False):

    plot_mode = "a"

    #construct MTD gaussian
    gaussian = norm.pdf(x, loc=0, scale=sigma)
    gaussian /= np.sum(gaussian)

    #simulate

    well_vol = system.well_volume(x)
    print(f"well volume = {well_vol}")

    #plt.ylim(-30,50)

    lim_ind = np.where(x>-5*system.stdev())[0][0]

    F = system.f1(x)
    V = np.zeros(len(F))

    rmse = []

    #nrounds = 50*round(well_vol/omega)
    nrounds = 99999
    conv = False
    #print(f"{nrounds} rounds")
    for t in range(nrounds):
        #for an even number of frames the convolution is shifted off by half a bin. 
        # For an odd number of frames it should already be symmetrical so the symmetrization should do nothing
        convolution_halfshifted = np.convolve(gaussian, np.exp(-V/deltaT)*np.exp(-(F+V)/T)/np.sum(np.exp(-(F+V)/T)),mode="same")
        convolution_symmetrized = (convolution_halfshifted+convolution_halfshifted[::-1])/2
        V += omega*convolution_symmetrized
        if t%100==0:
            F_norm = F[lim_ind:-lim_ind]+T*np.log(np.sum(np.exp(-F[lim_ind:-lim_ind]/T)))
            fV = -V*(T+deltaT)/deltaT
            fV_norm = fV[lim_ind:-lim_ind]+T*np.log(np.sum(np.exp(-fV[lim_ind:-lim_ind]/T)))
            if plot:
                if plot_mode=="a":
                    plt.plot(x, F+V*(T+deltaT)/deltaT, color=(t/nrounds,0.8,0))
                else:
                    plt.plot(x[lim_ind:-lim_ind], fV_norm, color=(t/nrounds,0.8,0))
            rmse.append(np.sqrt(np.mean((F_norm - fV_norm)**2)))
            if rmse[-1] < threshold:
                print(f"converged after {t} rounds")
                conv=True
                break

    if not conv:
        print(f"did not converge after {nrounds} rounds")

    if plot:
        if plot_mode=="a":
            plt.plot(x,F, color="black")
            plt.axvline(x[lim_ind], linestyle="dotted", color="green")
        else:
            plt.plot(x[lim_ind:-lim_ind], F_norm, color="black", linestyle="dashed")
        plt.show()

        plt.plot(rmse)
        plt.ylim(0,0.3)
        plt.show()

    return rmse, t