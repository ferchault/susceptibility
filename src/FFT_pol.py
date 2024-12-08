# %%
import pyscf.scf
import pyscf.gto
import pyscf.dft
import pyscf.qmmm
import numpy as np
import itertools as it
from scipy.sparse.linalg import lsqr
from os.path import exists as file_exists
import time

from scipy.fft import fft, ifft, fftn, ifftn, fftfreq, fftshift, ifftshift

def regularized_least_squares(A, y, lamb=0):
    n_col = A.shape[1]
    return np.linalg.lstsq(
        A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y), rcond=None
    )

def CutOutFar(gridcoords, atomobject, radius): #review
    notFarCoords = []
    for gridpoint in gridcoords:
        largestDist = 0
        for atompoint in atomobject:
            if np.linalg.norm(gridpoint - atompoint[1]) < largestDist:
                largestDist = np.linalg.norm(gridpoint - atompoint[1])
        if largestDist < radius:
            notFarCoords.append(gridpoint)
    return np.asarray(notFarCoords)

class ResponseCalculator:
    """Implements a generalized case of 10.1021/ct1004577, section 2."""

    def __init__(self, mol, calc):
        self._mol = mol
        self._calc = calc
        self._q = 0.001
        self._grid = pyscf.dft.gen_grid.Grids(self._mol)
        self._grid.level = 8
        self._grid.build()
        calc = self._calc(self._mol)
        calc.kernel()
        self._center = calc.energy_elec()[0]

    def get_energy_derivative(self, pos):
        up = pyscf.qmmm.mm_charge(
            self._calc(self._mol), np.array((pos,)), np.array((self._q,))
        )
        up.kernel()
        dn = pyscf.qmmm.mm_charge(
            self._calc(self._mol), np.array((pos,)), np.array((-self._q,))
        )
        dn.kernel()
        assert up.converged and dn.converged
        return up.energy_elec()[0] + dn.energy_elec()[0] - 2 * self._center

    def get_derivative(self, pos: np.ndarray):
        d = np.linalg.norm(self._grid.coords - pos, axis=1)
        combined = self._q * self._grid.weights / d
        ao_value = pyscf.dft.numint.eval_ao(self._molresp, self._grid.coords, deriv=0)
        integrals = np.dot(ao_value.T, combined.T)

        B_j = np.outer(integrals, integrals).reshape(-1)
        D_j = self.get_energy_derivative(pos)

        return D_j, B_j

    def build_susceptibility(self, coords: np.ndarray, fit_mol):
        self._molresp = fit_mol
        if file_exists("chi.npy"):
            self._Qvec = np.load("chi.npy")
        else:
            D = []
            B = []

            if len(coords) < self._molresp.nao:
                # print("Would be underdetermined. Aborting.")
                raise ValueError("Underdetermined")
            #for coord in tqdm.tqdm(coords, desc="Chi", leave=False):
            for coord in coords:
                # for coord in coords:
                D_j, B_j = self.get_derivative(coord)
                D.append(D_j)
                B.append(B_j)
            D = np.array(D)
            B = np.array(B)

            lstsq = lsqr(B, D)
            res = (np.sqrt((D - B @ lstsq[0]) ** 2).mean()) / np.abs(D).mean()
            print(f"Chi:   Average relative residual {res*100:8.3f} %")
            self._Q = lstsq[0].reshape(self._molresp.nao, self._molresp.nao)
            self._Qvec = lstsq[0]
            np.save("chi.npy", self._Qvec)
            np.save("B.npy", B)
            np.save("D.npy", D)
            return res

    def evaluate_susceptibility(self, r: np.ndarray, rprime: np.ndarray) -> float:
        coords = np.array((r, rprime))
        beta_k, beta_l = pyscf.dft.numint.eval_ao(self._molresp, coords, deriv=0)

        # return np.sum(self._Q * np.outer(beta_k, beta_l))
        return np.dot(self._Qvec, np.outer(beta_k, beta_l).reshape(-1))

    def get_ao_integrals(self) -> float:
        basis_set_values = pyscf.dft.numint.eval_ao(
            self._molresp, self._grid.coords, deriv=0
        )
        ao_integrals = np.dot(self._grid.weights, basis_set_values)
        return ao_integrals


def real_space_scan():
    mol = pyscf.gto.M(
        atom=f"He 0 0 0",
        basis="def2-TZVP",
        verbose=0,
    )
    chi_mol = pyscf.gto.M(
        atom=f"He 0 0 0",
        basis="unc-def2-TZVPP",
        verbose=0,
    )

    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.level = 0
    grid.build()

    rc = ResponseCalculator(mol, pyscf.scf.RHF)

    rc.build_susceptibility(grid.coords, chi_mol)
    return rc

def FFT_nlpsi(rc, npts, xmin):
    if file_exists("psi.npy"):
        #alpha = np.load("alpha.npy")
        psi = np.load("psi.npy")
        xord, yord, zord =  np.load("xord.npy"), np.load("yord.npy"), np.load("zord.npy")
        xpord, ypord, zpord =  np.load("xpord.npy"), np.load("ypord.npy"), np.load("zpord.npy")
        psifft = np.load("psi_FFT.npy")
        #alphafft = np.load("alpha_FFT.npy")
        susfft = np.load("chi_FFT.npy")
    else:

        ##############################################################################
        # Setting up Fourier grid to obtain /chi(r, -r)
        pts, ptmin, ptmax = npts, xmin, -1.0 * xmin
        squaregrid = np.linspace(ptmin, ptmax, pts) #odd points so zero is in there

        x, y, z, xp, yp, zp = np.meshgrid(squaregrid, squaregrid, squaregrid, squaregrid, squaregrid, squaregrid, indexing='ij')
        # reordering so FFT can work
        xord, yord, zord =  ifftshift(x), ifftshift(y), ifftshift(z)
        xpord, ypord, zpord =  ifftshift(xp), ifftshift(yp), ifftshift(zp)

        sus = np.zeros(zord.shape)

        # filling in susceptibility. TODO: utilize symmetry
        for (i,j,k,l,m,n), value in np.ndenumerate(xord):
            suscept = rc.evaluate_susceptibility( (xord[i,j,k,l,m,n], yord[i,j,k,l,m,n], zord[i,j,k,l,m,n]), (-xpord[i,j,k,l,m,n], -ypord[i,j,k,l,m,n], -zpord[i,j,k,l,m,n]) )
            sus[i,j,k,l,m,n] = -suscept # negative sign to match the definition

        susfft = fft(sus)

        # k-space grid
        x_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)
        y_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)
        z_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)

        xp_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)
        yp_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)
        zp_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)

        #alphafft = np.zeros(susfft.shape)
        psifft = np.zeros(susfft.shape)

        # polarization potential
        for (i,j,k,l,m,n), value in np.ndenumerate(susfft):
            ksize = np.linalg.norm( (x_freq_grid[i], y_freq_grid[j], z_freq_grid[k]) )
            kpsize = np.linalg.norm( (xp_freq_grid[l], yp_freq_grid[m], zp_freq_grid[n]) )
            psifft[i,j,k,l,m,n] = susfft[i,j,k,l,m,n]/(ksize**2 * kpsize**2)

        psi = ifft(psifft) #result

        # filling in alpha. TODO: make sure 0 is not an issue?
        #for (i,j,k,l,m,n), value in np.ndenumerate(susfft):
        #    ksize = np.linalg.norm( (x_freq_grid[i], y_freq_grid[j], z_freq_grid[k]) )
        #    kpsize = np.linalg.norm( (xp_freq_grid[l], yp_freq_grid[m], zp_freq_grid[n]) )
        #    alphafft[i,j,k,l,m,n] = susfft[i,j,k,l,m,n]/(ksize * kpsize)
        #    if i==0 and j==0 and k==0 and l==0 and m==0 and l==0:
        #        print('allzero')
        #        print(alphafft[i,j,k,l,m,n])

        #alpha = ifft(alphafft) #result

        #np.save("alpha_FFT.npy", alphafft)
        np.save("psi_FFT.npy", psifft)
        np.save("chi_FFT.npy", susfft)
        np.save("psi.npy", psi)
        #np.save("alpha.npy", alpha)

        np.save("xord.npy", xord)
        np.save("yord.npy", yord)
        np.save("zord.npy", zord)

        np.save("xpord.npy", xpord)
        np.save("ypord.npy", ypord)
        np.save("zpord.npy", zpord)

    return xord, yord, zord, xpord, ypord, zpord, psi

def FFT_nlpol(rc_object, npts, xmin, psi):
    if file_exists("alphaxx.npy"):
        alphaxx, alphayy, alphazz =  np.load("alphaxx.npy"), np.load("alphayy.npy"), np.load("alphazz.npy")
        alphaxy, alphaxz, alphayz =  np.load("alphaxy.npy"), np.load("alphaxz.npy"), np.load("alphayz.npy")
        #psifft = np.load("psi_FFT.npy")
        #alphafft = np.load("alpha_FFT.npy")
        #susfft = np.load("chi_FFT.npy")
        print('alphas found, ezpz')
    else:
        pts, ptmin, ptmax = npts, xmin, -1.0 * xmin
        # k-space grid
        x_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)
        y_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)
        z_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)

        xp_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)
        yp_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)
        zp_freq_grid = fftfreq(npts, d=(ptmax-ptmin)/npts)

        psifft = fft(psi) #result

        alphafft_xx, alphafft_yy, alphafft_zz = np.zeros(psifft.shape), np.zeros(psifft.shape), np.zeros(psifft.shape)
        alphafft_xy, alphafft_xz, alphafft_yz = np.zeros(psifft.shape), np.zeros(psifft.shape), np.zeros(psifft.shape)

        print('time to iterate over i j k l m n')

        # filling in alpha. TODO: make sure 0 is not an issue?
        for (i,j,k,l,m,n), value in np.ndenumerate(psifft):
            ksize = np.linalg.norm( (x_freq_grid[i], y_freq_grid[j], z_freq_grid[k]) )
            kpsize = np.linalg.norm( (xp_freq_grid[l], yp_freq_grid[m], zp_freq_grid[n]) )
            alphafft_xx[i,j,k,l,m,n] = psifft[i,j,k,l,m,n] * x_freq_grid[i] * xp_freq_grid[l]
            alphafft_yy[i,j,k,l,m,n] = psifft[i,j,k,l,m,n]* y_freq_grid[j] * yp_freq_grid[m]
            alphafft_zz[i,j,k,l,m,n] = psifft[i,j,k,l,m,n]* z_freq_grid[k] * zp_freq_grid[n]

            alphafft_xy[i,j,k,l,m,n] = psifft[i,j,k,l,m,n]* x_freq_grid[i] * yp_freq_grid[m]
            alphafft_xz[i,j,k,l,m,n] = psifft[i,j,k,l,m,n]* x_freq_grid[i] * zp_freq_grid[n]
            alphafft_yz[i,j,k,l,m,n] = psifft[i,j,k,l,m,n]* y_freq_grid[j] * zp_freq_grid[n]

            if i==0 and j==1 and k==1 and l==0 and m==1 and n==1:
                print('ksize: ', ksize)
                print('x_freq_grid[0]: ', x_freq_grid[i])
                print('xx integral: ',alphafft_xx[i,j,k,l,m,n])

        alphaxx = ifft(alphafft_xx)
        alphayy = ifft(alphafft_yy)
        alphazz = ifft(alphafft_zz)

        alphaxy = ifft(alphafft_xy)
        alphaxz = ifft(alphafft_xz)
        alphayz = ifft(alphafft_yz)

        np.save("alphaxx.npy", alphaxx)
        np.save("alphayy.npy", alphayy)
        np.save("alphazz.npy", alphazz)

        np.save("alphaxy.npy", alphaxy)
        np.save("alphaxz.npy", alphaxz)
        np.save("alphayz.npy", alphayz)

    return alphaxx, alphayy, alphazz, alphaxy, alphaxz, alphayz

def 1d_integral(alpha_dd, pts, minx):
    trapezoidal = 'todo'
    return trapezoidal


##############################################################################
rc_object = real_space_scan() # Sets up the response calculator

pts, minx=17, -2.5
xord, yord, zord, xpord, ypord, zpord, psi = FFT_nlpsi(rc_object, npts=pts, xmin=minx)

print('psi done')

alphaxx, alphayy, alphazz, alphaxy, alphaxz, alphayz = FFT_nlpol(rc_object, pts, minx, psi)
