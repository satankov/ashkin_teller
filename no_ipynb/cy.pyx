#cython: language_level=3
# --compile-args="-O2"


cimport cython
import numpy as np

cimport numpy as cnp
from numpy cimport npy_intp, npy_longdouble
from libcpp cimport bool
from libc.math cimport exp


@cython.boundscheck(False)
cdef int calc_dE(npy_intp site,
                cnp.int32_t[::1] spins_1,  
                cnp.int32_t[::1] spins_2,
                cnp.int32_t[:, :] neighbors):    
    """Calculate e_jk: the energy change for spins[site] -> *= -1"""
    
    cdef:
        int site_1 = spins_1[site]
        int site_2 = spins_2[site]
        int dE = 0
        npy_intp idx
           
    
    for j in range(4):   
        idx = neighbors[site, j]
        dE += 2*site_1*(site_2*spins_1[idx]*spins_2[idx]+spins_1[idx])
    
    return dE


@cython.cdivision(True)
cdef bint mc_choice(int dE, double T, npy_longdouble uni):
    """принимаем или не принимаем переворот спина?"""
    cdef double r
    r = exp(-dE/T)
    if dE <= 0:
        return True
    elif uni <= r:
        return True
    else:
        return False

    

@cython.boundscheck(False) 
cdef void step(cnp.int32_t[::1] spins_s, cnp.int32_t[::1] spins_t,
               cnp.int32_t[:, :] neigh, double T, npy_intp lattice, npy_intp site, npy_longdouble uni):
    """крутим 1 спин"""
        
    cdef int L2, dE
    
    L2 = spins_s.shape[0]
    
    if lattice == 1:
        dE = calc_dE(site, spins_s, spins_t, neigh)   # if lattice s -> pass s,t
        if mc_choice(dE, T, uni):
            spins_s[site] *= -1
    else:
        dE = calc_dE(site, spins_t, spins_s, neigh)   # if lattice t -> pass t,s
        if mc_choice(dE, T, uni):
            spins_t[site] *= -1
    
    
        

@cython.boundscheck(False)
def mc_step(cnp.int32_t[::1] spins_s,
            cnp.int32_t[::1] spins_t,
            cnp.int32_t[:, :] neighbors,
            double T):
    """perform L*L flips for 1 MC step"""
    
    cdef npy_intp num_steps = spins_s.shape[0]
    cdef int _
    
    cdef cnp.ndarray[double,
                ndim=1,
                negative_indices=False,
                mode='c'] unis = np.random.uniform(size=num_steps)
    cdef cnp.ndarray[npy_intp,
                ndim=1,
                negative_indices=False,
                mode='c'] sites = np.random.randint(num_steps, size=num_steps)
    cdef cnp.ndarray[npy_intp,
                ndim=1,
                negative_indices=False,
                mode='c'] lattice = np.random.randint(2, size=num_steps)
    
    for _ in range(num_steps):
        step(spins_s, spins_t, neighbors, T, lattice[_], sites[_], unis[_])
       

@cython.boundscheck(False)  
# @cython.cdivision(True)
cdef double calc_e_c(cnp.int32_t[::1] spins_s,
                  cnp.int32_t[::1] spins_t,
                  cnp.int32_t[:, :] neighbors):
    cdef npy_intp L2 = spins_s.shape[0]
    cdef int site,j,idx
    cdef int E = 0
    cdef double r
    
    for site in range(L2):
        for j in range(2):
            idx = neighbors[site, j]
            E += (spins_s[site]*spins_s[idx] + spins_t[site]*spins_t[idx] + 
                  spins_s[site]*spins_s[idx]*spins_t[site]*spins_t[idx])
    r = -E/L2
    return r

def calc_e(cnp.int32_t[::1] spins_s,
                  cnp.int32_t[::1] spins_t,
                  cnp.int32_t[:, :] neighbors):
    cdef double E
    E = calc_e_c(spins_s, spins_t, neighbors)
    return E


@cython.boundscheck(False)  
# @cython.cdivision(True)
cdef double calc_m_c(cnp.int32_t[::1] spins):
    cdef npy_intp L2 = spins.shape[0]
    cdef int site,j,idx
    cdef int M = 0
    cdef double r
    
    for site in range(L2):
        M += spins[site]
    r = M/L2
    return r

def calc_m(cnp.int32_t[::1] spins):
    cdef double M
    M = calc_m_c(spins)
    return M