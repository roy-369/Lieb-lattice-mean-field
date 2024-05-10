# Spinless 1D chain with NN Hubbard repulsion

import numpy as np
import numpy.linalg as LA

from tqdm import tqdm

import time

t_start = time.time()

# parameters

Ns = 16   # number of sites
Np = 6   # number of particles, filling = Np/Ns

MCsteps = int(1.e4)

burn = 1.e3  # MC steps to throw away
Nbins = 100    # Number of bins for MC average

dg = 10.0    # g goes from 0 to 1 in units of dg


# useful functions
def pbc(n):     # implements Periodic Boundary Conditions
    if n<0:
        return n+Ns
    elif n>=Ns:
        return n-Ns
    else:
        return n

def Metropolis(ratio):  # implements Metropolis MC
    r = np.random.random()
    if ratio > r:
        return True
    else:
        return False

def MCBins(MCQuantity):
    # given a MC Quantity as a function of MC steps
    # this function bins the date and returns the 
    # mean and error bar
    MCQuantity = np.asarray(MCQuantity)
    MCbins = np.array_split(MCQuantity, Nbins)
    mean_ar = [np.mean(abin) for abin in MCbins]

    return np.mean(mean_ar), np.std(mean_ar)


# The beginning is a very delicate time...

# single particle states
all_sp_states = np.arange(-np.pi, np.pi, 2*np.pi/Ns)
energy = -2*np.cos(all_sp_states)
idx = energy.argsort()[:]
all_sp_states_energy_sorted = all_sp_states[idx]

# for our problem, we only need Np of the lowest energy states
sp_states = all_sp_states_energy_sorted[0:Np]

# kinetic energy
ke_exact = np.sum( -2*np.cos(sp_states) )   # -t cos(k)

t1 = time.time()
print('cp1: Time elapsed %fs'%(t1-t_start))
print('Exact K.E. ',ke_exact,'\n')


# slater determinant

def slaterPsi(particle_positions):
    slater = np.zeros([Np,Np]).astype(complex)
    for i in range(Np):
        for j in range(Np):
            ki = sp_states[i]
            rj = particle_positions[j]
            slater[i,j] = np.exp(1j*ki*rj)  # 1j = imaginary i in python
    return np.asmatrix(slater)

# jastrow factor - correlations
def jasPsi(particle_positions,g):
    particle_positions = np.asarray(particle_positions)
    # jastrow factor
    jastrow = 1.0
    for p1 in particle_positions:
        if pbc(p1+1) in particle_positions:  # right ngbr
            jastrow += 1 
        if pbc(p1-1) in particle_positions:  # left ngbr
            jastrow += 1 

    return g**(jastrow/Np)

def Dprime(D0, detD0, D0bar, D1, part_index):
    # gives detD1, D1bar as output -- *without* calculating det(D1) and inv(D1).T :: ref Nandini Notes

    j0 = part_index     # particle that has moved
    Dinv = D0bar.T
    diff_pos = np.asarray((D1-D0)[:,j0])
    ek = np.zeros([len(D0),1])
    ek[j0,0] = 1
    qRatio = (1.0 + ek.T*Dinv*diff_pos)[0,0]
    detD1 = qRatio*detD0

    if np.abs( detD1 ) > 0.0:       
        D1inv = Dinv - (Dinv*diff_pos*ek.T*Dinv)/qRatio       
        return detD1, D1inv.T
    else:
        return 0.0, np.eye(Np)


def MC_Configs(g):

    rij_list = []       # list of MC configurations
    d0_list = []
    detD0_list = []
    d0bar_list = []

    if ( float(Np/Ns) == 0.5 ):
        pos = np.arange(0,Ns,2)
    else:
        pos = np.random.choice(Ns, Np, replace=False)   # starting position
    
    d0 = slaterPsi(pos)
    d0bar = LA.inv(d0).T
    detD0 = LA.det(slaterPsi(pos))
    jas0 = jasPsi(pos,g)

    acpr = 0.0

    for step in range(MCsteps):
        
        prob = (jas0**2)*(np.abs(detD0))**2

        # go to each particle
        for particle in range(Np):
            new_pos = np.copy(pos)
            # change its position
            kick = np.random.choice(np.setdiff1d(np.arange(Ns),pos))
            # update the new configuration
            new_pos[particle] = kick
            # find the slater state
            D1 = slaterPsi(new_pos)

            detD1, d1bar = Dprime(d0, detD0, d0bar, D1, particle)

            jas1 = jasPsi(new_pos,g)

            prob_new = (jas1**2)*(np.abs(detD1))**2

            # is it any better?
            ratio = prob_new/prob

            # should we accept?
            if Metropolis(ratio):
            # make the move
                pos = new_pos
                prob = prob_new
                detD0 = detD1
                d0bar = d1bar
                d0 = slaterPsi(new_pos)
                acpr += 1.0/(MCsteps*Np)

        if step > burn:
            rij_list.append(pos)
            detD0_list.append(detD0)
            d0_list.append(d0)
            d0bar_list.append(d0bar)

    return rij_list, detD0_list, d0_list, d0bar_list

# observables
# given a configuration, how do you calculate total Energy?
# no interactions, only KE

def KEconfig(config, detD0, d0, d0bar,g):
    kec = 0.0
    jas0 = jasPsi(config,g)
    if np.abs(jas0) > 0.0:
        for particle in range(Np):
            new_config = np.copy(config)
            new_config[particle] = pbc(config[particle]+1)
            jas1 = jasPsi(new_config,g)
            D1 = slaterPsi(new_config)
            detD1, d1bar = Dprime(d0, detD0, d0bar, D1, particle)
            kec += (detD1/detD0)*(jas1/jas0)

            new_config = np.copy(config)
            new_config[particle] = pbc(config[particle]-1)
            jas1 = jasPsi(new_config,g)
            D1 = slaterPsi(new_config)
            detD1, d1bar = Dprime(d0, detD0, d0bar, D1, particle)
            kec += (detD1/detD0)*(jas1/jas0)

    return -kec     # since hopping is -t

def NNCorr(config, detD0, d0, d0bar,g,dist):
    nnc = 0.0
    jas0 = jasPsi(config,g)
    if np.abs(jas0) > 0.0:
        for particle in range(Np):
            new_config = np.copy(config)
            new_config[particle] = pbc(config[particle]+dist)
            jas1 = jasPsi(new_config,g)
            D1 = slaterPsi(new_config)
            detD1, d1bar = Dprime(d0, detD0, d0bar, D1, particle)
            nnc += (detD1/detD0)*(jas1/jas0)

    return nnc     # since hopping is -t

def PEconfig(config):
    pec = 0.0
    for pos in config:
        ngbr = pbc(pos-1)
        if ngbr in config:
            pec += 1.0
        ngbr = pbc(pos+1)
        if ngbr in config:
            pec += 1.0   
             
    return pec      # in units of U (repulsive > 0)

#################################################################
#                       Main Function                           #
#################################################################

gar = np.arange(0.0,1.0+dg,dg)

KEg_ar = []
PEg_ar = []
KEg_err_ar = []
PEg_err_ar = []

for g in tqdm(gar):
    rij_list, detD0_list, d0_list, d0bar_list = MC_Configs(g)

    KE = np.zeros(len(rij_list))
    PE = np.zeros(len(rij_list))

    for i in range(len(rij_list)):
        config = rij_list[i]
        detD0 = detD0_list[i]
        d0 = d0_list[i]
        d0bar = d0bar_list[i]
        KE[i] = KEconfig(config, detD0, d0, d0bar,g)
        PE[i] = PEconfig(config)

    # write PE to external file -- debug mode
    np.save( ('data/PE_MC_g%3.2f'%g).replace('.','p') , PE)

    ke_mean, ke_std = MCBins(KE)
    pe_mean, pe_std = MCBins(PE)

    KEg_ar.append( ke_mean )
    PEg_ar.append( pe_mean )
    KEg_err_ar.append( ke_std )
    PEg_err_ar.append( pe_std )

    
fname_suffix = ('_dg%3.2f_fl_%3.2f_%dNbins'%(dg, float(Np)/Ns, Nbins)).replace('.','p')


np.save('data/KEg'+fname_suffix,KEg_ar)
np.save('data/PEg'+fname_suffix,PEg_ar)
np.save('data/KEg_err'+fname_suffix,KEg_err_ar)
np.save('data/PEg_err'+fname_suffix,PEg_err_ar)

print('Saved to ', 'data/KEg'+fname_suffix+'.npy')