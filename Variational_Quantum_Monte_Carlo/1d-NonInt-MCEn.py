# Non-interacting System

import numpy as np
import numpy.linalg as LA

from tqdm import tqdm

# parameters

Ns = 100   # number of sites
Np = 20   # number of particles, filling = Np/Ns

MCsteps = int(1.e4)

burn = 0.0  # MC steps to throw away
Nbins = 1.e3    # Number of bins for MC average

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

print('Exact K.E. ',ke_exact)

# slater determinant

def slaterPsi(particle_positions):
    slater = np.zeros([Np,Np]).astype(complex)
    for i in range(Np):
        for j in range(Np):
            ki = sp_states[i]
            rj = particle_positions[j]
            slater[i,j] = np.exp(1j*ki*rj)  # 1j = imaginary i in python
    return np.asmatrix(slater)

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


rij_list = []       # list of MC configurations
d0_list = []
detD0_list = []
d0bar_list = []

pos = np.random.choice(Ns, Np, replace=False)   # starting position

d0 = slaterPsi(pos)
d0bar = LA.inv(d0).T
detD0 = LA.det(slaterPsi(pos))

acpr = 0.0

for step in tqdm(range(MCsteps)):
    
    prob = (np.abs(detD0))**2

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

        prob_new = (np.abs(detD1))**2

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

print('acceptance ratio ',acpr,'\n')

# observables
# given a configuration, how do you calculate total Energy?
# no interactions, only KE

def KEconfig(config, detD0, d0, d0bar):
    kec = 0.0
    for particle in range(Np):
        new_config = np.copy(config)
        new_config[particle] = pbc(config[particle]+1)
        D1 = slaterPsi(new_config)
        detD1, d1bar = Dprime(d0, detD0, d0bar, D1, particle)
        kec += (detD1/detD0)

        new_config = np.copy(config)
        new_config[particle] = pbc(config[particle]-1)
        D1 = slaterPsi(new_config)
        detD1, d1bar = Dprime(d0, detD0, d0bar, D1, particle)
        kec += (detD1/detD0)

    return -kec     # since hopping is -t


KE = np.zeros(len(rij_list))

for i in tqdm(range(len(rij_list))):
    config = rij_list[i]
    detD0 = detD0_list[i]
    d0 = d0_list[i]
    d0bar = d0bar_list[i]
    KE[i] = KEconfig(config, detD0, d0, d0bar)
for config, detD0, d0, d0bar in zip(rij_list,detD0_list,d0_list,d0bar_list):
    KE[i] = KEconfig(config, detD0, d0, d0bar)

for arg in zip(rij_list,detD0_list,d0_list,d0bar_list):
    KE[i] = KEconfig(*arg)

    
fname_suffix = ('_Nb%d_fl_%3.2f'%(Nbins, float(Np)/Ns)).replace('.','p')

np.save('data/KE'+fname_suffix,KE)

print('data saved to data/KE'+fname_suffix+'.npy')