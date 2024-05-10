import numpy as np
import numpy.linalg as LA

from tqdm import tqdm

# parameters

Ns = 100   # number of sites
Np = 10   # number of particles, filling = Np/Ns

MCsteps = int(1.e5)

burn = 5.e3     # how many steps to burn
sweep = 1.e3    # sample configs after what steps

dg = 0.1       # resolution for g values

def pbc(n):
    if n<0:
        return n+Ns
    elif n>=Ns:
        return n-Ns
    else:
        return n


# single particle states

all_sp_states = np.arange(-np.pi, np.pi, 2*np.pi/Ns)
energy = all_sp_states*all_sp_states
idx = energy.argsort()[:]
all_sp_states_energy_sorted = all_sp_states[idx]
# for our problem, we only need Np of the lowest states
sp_states = all_sp_states_energy_sorted[0:Np]

# slater determinants
def slaterPsi(particle_positions):

    # slater determinant
    slater = np.zeros([Np,Np]).astype(complex)

    for i in range(Np):
        for j in range(Np):
            ki = sp_states[i]
            rj = particle_positions[j]
            slater[i,j] = np.exp(1j*ki*rj)
    
    return np.asmatrix(slater)

def jasPsi(particle_positions,g):
    # jastrow factor
    jastrow = 1.0
    for p1 in particle_positions:
        ngbr = [pbc(p1-1),pbc(p1+1)]
        if ngbr in particle_positions:  # neareast neighbors
            jastrow = jastrow*g

    return jastrow


def Dprime(D0, detD0, D0bar, new_config, part_index):
    # gives detD1, D1bar as output -- *without* calculating det(D1) and inv(D1).T :: ref Nandini Notes

    j0 = part_index     # particle that has moved

    D1 = slaterPsi(new_config)

    qRatio = np.dot( D1[:,j0].T,D0bar[:,j0] )[0,0]
    detD1 = qRatio*detD0

    DbarAr = D0bar[:,j0]
    M = (D0bar.T)*(D1-D0)

    D1bar = D0bar - np.outer(D0bar[:,j0],M[:,j0])/qRatio

    return detD1, D1bar


# now let's do a MC step

# start with random position

def GiveList(g):

    rij_list = []
    d0_list = []
    detD0_list = []
    d0bar_list = []

    pos = np.random.choice(Ns, Np, replace=False)   # starting position

    jas0 = jasPsi(pos,g)
    d0 = slaterPsi(pos)
    d0bar = LA.inv(d0).T
    detD0 = LA.det(slaterPsi(pos))

    for step in range(MCsteps):
        
        prob = (jas0*np.abs(detD0))**2
    
        # select a particle at random
        pp_rand = np.random.randint(Np)
        # change its position
        new_pos = np.copy(pos)
        # give it a kick
        kick = np.random.choice(np.setdiff1d(np.arange(Ns),pos))
        # update the position
        new_pos[pp_rand] = kick

        detD1, d1bar = Dprime(d0, detD0, d0bar, new_pos, pp_rand)
        jas1 = jasPsi(new_pos,g)

        prob_new = (jas1*np.abs(detD1))**2

        # is it any better?
        ratio = prob_new/prob

        # make the move
        if (ratio > 1):
            pos = new_pos
            prob = prob_new
            detD0 = detD1
            d0bar = d1bar
            d0 = slaterPsi(new_pos)
#            acpr += 1.0/MCsteps
        else:
            r = np.random.random()  # metropolis step
            if (r < ratio ):
                pos = new_pos
                prob = prob_new
                detD0 = detD1
                d0bar = d1bar
                d0 = slaterPsi(new_pos)     # acpr += 1.0/MCsteps     # acceptance ratio

        if (step > burn and step%sweep ==0):
            rij_list.append(pos)
            detD0_list.append(detD0)
            d0_list.append(d0)
            d0bar_list.append(d0bar)

    return rij_list, detD0_list, d0_list, d0bar_list
#print('acceptance ratio', acpr)
    


# observables
# given a configuration, how do you calculate total Energy?

def PEconfig(config):
    pec = 0.0
    for pos in config:
        ngbr = pbc(pos-1)
        config = np.asarray(config)
        config = config.tolist()
        if ngbr in config:
            pec += 1.0
        ngbr = pbc(pos+1)
        if ngbr in config:
            pec += 1.0   
             
    return pec

def KEconfig(config, detD0, d0, d0bar,g):
    kec = 0.0
    jas0 = jasPsi(config,g)
    for particle in range(Np):
        new_config = np.copy(config)
        new_config[particle] = pbc(config[particle]+1)
        #print(config, new_config, particle,kec,'chk\n')
        detD1, d1bar = Dprime(d0, detD0, d0bar, new_config, particle)
        jas1 = jasPsi(new_config,g)
        kec += (detD1/detD0)*(jas1/jas0)

        new_config = np.copy(config)
        new_config[particle] = pbc(config[particle]-1)
        #print(config, new_config, particle,kec,'chk\n')
        detD1, d1bar = Dprime(d0, detD0, d0bar, new_config, particle)
        jas1 = jasPsi(new_config,g)
        kec += (detD1/detD0)*(jas1/jas0)

    return -kec


# let us calculate energy for g=1, U=0 (no PE)

gar = np.arange(0,1+dg,dg)

KE_g = []
PE_g = []
KE_g_err = []
PE_g_err = []

for g in tqdm(gar):
    rijlist, detlist, d0list, d0barlist = GiveList(g)
    KE = []
    PE = []
    for i in range(len(rijlist)):
        config = rijlist[i]
        detD0 = detlist[i]
        d0 = d0list[i]
        d0bar = d0barlist[i]
        KE.append( KEconfig(config, detD0, d0, d0bar,g) )
        PE.append( PEconfig(config) )


    KE_g.append(np.mean(KE))
    KE_g_err.append(np.std(KE))
    PE_g.append(np.mean(PE))
    PE_g_err.append(np.std(PE))

fname_suffix = ('_dg%3.2f_fl_%3.2f'%(dg, float(Np)/Ns)).replace('.','p')

np.save('data/KEg'+fname_suffix,KE_g)
np.save('data/PEg'+fname_suffix,PE_g)
np.save('data/KEg_err'+fname_suffix,KE_g_err)
np.save('data/PEg_err'+fname_suffix,PE_g_err)