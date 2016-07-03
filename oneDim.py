
# packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from types import *
import warnings
import time
import pickle
import os
import peakutils

try:
    from scipy.integrate import ode
    from scipy.signal import find_peaks_cwt
    import scipy.ndimage as ndimage
except:
    warnings.warn('SciPy could not be imported')


def f(t, u, Fargs):
    # unwrap funciton arguments
    m = Fargs[0]
    a = Fargs[1]
    Delta = Fargs[2]
    n = Fargs[3]
    beta = Fargs[5]
    mu = Fargs[6]
    force = Fargs[7]
    
    # prepare arguments for main function evaluation
    a_left = a[:-1]
    a_right = a[1:]
    Delta_left = Delta[:-1]
    Delta_right = Delta[1:]
    n_left = n[:-1]
    n_right = n[1:]
    u_left = u[:-2]
    u_right = u[2:]
    u = u[1:-1]
    
    return (mu*(beta**2.0)/(m))*( n_left*a_left*np.maximum(Delta_left + u_left - u,0.0)**(n_left - 1.0) - n_right*a_right*np.maximum(Delta_right + u - u_right,0.0)**(n_right - 1.0) ) + ((beta**2.0)/m)*force

# function for the RHS of the e.o.m. in the expanded system
def F(t, z, Fargs):
    N = Fargs[4]
    vec_return = np.zeros(2.0*N+2)
       
    # u' = v
    vec_return[0] = 0
    vec_return[1:N+1] = z[N+2:]
    vec_return[N+1] = 0
    
    # v' = f(u)
    vec_return[N+2:] = f(t, z[:N+2], Fargs)
    return vec_return

# other functions
def order_of_magnitude(x):
    return int(np.log10(x))

# chain class
class Chain(object):
    """
    
    Class of granular chains.
    
    Members
    -------
    
    N : int
        Number of beads in the chain. (default=2)
    m : array-like
        Masses of the beads in kg. (default=np.zeros(N))
    R : array-like
        Radii of the beads in meters. (default=np.zeros(N))
    a : array-like
        Prefactors containing elastic properties of the beads, always in simulation units. (default=np.zeros(N-1))
    force : array-like
        External force acting on the beads in SI units. (default=np.zeros(N))
    Delta : array-like
        Bead precompression in meters. (default=np.zeros(N - 1))
    n : array-like
        Exponents of interaction potential. (default=2.5*np.ones(N - 1))
    beta : float
        Time scaling parameter. Has to be set as sqrt(m0/(n*mu)) for some base mass m0. (default=1.0)
    m0 : float
        Base mass used for calculation of beta. (default=1.0)
    mu : float
        Prefactor scaling parameter. (default=1.0)
    
    """
    N = 2                           # number of beads
    m = np.zeros(N)                 # masses of beads
    R = np.zeros(N)                 # bead radii
    a = np.zeros(N - 1)             # prefactors
    force = np.zeros(N)             # external force
    Delta = np.zeros(N - 1)         # precompression
    n = 2.5*np.ones(N - 1)          # powers in Hertz potential
    beta = 1.0                      # time scaling
    m0 = 1.0                        # base mass
    mu = 1.0                        # prefactor scaling
    
    # initialization      
    def __init__(self, N_value=2, m_values=None, R_values=None, a_values=None, force_values=None, n_values=None, Delta_values=None, beta_value=1.0, m0_value=1.0, mu_value=1.0):
        assert type(N_value) is IntType
        self.N = N_value
        
        if m_values is None:
            self.m = np.zeros(self.N)
        else:
            self.m = m_values
        
        if R_values is None:
            self.R = np.zeros(self.N)
        else:
            self.R = R_values
        
        if a_values is None:
            self.a = np.zeros(self.N-1)
        else:
            self.a = a_values
        
        if force_values is None:
            self.force = np.zeros(self.N)
        else:
            self.force = force_values
        
        if n_values is None:
            self.n = 2.5*np.ones(self.N-1)
        else:
            self.n = n_values
        
        if Delta_values is None:
            self.Delta = np.zeros(self.N-1)
        else:
            self.Delta = Delta_values
        
        self.beta = beta_value
        self.m0 = m0_value
        self.mu = mu_value
    
    def save(self, filename='test'):
        """
        
        Function that saves the instance of the chain to a file.
        
        """
        file = open(filename+'.txt','w')
        pickle.dump(self, file)
        file.close()
    
    def load(self, filename='test'):
        """
        
        Function that loads the instance of the chain from a file.
        
        """
        file = open(filename+'.txt','r')
        loaded_chain = pickle.load(file)
        
        self.N      = loaded_chain.N
        self.m      = loaded_chain.m
        self.R      = loaded_chain.R
        self.a      = loaded_chain.a
        self.force  = loaded_chain.force
        self.Delta  = loaded_chain.Delta
        self.n      = loaded_chain.n
        self.beta   = loaded_chain.beta
        self.m0     = loaded_chain.m0
        self.mu     = loaded_chain.mu
        
        file.close()
    
    def is_complete(self):
        """
        
        Function checking whether the chain was set up correctly.
        
        Returns
        -------
        
        is_complete : bool
        
        """
        is_complete = True
        
        if (type(self.N) is not IntType) or self.N < 2:
            warnings.warn('N not set up properly.')
            is_complete = False
        
        if self.m is None or len(self.m) != self.N:
            warnings.warn('m not set up properly.')
            is_complete = False
        
        if self.R is None or len(self.R) != self.N:
            warnings.warn('R not set up properly.')
            is_complete = False
        
        if self.a is None or len(self.a) != self.N - 1:
            warnings.warn('a not set up properly.')
            is_complete = False
        
        if self.force is None or len(self.force) != self.N:
            warnings.warn('force not set up properly.')
            is_complete = False
        
        if self.Delta is None or len(self.Delta) != self.N - 1:
            warnings.warn('Delta not set up properly.')
            is_complete = False
        
        if self.n is None or len(self.n) != self.N - 1:
            warnings.warn('n not set up properly.')
            is_complete = False
        
        if self.beta < 0.0:
            warnings.warn('beta not set up properly.')
            is_complete = False
        
        if self.m0 < 0.0:
            warnings.warn('m0 not set up properly.')
            is_complete = False
        
        if self.mu < 0.0:
            warnings.warn('mu not set up properly.')
            is_complete = False
        
        return is_complete
    
    def set_scalings(self, beta_value=1.0, mu_value=1.0):
        self.beta = beta_value
        self.mu = mu_value
    
    # set unscaled prefactors
    def set_prefactors(self, a_values):
        self.a = a_values/self.mu
    
    # set already scaled prefactors
    def set_scaled_prefactors(self, a_values):
        self.a = a_values
    
    # calculate prefactors from elastic properties
    def calculate_prefactors(self, R_values=1.0, Y_values=1.0, sigma_values=1.0):
        # make arrays for arguments that were given as floats
        if type(R_values) is float or int:
            R = R_values*np.ones(self.N)
        else:
            R = R_values
        
        if type(Y_values) is float or int:
            Y = Y_values*np.ones(self.N)
        else:
            Y = Y_values
        
        if type(sigma_values) is float or int:
            sigma = sigma_values*np.ones(self.N)
        else:
            sigma = sigma_values
        
        self.a = np.ones(self.N-1)
        for i in range(len(self.a)):
            D = (3.0/4.0)*( (1.0 - sigma[i]**2.0)/Y[i] + (1.0 - sigma[i+1]**2.0)/Y[i+1] )
            R_eff = np.sqrt(R[i]*R[i+1]/(R[i] + R[i+1]))
            self.a[i] = (0.4/D)*R_eff/self.mu
    
    def set_Delta_from_force(self, F_value, from_bead=None, to_bead=None):
        """
        
        Function setting the apropriate precompression delta for a given force assuming a uniform chain. The force has to be give in SI units.
    
        Parameters
        ----------
        
        F_value : double
            Externally applied force in SI units (N).
        from_bead : int
            Index of first bead of the section of the chain for which to apply precompression. (default=None)
        to_bead : int
            Index of last bead of the section of the chain for which to apply precompression. (default=None)
        
        """
        # set whole chain as precompression section by default
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.N - 1
        
        # radii, masses and prefactors in precompression section
        R_section = self.R[from_bead:to_bead]
        m_section = self.m[from_bead:to_bead]
        a_section = self.a[from_bead:to_bead]
        n_section = self.n[from_bead:to_bead]
        
        # assertations for uniform chain in precompression section
        assert np.all( R_section == R_section[0] )
        assert np.all( m_section == m_section[0] )
        assert np.all( a_section == a_section[0] )
        assert np.all( n_section == n_section[0] )
        
        n = self.n_section[0]
        mu = self.mu
        a = a_section[0]
        
        Delta_value = (F_value/(n*mu*a))**(1.0/(n - 1.0))
        self.Delta[from_bead:to_bead] = Delta_value*np.ones_like(R_section)
    
    def apply_precompression(self, F_value, from_bead=None, to_bead=None):
        """
        
        Function setting the apropriate precompression delta and constant external force for a given precompression force. The force has to be give in SI units.
    
        Parameters
        ----------
        
        F_value : double
            Externally applied force in SI units (N).
        from_bead : int
            Index of first bead of the section of the chain for which to apply precompression. (default=None)
        to_bead : int
            Index of last bead of the section of the chain for which to apply precompression. (default=None)
        
        """
        # set whole chain as precompression section by default
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.N - 1
        
        self.set_Delta_from_force(F_value, from_bead, to_bead)
        self.force = np.zeros(self.N)
        self.force[from_bead] = F_value
        self.force[to_bead] = -F_value
        
    # check scalings
    def analyze_scalings(self):
        # masses
        m_max = np.max(self.m)
        m_max_order = order_of_magnitude( m_max )
        m_min_order = order_of_magnitude( np.min(self.m) )
        if m_max_order != m_min_order:
            print 'masses are of order 10^', m_min_order, 'to order 10^', m_max_order
        else:
            print 'masses are of order 10^', m_min_order
        
        # prefactors
        a_max_order = order_of_magnitude( np.max(self.a) )
        a_min_order = order_of_magnitude( np.min(self.a) )
        if a_max_order != a_min_order:
            print 'prefactors are of order 10^', a_min_order, 'to order 10^', a_max_order
        else:
            print 'prefactors are of order 10^', a_min_order
    
    def set_wall(self, mass_factor):
        assert self.m is not None
        self.m[-1] = mass_factor*self.m[-2]
    
    # get function arguments for simulation
    def get_Fargs(self):
        return [self.m, np.concatenate(([0], self.a, [0]), axis=0), np.concatenate(([0], self.Delta, [0]), axis=0), np.concatenate(([1], self.n, [1])), self.N, self.beta, self.mu, self.force]
    
    def get_bead_numbers(self):
        return np.arange(self.N)
    

# 1D simulation class
class Simulation_1D(object):
    """
    
    Class for one-dimensional simulations of granular chains.
    
    Members
    -------
    
    sim_chain : Chain object
        Chain on which to carry out the simulation. (default=None)
    
    t0 : float
        Start time of simulation. (default=0.0)
    
    t1 : float
        End time of simulation. (default=1.0)
    
    dt : float
        Time step after which the simulation results are stored. Non necessarily the timestep for the numerical integrator. (default=0.01)
    
    z0 : array-like
        Initial value for displacement (z0[:N]]) and velocity (z0[N:]]) of beads in chain, where N is the number of beads in sim_chain. (default=None)
    
    data : array-like
        2D array with size storing displacements, velocities and time at each time step. The first index corresponds to time in the simulation and the second index selects the stored value. (default=None)
    
    """
    sim_chain = None                # chain
    t0 = 0.0                        # start time
    t1 = 1.0                        # end time
    dt = 0.01                       # time step
    z0 = None                       # initial conditions
    data = None                     # simulation data
    
    # initialization
    def __init__(self, sim_chain_val=None, t0_value=0.0, t1_value=1.0, dt_value=0.01, z0_values=None):
        if sim_chain_val is None:
            self.sim_chain = Chain()
        else:
            self.sim_chain = sim_chain_val
        
        self.t0 = t0_value
        self.t1 = t1_value
        self.dt = dt_value
        
        if z0_values is None:
            self.z0 = np.zeros(self.sim_chain.N*2)
        else:
            self.z0 = z0_values
    
    # overlap variables
    def overlap(self, t_measure=None):
        assert self.data is not None
        time = self.get_time()
        N = self.get_N()
        
        if t_measure is not None:
            # find index for time closest to t_measure
            idx = self._find_index_for_time(t_measure)
            
            # calculate overlap
            u = self.data[idx,:N]
            u_left = u[:-1]
            u_right = u[1:]
            overlap = np.maximum( u_left - u_right,0 )
            return overlap
            
        else:
            overlaps = np.zeros((len(time),N-1))
            ctr = 0
            for tau in time:
                overlaps[ctr,:] = self.overlap(tau)
                ctr += 1
            return overlaps
    
    def save(self, dirname='testdir', chainname='testchain', dataname='data'):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.sim_chain.save(dirname+'/'+chainname)
        self.save_data_as_txt(dirname+'/'+dataname)
    
    def save_data_as_npy(self, filename='test'):
        np.save( filename, self.data )
    
    def save_data_as_txt(self, filename='test', fmt='%.4e'):
        np.savetxt( filename, self.data, fmt=fmt )
        
    # shrink data
    def shrink_data(self, factor):
        assert self.data is not None
        self.data = self.data[::factor, :]
    
    def shrink_data_to(self, n_data):
        assert self.data is not None
        factor = int( float(self.data.shape[0])/float(n_data) )
        if factor < 1:
            warnings.warn('Data is already smaller than suggested size.')
        else:
            self.data = self.data[::factor, :]
    
    # run simulation
    def run(self, rtol=1e-6, print_runtime=False):
        """
        
        Runs the simulation and stores results in data.
        
        Parameters
        ----------
        
        rtol : float, optional
            Relative tolerance for solution. (default=1e-6)
        print_runtime : bool, optional
            Print runtime of solver. (default=False)
        
        """
        if print_runtime:
            start = time.time()
        
        assert self.sim_chain is not None
        assert self.sim_chain.is_complete()
        
        # get function arguments
        Fargs = self.sim_chain.get_Fargs()
        N = Fargs[4]
        
        # set up solver
        solver = ode(F, None).set_integrator('dopri5', verbosity=1, first_step=self.dt, atol=rtol)
        
        # initial condition in simulation form
        u_sim = np.concatenate(([0], self.z0[:N], [0]), axis=0)
        z0_sim = np.concatenate((u_sim,self.z0[N:]), axis=0)
        solver.set_initial_value(z0_sim, self.t0)
        
        # function arguments
        Fargs = self.sim_chain.get_Fargs()
        solver.set_f_params(Fargs)
        
        # data storage
        N = Fargs[4]
        n_steps = self.t1/self.dt + 1
        self.data = np.zeros((n_steps, 2*N + 1))
        
        # store initial condition
        self.data[0,:-1] = self.z0
        self.data[0,-1] = self.t0
        
        # run time integration
        ctr = 1
        while solver.successful() and solver.t < self.t1:
            # integrate
            solver.integrate(solver.t + self.dt)
            
            # store results
            self.data[ctr,:N] = solver.y[1:N+1]             # u values
            self.data[ctr,N:2*N] = solver.y[N+2:2*N+2]      # v values
            self.data[ctr,-1] = solver.t                    # time
            
            # stop integration if next step would take it too far in time
            if solver.t + self.dt > self.t1:
                break
            ctr += 1
        
        if solver.t < self.t1 - self.dt:
            print "Simulation didn't work! t_end =", solver.t
        
        if print_runtime:
            end = time.time()
            print 'runtime:', (end - start), 's'
    
    # get length of chain
    def get_N(self):
        assert self.sim_chain is not None
        return self.sim_chain.N
    
    # get array of times
    def get_time(self):
        assert self.data is not None
        return self.data[:,-1]
    
    # get velocities of beads at a given time
    def get_velocities(self, t_measure):
        self._check_time(t_measure)
        idx = self._find_index_for_time(t_measure)
        N = self.get_N()
        return self.data[idx,N:2*N]
    
    # bead velocity for a given bead
    def get_bead_velocity(self, bead_number, t_1=None, t_2=None):
        if t_1 is None:
            t_1 = self.t0
        if t_2 is None:
            t_2 = self.t1
        
        N = self.get_N()
        if bead_number >= N:
            warnings.warn('bead number out of range')
            return np.nan
        
        self._check_time(t_1)
        self._check_time(t_2)
        idx_1 = self._find_index_for_time(t_1)
        idx_2 = self._find_index_for_time(t_2)
        return self.data[idx_1:idx_2+1,N+bead_number]
    
    # get time difference for one index step in self.data
    def get_dt_per_index(self):
        """
        Get time difference for one index step in self.data assuming uniform time steps.
        """
        dt = self.data[1,-1] - self.data[0,-1]
        return dt
        
    # measure kinetic energy
    def measure_kinetic_energy(self, t_measure=None):
        """
        
        Function measuring the total kinetic energy for all times in the simulation (t_measure is None) or for one particular time t_measure. In the re-scaled equation of motion the kinetic energy is calculated by E_kin = 0.5*(m/m0)*v**2, where m0 is the base mass used for the time scaling parameter beta.
        
        
        Returns
        -------
        
        kinetic_energies: array-like (t_measure is None)
            Total kinetic energy of the chain for all time steps in the simulation.
        
        kinetic_energy : double (t_measure is not None)
            Total kinetic energy of the chain at time t_measure.
        
    
        Parameters
        ----------
        
        t_measure : double, optional
            Time in the simulation at which to measure the kinetic energy.
            
        """
        assert self.data is not None
        time = self.get_time()
        N = self.get_N()
        m0 = self.sim_chain.m0
        #print 'm0', m0
        
        if t_measure is not None:
            # find index for time closest to t_measure
            idx = self._find_index_for_time(t_measure)
            
            # calculate kinetic energy
            velocities = self.data[idx,N:2*N]
            masses = self.sim_chain.m
            kinetic_energy = 0.5*np.sum( np.multiply(masses/m0, velocities**2.0) )
            return kinetic_energy
        else:
            kinetic_energies = np.zeros(len(time))
            ctr = 0
            for tau in time:
                kinetic_energies[ctr] = self.measure_kinetic_energy(tau)
                ctr += 1
            return kinetic_energies
    
    # measure potential energy
    def measure_potential_energy(self, t_measure=None):
        """
        
        Function measuring the total potential energy of the chain for all times in the simulation (t_measure is None) or for one particular time t_measure. In the re-scaled equation of motion the potential energy is calculated from the potential V(overlap) = (a/n)*overlap**n .
        
        Returns
        -------
        
        potential_energies: array-like (t_measure is None)
            Total potential energy of the chain for all time steps in the simulation.
        potential_energy : double (t_measure is not None)
            Total potential energy of the chain at time t_measure.
        
    
        Parameters
        ----------
        
        t_measure : double, optional
            Time in the simulation at which to measure the potential energy.
        
        """
        assert self.data is not None
        time = self.get_time()
        
        if t_measure is not None:
            self._check_time(t_measure)
            
            # calculate potential energy
            overlap = self.overlap(t_measure)
            a = self.sim_chain.a
            n = self.sim_chain.n
            potential_energy = np.sum( np.multiply( (1.0/n), np.multiply( a, np.power(overlap,n) ) ) )
            
            return potential_energy
        
        else:
            # calculate potential energy
            overlaps = self.overlap()
            a = self.sim_chain.a
            n = self.sim_chain.n
            potential_energies = np.zeros(len(time))
            for i in range(len(time)):
                potential_energies[i] = np.sum( np.multiply( (1.0/n), np.multiply( a, np.power(overlaps[i,:],n) ) ) )
            
            return potential_energies
    
    # measure kinetic energy distribution
    def measure_kinetic_energy_distr(self, t_measure=None):
        """
        
        Function measuring the total kinetic energy for all times in the simulation (t_measure is None) or for one particular time t_measure. In the re-scaled equation of motion the kinetic energy is calculated by E_kin = 0.5*(m/m0)*v**2, where m0 is the base mass used for the time scaling parameter beta.
        
        
        Returns
        -------
        
        kinetic_energy_distrs: array-like (t_measure is None)
            Kinetic energy distributions of the chain for all time steps in the simulation.
        
        kinetic_energy_distr : double (t_measure is not None)
            Kinetic energy distribution of the chain at time t_measure.
        
    
        Parameters
        ----------
        
        t_measure : double, optional
            Time in the simulation at which to measure the kinetic energy.
            
        """
        assert self.data is not None
        time = self.get_time()
        N = self.get_N()
        m0 = self.sim_chain.m0
        #print 'm0', m0
        
        if t_measure is not None:
            # find index for time closest to t_measure
            idx = self._find_index_for_time(t_measure)
            
            # calculate kinetic energy
            velocities = self.data[idx,N:2*N]
            masses = self.sim_chain.m
            kinetic_energy_distr = 0.5*np.multiply(masses/m0, velocities**2.0)
            return kinetic_energy_distr
        else:
            kinetic_energy_distrs = np.zeros((len(time),N))
            ctr = 0
            for tau in time:
                kinetic_energy_distrs[ctr,:] = self.measure_kinetic_energy_distr(tau)
                ctr += 1
            return kinetic_energy_distrs
    
    # measure potential energy distribution
    def measure_potential_energy_distr(self, t_measure=None):
        """
        
        Function measuring the potential energy distribution of the chain. In the re-scaled equation of motion the potential energy is calculated from the potential V(overlap) = (a/n)*overlap**n .
        
        
        Returns
        -------
        
        potential_energy_distrs: array-like (t_measure is None)
            Potential energy distribution of the chain for all time steps in the simulation.
            
        potential_energy_distr : double (t_measure is not None)
            Potential energy distribution of the chain at time t_measure.
        
        
        Parameters
        ----------
        
        t_measure : double, optional
            Time in the simulation at which to measure the potential energy distribution.
        
        """
        assert self.data is not None
        time = self.get_time()
        
        a = self.sim_chain.a
        n = self.sim_chain.n
        N = self.get_N()
        
        if t_measure is not None:
            self._check_time(t_measure)
            
            # calculate potential energy
            overlap = self.overlap(t_measure)
            potential_energy_distr = np.multiply( (1./n), np.multiply(a, overlap**n) )
            
            return potential_energy_distr
        
        else:
            # calculate potential energy
            #overlaps = self.overlap()
            potential_energy_distrs = np.zeros((len(time),N-1))
            ctr = 0
            for t in time:
                potential_energy_distrs[ctr,:] = self.measure_potential_energy_distr(t)
                ctr += 1
            
            return potential_energy_distrs
    
    # measure total energy
    def measure_total_energy(self, t_measure=None):
        """
        
        Function measuring the total energy of the chain for all times in the simulation (t_measure is None) or for one particular time t_measure.
        
        Returns
        -------
        
        total_energies: array-like (t_measure is None)
        total_energy : double (t_measure is not None)
        
    
        Parameters
        ----------
        
        t_measure : double, optional
            Time in the simulation at which to measure the total energy.
        
        """
        assert self.data is not None
        
        if t_measure is not None:
            self._check_time(t_measure)
            
            # calculate total energy
            kinetic_energy = self.measure_kinetic_energy(t_measure)
            potential_energy = self.measure_potential_energy(t_measure)
            total_energy = kinetic_energy + potential_energy
            
            return total_energy
            
        else:
            # calculate total energy
            kinetic_energies = self.measure_kinetic_energy()
            potential_energies = self.measure_potential_energy()
            total_energies = kinetic_energies + potential_energies
            
            return total_energies
    
    def calculate_energy_conservation_error(self):
        """
        
        Function calculating the relative error in energy conservation between the start and the end of the simulation.
        
        Returns
        -------
        
        error : float
        
        """
        assert self.data is not None
        # calculate total energy at start and end of simulation
        energy_start = self.measure_total_energy(self.t0)
        energy_end = self.measure_total_energy(self.t1)
        
        # calculate accuracy
        error = abs(1.0 - energy_start/energy_end)
        
        return error
    
    def plot_velocity(self, t_plot):
        idx = self._find_index_for_time(t_plot)
        N = self.get_N()
        bead_numbers = self.sim_chain.get_bead_numbers()
        velocity = self.data[idx,N:2*N]
        plt.plot(bead_numbers, velocity, marker='.')
        plt.xlabel('bead number')
        plt.ylabel('velocity (computer units)')
        plt.show()
    
    def animate(self, interval=100, title='test', medium_from=None, medium_to=None):
        """
        
        Function creating an animation of the eave propagation in the chain.
        
        Returns
        -------
        
        anim : animation object        
    
        Parameters
        ----------
        
        interval : int, optional
            Time between frames in milliseconds. (default=100)
        
        title : str, optional
            Title of plot. (default='test')
        
        medium_from : int, optional
            Bead number at which a different medium in the chain is starting. (default=None)
        
        medium_to : int, optional
            Bead number at which a different medium in the chain is ending. (default=None)
        
        """
        # data input
        N = self.get_N()
        plotting_data = self.data[:,N:2*N]
        time = self.get_time()
        y_min = np.min(plotting_data)
        y_max = np.max(plotting_data)
        n_frames = plotting_data.shape[0]
        
        # initialization of plots
        fig = plt.figure()
        ax = plt.axes(xlim=(0, N), ylim=(y_min, y_max))
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        time_text.set_text('')
        l, = plt.plot([], [], '.-')
        plt.xlabel('bead number')
        plt.ylabel('velocity (computer units)')
        plt.title(title)
        
        # visualize interfaces
        if medium_from is not None:
            if medium_to is None:
                medium_to = N
            
            plt.fill_between(range(medium_from, medium_to+1), y_min, y_max, facecolor='red', alpha=0.5)
        
        # updat4e function for animation
        def update_line(num, plotting_data, time, line):
            dat = plotting_data[num,:]
            line.set_data([range(len(dat)), dat])
            time_text.set_text('time = %.1f' % time[num])
            line.set_label('t= 10')
            return line,
        
        line_ani = animation.FuncAnimation(fig, update_line, n_frames, fargs=(plotting_data, time, l), interval=interval, blit=False)
        return line_ani
    
    # peak finding routine 
    def find_peaks(self, t_measure):
        """
        Find the peaks of solitary waves. Only reliable for a single wave with no other disturbances.
        
        Parameters
        ----------
        
        t_measure : float
            Time at which to find the peaks.
        """
        self._check_time(t_measure)
        #widths = np.arange(2,7)  # range of widths to check by find_peaks_cwt
        #peak_nodes = find_peaks_cwt(self.get_velocities(t_measure), widths, min_snr=2.0,noise_perc=30.0)
        peak_beads = peakutils.peak.indexes(self.get_velocities(t_measure), thres=0.75, min_dist=7)
        return peak_beads
    
    # measure wave velocity
    def measure_wave_velocity(self, t_1, t_2, units='computer', shrink_range=False, max_tries=10):
        """
        
        Function measuring the wave velocity of a solitary wave by calculating the average speed in the time interval [t1,t2]. It is assumed and asserted that all bead radii are equal. This function is only guaranteed to work for a single wave propagating through a uniform medium with no other disturbances.
        
        Returns
        -------
        
        vel : float        
    
        Parameters
        ----------
        
        t_1 : double
            Start time for speed measurement. 
        
        t_2 : double
            Start time for speed measurement.
        
        units : str, optional
            Units in which to return the speed: 'computer' for computer units and 'SI' for SI units. (default='computer')
        
        shrink_range : bool, optional
            If velocity not determinable in given range iteratively try to determine on shrinked range. (default=False)
        
        max_tries : int, optional
            Maximum number of tries for shrinking the range. (default=10)
        
        """
        assert t_1<t_2
        assert units=='computer' or units == 'SI'
        assert np.all(self.sim_chain.R==self.sim_chain.R[0])
        
        # find the peaks at first time
        peak1 = self.find_peaks(t_1)
        
        if len(peak1) > 1:
            warnings.warn('Velocity cannot be determined unambiguously. More than one peak detected at first time.')
            return np.nan
        elif len(peak1) == 0:
            warnings.warn('Velocity cannot be determined. No peaks could be detected at first time.')
            return np.nan
        
        # find the peaks at second time
        peak2 = self.find_peaks(t_2)
        
        if len(peak2) > 1 or len(peak2) == 0 or peak2[0] > self.get_N()-10:
            if shrink_range == True:
                try_ctr = 0
                time_diff = t_2 - t_1
                alpha = 0.5                 # shrinking factor
                while try_ctr < max_tries:
                    # shrink range 
                    t_2 = t_1 + alpha*time_diff
                    peak2 = self.find_peaks(t_2)
                    time_diff = t_2 - t_1
                    try_ctr += 1
                    
                    # check if time_diff is too small
                    if time_diff < 20*self.dt:
                        warnings.warn('Time difference too small to determine wave speed.')
                        return np.nan
                    
                    # check if velocity determinable
                    if len(peak2) > 1 or len(peak2) == 0 or peak2[0] > self.get_N()-10:
                         pass
                    else:
                        # calculate velocity
                        vel = (peak2[0] - peak1[0])/time_diff
                        if units == 'computer':
                            return vel
                        elif units == 'SI':
                            R = self.sim_chain.R[0]
                            beta = self.sim_chain.beta
                            return vel*2.0*R/beta
                
                # if not succeeded by now return warning
                warnings.warn('Velocity cannot be determined unambiguously in given range.')
                return np.nan
            
            else:
                if len(peak2) > 1:
                    warnings.warn('Velocity cannot be determined unambiguously. More than one peak detected at seocond time.')
                    return np.nan
                elif len(peak2) == 0:
                    warnings.warn('Velocity cannot be determined. No peaks could be detected at second time.')
                    return np.nan
                elif peak2[0] > self.get_N()-10:
                    warnings.warn('The second peak is too close to the boundary. Wave could have left the chain.')
                    return np.nan
        else:
            # calculate velocity
            time_diff = t_2 - t_1
            vel = (peak2[0] - peak1[0])/time_diff
            if units == 'computer':
                return vel
            elif units == 'SI':
                R = self.sim_chain.R[0]
                beta = self.sim_chain.beta
                return vel*2.0*R/beta
        
    def measure_forces(self, from_time=None, to_time=None, units='computer'):
        """
        
        Function measuring the net forces on the beads by calculating their acceleration.
        
        Returns
        -------
        
        forces : array-like        
    
        Parameters
        ----------
        
        from_time : double
            Start time for speed measurement. 
        
        to_time : double
            Start time for speed measurement.
        
        units : str, optional
            Units in which to return the forces: 'computer' for computer units and 'SI' for SI units. (default='computer')
        
        """
        assert units=='computer' or units == 'SI'
        if from_time == None:
            from_time = self.t0
        if to_time == None:
            to_time = self.t1
        
        from_idx = self._find_index_for_time(from_time)
        to_idx = self._find_index_for_time(to_time)
        
        dt_ind = self.get_dt_per_index()
        #dt_ind = 1.0
        
        N = self.get_N()
        forces = np.zeros((self.data[from_idx:to_idx+1,:].shape[0], N))
        for bead in range(N):
            vel = self.get_bead_velocity(bead, from_time, to_time)
            acc = np.gradient(vel, dt_ind)
            mass = self.sim_chain.m[bead]
            forces[:,bead] = mass*acc
        
        if units=='computer':
            return forces
        elif units=='SI':
            return forces/(self.sim_chain.beta**2)
    
    def measure_max_forces(self, from_time=None, to_time=None, units='computer'):
        assert units=='computer' or units == 'SI'
        return np.max( self.measure_forces(from_time, to_time, units=units), axis=1 )
    
    def measure_force_at_wall(self, from_time=None, to_time=None, units='computer'):
        assert units=='computer' or units == 'SI'
        if from_time == None:
            from_time = self.t0
        if to_time == None:
            to_time = self.t1
        N = self.get_N()
        vel = self.get_bead_velocity(N-1, from_time, to_time)
        dt_ind = self.get_dt_per_index()
        acc = np.gradient(vel, dt_ind)
        mass = self.sim_chain.m[N-1]
        force = mass*acc
        
        if units=='computer':
            return force
        elif units=='SI':
            return force/(self.sim_chain.beta**2)
    
    def measure_wavelength(self, t_measure):
        """
        
        Function measuring the wavelength ov a wave in the granular chain. This is only working reliably for a single wave with no other disturbances.
        
        Returns
        -------
        
        wavelength : int
            Number of beads spanning the wave.
    
        Parameters
        ----------
        
        t_measure : double
            Time at which to measure the wavelength.
        
        """
        overlap = self.overlap(t_measure)
        
        # set all overlap entries below 0.1% of the maximum overlap to zero
        overlap[overlap<0.001*np.max(overlap)] = 0.0
        nonzero = np.nonzero(overlap)
        
        # find connected section of chain with nonzero overlap
        consecutives = np.split(nonzero, np.where(np.diff(nonzero) != 1)[0]+1)
        if len(consecutives) != 1:
            warnings.warn('Wavelength could not be determined unambiguously.')
            return np.nan
        else:
            # add 1 since overlap involves two beads each
            wavelength = len(consecutives[0][0]) + 1
            return wavelength
    
    def measure_wavelength_avg(self, from_time, to_time, print_n=False):
        """
        
        Function measuring the average wavelength of a wave in the granular chain. This is only working reliably for a single wave with no other disturbances.
        
        Returns
        -------
        
        wavelength_avg : float
            Average number of beads spanning the wave.
    
        Parameters
        ----------
        
        from_time : float
            Time at which to start the average wavelength measurement.
        
        to_time : float
            Time at which to end the average wavelength measurement.
        
        print_n : bool, optional
            Print number of wavelength measurements used for average. (default=False)
        
        """
        from_idx = self._find_index_for_time(from_time)
        to_idx = self._find_index_for_time(to_time)
        
        # print number of measurements
        if print_n:
            n_measurements = to_idx - from_idx + 1
            print 'n measurements:', n_measurements
        
        # calculate overlap
        overlap = self.overlap()[from_idx:to_idx+1,:]
        
        # intitialize wavelength storage
        wavelengths = np.zeros(overlap.shape[0])
        
        for i in range(overlap.shape[0]):
            this_overlap = overlap[i,:]
            
            # set all overlap entries below 0.1% of the maximum overlap to zero
            this_overlap[this_overlap<0.001*np.max(this_overlap)] = 0.0
            
            nonzero = np.nonzero(this_overlap)
            
            # find connected section of chain with nonzero overlap
            consecutives = np.split(nonzero, np.where(np.diff(nonzero) != 1)[0]+1)
            
            if len(consecutives) != 1:
                warnings.warn('Wavelength could not be determined unambiguously.')
                return np.nan
            else:
                # add 1 since overlap involves two beads each
                wavelengths[i] = float( len(consecutives[0][0]) ) + 1
            
        return np.mean( wavelengths )
    
    def measure_multipulse_energies(self, t_measure, n, cut_per=0.001):
        """
        
        Function measuring the energies carried by the first n peaks of a multipulse structure. This function assumes that the section of the chain in which the multipulse structure propagates is uniform.
        
        Returns
        -------
        
        energies : array-like
            Energies of first n peaks.
    
        Parameters
        ----------
        
        t_measure : float
            Time of measurement.
        
        n : int
            Number of peaks to detect.
        
        cut_per : float, optional
            Cut-off value for finding the right end of the largest peak relative to its size. (default=0.001)
        
        """
        pulse_bounds = self.find_multipulse_bounds(t_measure, n, cut_per=cut_per)
        kinetic_energy_distr = self.measure_kinetic_energy_distr(t_measure)
        potential_energy_distr = self.measure_potential_energy_distr(t_measure)
        energies = np.zeros(len(pulse_bounds)-1)
        
        for i in range( len(pulse_bounds[:-1]) ):
            idx = pulse_bounds[i]
            idx_next = pulse_bounds[i+1]
            kin_energy = np.sum( kinetic_energy_distr[idx:idx_next] )
            pot_energy = np.sum( potential_energy_distr[idx:idx_next] )
            energies[i] = kin_energy + pot_energy
        
        return energies
    
    def measure_multipulse_separation(self, t_measure, n, vel_per=0.001):
        """
        
        Function measuring whether or not the peaks in the multipulse structure are separated enough using the minimum value of the velocity between two peaks as a measure for separation. If this velocity is smaller than vel_per times the velocity of the leading pulse, two peaks are said to be separated.
        
        Returns
        -------
        
        separation : array-like
            Boolean values for the separation between the peaks.
        
        
        Parameters
        ----------
        
        t_measure : float
            Time of measurement.
        
        n : int
            Number of peaks to detect.
        
        vel_per : float, optional
            Bound for separation between two peaks as percentage of the velocity of the leading peak. (default=0.001)
        
        """
        pulse_peaks = self.find_multipulse_peaks(t_measure, n)
        velocities = self.get_velocities(t_measure)
        max_vel = np.max( velocities[pulse_peaks] )
        pulse_bounds = self.find_multipulse_bounds(t_measure, n)
        separation = ( velocities[pulse_bounds]<(vel_per*max_vel) )
        return separation
        
    def find_multipulse_bounds(self, t_measure, n, cut_per=0.001):
        """
        
        Function finding the boundaries of the mulitipulse peaks.
        
        Returns
        -------
        
        pulse_bounds : array-like
            Bounds of first n peaks.
        
        
        Parameters
        ----------
        
        t_measure : float
            Time of measurement.
        
        n : int
            Number of peaks to detect.
        
        cut_per : float, optional
            Cut-off value for finding the right end of the largest peak relative to its size. (default=0.001)
        
        """
        pulse_idx = self.find_multipulse_peaks(t_measure, n)
        velocities = self.get_velocities(t_measure)
        
        
        # find minima around pulses to determine pulse extension
        """
        Determine the left end of the smallest peak by finding the minimum of the velocities through the change of sign in the gradient.
        """
        pulse_bounds = np.zeros(len(pulse_idx) + 1, dtype=int)
        vel_grad = np.gradient(velocities)
        pulse_bounds[0] = np.max( np.argwhere( vel_grad[:pulse_idx[0]]<0 ) ) + 1
        # Select the minimum between adjacent peaks as the boundary
        for i in range( len(pulse_idx[:-1]) ):
            idx = pulse_idx[i]
            idx_next = pulse_idx[i+1]
            pulse_bounds[i+1] = idx + np.argmin( velocities[idx:idx_next+1] )
        
        """
        Determine the right end of the largest peak by finding the first bead below the cut-off value velocity.
        """
        max_value = np.max( velocities[pulse_idx[-1]] )
        pulse_bounds[-1] = pulse_idx[-1] + np.min(np.argwhere( velocities[pulse_idx[-1]:]<(cut_per*max_value) ))
        
        return pulse_bounds
    
    def find_multipulse_peaks(self, t_measure, n, cut_per=0.001):
        """
        
        Function finding the first n peaks of a multipulse structure.
        
        Returns
        -------
        
        peaks : array-like
            Inidces of first n peaks.
    
        Parameters
        ----------
        
        t_measure : float
            Time of measurement.
        
        n : int
            Number of peaks to detect.
        
        cut_per : float, optional
            Cut-off value for accepting maxima relative to the size of the largest peak. (default=0.001)
        
        """
        velocities = self.get_velocities(t_measure)
        
        # calculate local maxima
        peaks = local_maxima(velocities)
        
        # apply cut-off
        max_value = np.max( velocities[peaks] )
        peaks = peaks[ np.abs(velocities[peaks])>(cut_per*max_value)  ]
        
        return peaks[-n:]
      
    def find_exit_time(self):
        """
        
        Function that finds the time for which the wave reaches the end of the chain. This only works if the wave is travelling from left to right and if there are no other disturbances.
        
        Returns
        -------
        
        exit_time : float

        """
        
        # check when last bead starts moving
        vel_last_bead = self.get_bead_velocity(self.get_N()-1)
        nonzero = np.flatnonzero( vel_last_bead )
        
        if nonzero.size == 0:
            warnings.warn('The wave did not exit the chain or it could not be determined.')
            return np.nan
        else:
            exit_index = np.min( nonzero )
            time = self.get_time()
            exit_time = time[exit_index]
            return exit_time
         
    # check if a provided time is in the time span of the simulation
    def _check_time(self, t_provided):
        assert self.data is not None
        time = self.get_time()
        if t_provided < time.min() - self.dt:
            warnings.warn('provided time %s smaller than simulation start time' %str(t_provided), stacklevel=2)
        if t_provided > time.max() + self.dt:
            warnings.warn('provided time %s larger than simulation end time' %str(t_provided), stacklevel=2)
    
    def _find_index_for_time(self, t_provided):
        self._check_time(t_provided)
        time = self.get_time()
        idx = (np.abs(time - t_provided)).argmin()
        return idx
    

# other functions 
def vel_force_function(sigma, Y, m, R, Fm):
    theta = 3*(1-sigma**2)/(4*Y)
    C = np.sqrt( (2*R**3)/(m*theta) )
    return C*(np.sqrt(0.8))*( 2*theta*Fm/(R**2) )**(1.0/6.0)

def local_maxima(a):
    a_left = a[:-2]
    a_right = a[2:]
    a_max = np.maximum(a_left, a_right)
    a = a[1:-1]
    comp = a>a_max
    maxima = np.swapaxes( np.argwhere( comp == True) + 1, 0, 1 )[0]
    return maxima

def local_maxima2(array, min_distance = 1, periodic=False, edges_allowed=True): 
    """Find all local maxima of the array, separated by at least min_distance."""
    array = np.asarray(array)
    cval = 0 
    
    if periodic: 
            mode = 'wrap' 
    elif edges_allowed: 
            mode = 'nearest' 
    else: 
            mode = 'constant' 
    cval = array.max()+1 
    max_points = array == ndimage.maximum_filter(array, 1+2*min_distance, mode=mode, cval=cval) 
    
    return [indices[max_points] for indices in np.indices(array.shape)][0]
