# This module implements path integral MD integrator using normal mode coordinates
#
# Written by Konrad Hinsen
#

#cython: boundscheck=False, wraparound=False, cdivision=True

"""
Path integral MD integrator using normal-mode coordinates
"""

__docformat__ = 'restructuredtext'

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New

from libc.stdint cimport int32_t
import numpy as N
cimport numpy as N

from MMTK import Units, ParticleProperties, Features, Environment, Vector
import MMTK.PIIntegratorSupport
cimport MMTK.PIIntegratorSupport
import numbers

from MMTK.forcefield cimport energy_data
cimport MMTK.mtrand

include 'MMTK/trajectory.pxi'

cdef extern from "fftw3.h":
    ctypedef struct fftw_complex
    ctypedef void *fftw_plan
    cdef int FFTW_FORWARD, FFTW_BACKWARD, FFTW_ESTIMATE
    cdef void fftw_execute(fftw_plan p)
    cdef fftw_plan fftw_plan_dft_1d(int n, fftw_complex *data_in, fftw_complex *data_out,
                                    int sign, int flags)
    cdef void fftw_destroy_plan(fftw_plan p)

cdef extern from "stdlib.h":
    cdef double fabs(double)
    cdef double sqrt(double)
    cdef double sin(double)
    cdef double cos(double)
    cdef double exp(double)
    cdef double M_PI

cdef extern from "time.h":
    ctypedef unsigned long clock_t
    cdef clock_t clock()
    cdef enum:
        CLOCKS_PER_SEC

cdef double hbar = Units.hbar
cdef double k_B = Units.k_B

cdef bytes PLAN_CAPSULE_NAME = b'plan_capsule'

cdef void plan_capsule_destructor(object cap):
    fftw_destroy_plan(PyCapsule_GetPointer(cap, PLAN_CAPSULE_NAME))

#
# Velocity Verlet integrator in normal-mode coordinates
#
cdef class Rot2DOnly_PINormalModeIntegrator(MMTK.PIIntegratorSupport.PIIntegrator):

    """
    Molecular dynamics integrator for path integral systems using
    normal-mode coordinates.

    The integration is started by calling the integrator object.
    All the keyword options (see documentation of __init__) can be
    specified either when creating the integrator or when calling it.

    The following data categories and variables are available for
    output:

     - category "time": time

     - category "configuration": configuration and box size (for
       periodic universes)

     - category "velocities": atomic velocities

     - category "gradients": energy gradients for each atom

     - category "energy": potential and kinetic energy, plus
       extended-system energy terms if a thermostat and/or barostat
       are used

     - category "thermodynamic": temperature

     - category "auxiliary": primitive and virial quantum energy estimators

    """

    cdef dict plans
    cdef N.ndarray densmat, rotengmat
    cdef double rotmove #dipole,rotmove

    def __init__(self, universe, **options):
        """
        :param universe: the universe on which the integrator acts
        :type universe: MMTK.Universe
        :keyword steps: the number of integration steps (default is 100)
        :type steps: int
        :keyword delta_t: the time step (default is 1 fs)
        :type delta_t: float
        :keyword actions: a list of actions to be executed periodically
                          (default is none)
        :type actions: list
        :keyword threads: the number of threads to use in energy evaluation
                          (default set by MMTK_ENERGY_THREADS)
        :type threads: int
        :keyword background: if True, the integration is executed as a
                             separate thread (default: False)
        :type background: bool
        """
        MMTK.PIIntegratorSupport.PIIntegrator.__init__(
            self, universe, options, "Path integral normal-mode integrator")
        # Supported features: PathIntegrals
        self.features = [Features.PathIntegralsFeature]

    default_options = {'first_step': 0, 'steps': 100, 'delta_t': 1.*Units.fs,
                       'background': False, 'threads': None,
                       'frozen_subspace': None, 'actions': []}

    available_data = ['time', 'configuration', 'velocities', 'gradients',
                      'energy', 'thermodynamic', 'auxiliary']

    restart_data = ['configuration', 'velocities', 'energy']

    # The implementation of the equations of motion follows the article
    #   Ceriotti et al., J. Chem. Phys. 133, 124104 (2010)
    # with the following differences:
    # 1) All the normal mode coordinates are larger by a factor sqrt(nbeads),
    #    and the non-real ones (k != 0, k != n/2) are additionally smaller by
    #    sqrt(2).
    # 2) The spring energy is smaller by a factor of nbeads to take
    #    into account the factor nbeads in Eq. (3) of the paper cited above.
    #    The potential energy of the system is also smaller by a factor of
    #    nbeads compared to the notation in this paper.
    # 3) Velocities are used instead of momenta in the integrator.
    # 4) Eq. (18) is also used for odd n, ignoring the k = n/2 case.

    cdef void atomtocm(self, N.ndarray[double, ndim=2] x, N.ndarray[double, ndim=2] v,
                       N.ndarray[double, ndim=2] g, N.ndarray[double, ndim=1] m,
                       N.ndarray[double, ndim=2] xcm, N.ndarray[double, ndim=2] vcm,
                       N.ndarray[double, ndim=2] gcm, N.ndarray[double, ndim=1] mcm,
                       N.ndarray[N.int32_t, ndim=2] bdcm, int Nmol):

         cdef int tot_atoms,i,j,k,z,natomspmol,nbeadspmol, atom_index
         tot_atoms=0
         for i in range (Nmol):
            natomspmol=self.universe.objectList()[i].numberOfAtoms()
            # nbeadspmol is the number of beads we want our molecule COM to have. 
            # Therefore is the number of beads each atom has in the molecule.
            nbeadspmol=self.universe.objectList()[i].numberOfPoints()/natomspmol          
            
            for z in range (nbeadspmol):
                bdcm[i*nbeadspmol+z,0]=N.int32(z)
                if bdcm[i*nbeadspmol+z,0] == N.int32(0):
                    bdcm[i*nbeadspmol+z,1]=N.int32(nbeadspmol)
                mcm[i*nbeadspmol+z]=self.universe.objectList()[i].mass()/nbeadspmol
                
                
                for k in range(3):
                    xcm[i*nbeadspmol+z,k]=0.0
                    vcm[i*nbeadspmol+z,k]=0.0
                    gcm[i*nbeadspmol+z,k]=0.0
                    for j in range(natomspmol):
                        atom_index=tot_atoms+j
                        xcm[i*nbeadspmol+z,k]+=m[atom_index*nbeadspmol+z]*x[atom_index*nbeadspmol+z,k]/mcm[i*nbeadspmol+z]
                        vcm[i*nbeadspmol+z,k]+=m[atom_index*nbeadspmol+z]*v[atom_index*nbeadspmol+z,k]/mcm[i*nbeadspmol+z]
                        gcm[i*nbeadspmol+z,k]+=g[atom_index*nbeadspmol+z,k]

            tot_atoms+=natomspmol


    cdef void cmtoatom(self, N.ndarray[double, ndim=2] x, N.ndarray[double, ndim=2] v,
                       N.ndarray[double, ndim=2] g, N.ndarray[double, ndim=1] m,
                       N.ndarray[double, ndim=2] xcm, N.ndarray[double, ndim=2] vcm,
                       N.ndarray[double, ndim=2] gcm, N.ndarray[double, ndim=1] mcm,
                       int Nmol):

         #xcom is ORIGINAL center of mass!
         cdef N.ndarray[double,ndim=1] xcom
         cdef int tot_atoms,i,j,k,z,natomspmol,nbeadspmol, atom_index

         xcom=N.zeros((3,),N.float)

         tot_atoms=0

         for i in range (Nmol):
            natomspmol=self.universe.objectList()[i].numberOfAtoms()
            nbeadspmol=self.universe.objectList()[i].numberOfPoints()/natomspmol
            for z in range (nbeadspmol):
                for k in range(3):
                    xcom[k]=0.
                    for j in range(natomspmol):
                        atom_index=tot_atoms+j
                        xcom[k]+=m[atom_index*nbeadspmol+z]*x[atom_index*nbeadspmol+z,k]/mcm[i*nbeadspmol+z]
                for k in range(3):
                    for j in range(natomspmol):
                        atom_index=tot_atoms+j
                        x[atom_index*nbeadspmol+z,k]=x[atom_index*nbeadspmol+z,k]-xcom[k]+xcm[i*nbeadspmol+z,k]
                        g[atom_index*nbeadspmol+z,k]=gcm[i*nbeadspmol+z,k]*m[atom_index*nbeadspmol+z]/mcm[i*nbeadspmol+z]
                        v[atom_index*nbeadspmol+z,k]=vcm[i*nbeadspmol+z,k]

            tot_atoms+=natomspmol

    cdef void eulertocart(self, int bindex, int molind, int P, N.ndarray[double, ndim=2] x, N.ndarray[double, ndim=1] m, N.ndarray[double, ndim=2] xcm, N.ndarray[double, ndim=1] eulerangles, N.ndarray[double, ndim=1] bondlength):
        natomspmol=self.universe.objectList()[molind].numberOfAtoms()
        nbeadspmol=self.universe.objectList()[molind].numberOfPoints()/natomspmol
        aindex = molind*natomspmol
        a1index = (aindex+0)*nbeadspmol+bindex
        a2index = (aindex+1)*nbeadspmol+bindex
        pos1 = x[a1index]
        pos2 = x[a2index]
        mass1 = m[a1index]
        mass2 = m[a2index]
        v1 = bondlength[molind]*Vector(eulerangles[0],eulerangles[1],eulerangles[2])
        for i in range(3):
            x[a1index,i]=xcm[molind*nbeadspmol+bindex,i]+mass2*v1[i]/(mass1+mass2)
            x[a2index,i]=xcm[molind*nbeadspmol+bindex,i]-mass1*v1[i]/(mass1+mass2)

    def within(self, double value):
        if (value > 1.0):
            value = 1.0
        elif (value < -1.0):
            value = -1.0
        return value

    def energyCalculator(self,x):
        cdef energy_data energytemp
        energytemp.gradients = NULL
        energytemp.gradient_fn = NULL
        energytemp.force_constants = NULL
        energytemp.fc_fn = NULL
        self.calculateEnergies(x, &energytemp, 0)
        return energytemp.energy

    cdef start(self):
        cdef double acceptratio, rd, sint, pot_old, pot_new, dens_old, dens_new, indexp0val, indexp1val
        cdef int t0b, t1b, t2b, t0, t1, t2, atombead, indexp0, indexp1, indexp0n, indexp1n
        cdef N.ndarray[double, ndim=2] x, v, g, dv, nmc, nmv, xcm, vcm, gcm
        cdef N.ndarray[double, ndim=1] m, mcm
        cdef N.ndarray[double, ndim=1] bondlength
        cdef N.ndarray[N.int32_t, ndim=2] bd, bdcm
        cdef N.ndarray[double, ndim=3] ss
        cdef energy_data energy
        cdef double time, delta_t, ke, ke_nm, se, beta, temperature
        cdef double qe_prim, qe_vir, qe_cvir, qe_rot
        cdef int natoms, nbeads, nsteps, step, df, cdf, nb, Nmol
        cdef Py_ssize_t i, j, k

        cdef double propct, propphi
        cdef int P
        cdef N.ndarray[double, ndim=1] costheta,phi
        cdef N.ndarray[double, ndim=2] MCCosine
        cdef N.ndarray[double, ndim=1] MCCosprop
        cdef N.ndarray[double, ndim=2] xold
        cdef N.ndarray[double, ndim=1] densitymatrix, rotenergy
        cdef double rotstep, ndens, Ntruemol
        densitymatrix=self.densmat
        ndens=1.0*len(densitymatrix)
        rotenergy=self.rotengmat
        rotstep=self.rotmove
        #xold=N.zeros((2,3),float)

        # Check if velocities have been initialized
        if self.universe.velocities() is None:
            raise ValueError("no velocities")

        # Gather state variables and parameters
        configuration = self.universe.configuration()
        velocities = self.universe.velocities()
        gradients = ParticleProperties.ParticleVector(self.universe)
        masses = self.universe.masses()
        delta_t = self.getOption('delta_t')
        nsteps = self.getOption('steps')
        natoms = self.universe.numberOfAtoms()
        nbeads = self.universe.numberOfPoints()
        bd = self.evaluator_object.global_data.get('bead_data')
        pi_environment = self.universe.environmentObjectList(Environment.PathIntegrals)[0]
        beta = pi_environment.beta
        #print beta

        # For efficiency, the Cython code works at the array
        # level rather than at the ParticleProperty level.
        x = configuration.array
        v = velocities.array
        g = gradients.array
        m = masses.array

        # MATT-Introduce X-COM variable, number of molecules Nmol
        acceptratio=0.0
        P=nbeads/natoms
        Nmol = len(self.universe.objectList())
        #print Nmol
        nbeads_mol = N.int32(P*Nmol)
        xcm = N.zeros((nbeads_mol, 3), N.float)
        vcm = N.zeros((nbeads_mol, 3), N.float)
        gcm = N.zeros((nbeads_mol, 3), N.float)
        mcm = N.zeros(nbeads_mol, N.float)
        dv = N.zeros((nbeads_mol, 3), N.float)
        nmc = N.zeros((3, nbeads_mol), N.float)
        nmv = N.zeros((3, nbeads_mol), N.float)
        bdcm = N.zeros((nbeads_mol,2), N.int32)
        bondlength=N.zeros((Nmol),N.float)

        subspace = self.getOption('frozen_subspace')
        if subspace is None:
            ss = N.zeros((0, nbeads, 3), N.float)
        else:
            ss = subspace.getBasis().array

        #ROTATIONAL VARIABLES
        costheta = N.zeros((nbeads_mol), N.float)
        phi = N.zeros((nbeads_mol), N.float)
        MCCosine = N.zeros((nbeads_mol,3), N.float)
        MCCosprop=N.zeros((3), N.float)
        
        ##########################################
        ### CALCULATE ANGLES AND FILL MCCosine ###
        ##########################################
        for i in range(Nmol):
            bondlength[i]=(self.universe.atomList()[2*i+1].beadPositions()[0]-self.universe.atomList()[2*i].beadPositions()[0]).length()

        for k in range(Nmol):
            for i in range(P):
                rel=self.universe.atomList()[2*k+1].beadPositions()[i]-self.universe.atomList()[2*k].beadPositions()[i]
                unirel=rel/rel.length()
                costheta[k*P+i]=unirel[2]
                costheta[k*P+i]=self.within(costheta[k*P+i])
                phi[k*P+i]=N.arctan2(unirel[1],unirel[0])

                sint=sqrt(1.0-costheta[k*P+i]*costheta[k*P+i])
                MCCosine[k*P+i][0]=sint*N.cos(phi[k*P+i])
                MCCosine[k*P+i][1]=sint*N.sin(phi[k*P+i])
                MCCosine[k*P+i][2]=costheta[k*P+i]

        # Initialize the plan cache.
        self.plans = {}

        # Ask for energy gradients to be calculated and stored in
        # the array g. Force constants are not requested.
        energy.gradients = <void *>g
        energy.gradient_fn = NULL
        energy.force_constants = NULL
        energy.fc_fn = NULL

        # Declare the variables accessible to trajectory actions.
        self.declareTrajectoryVariable_double(
            &time, "time", "Time: %lf\n", time_unit_name, PyTrajectory_Time)
        self.declareTrajectoryVariable_array(
            v, "velocities", "Velocities:\n", velocity_unit_name,
            PyTrajectory_Velocities)
        self.declareTrajectoryVariable_array(
            g, "gradients", "Energy gradients:\n", energy_gradient_unit_name,
            PyTrajectory_Gradients)
        self.declareTrajectoryVariable_double(
            &energy.energy,"potential_energy", "Potential energy: %lf\n",
            energy_unit_name, PyTrajectory_Energy)
        self.declareTrajectoryVariable_double(
            &ke, "kinetic_energy", "Kinetic energy: %lf\n",
            energy_unit_name, PyTrajectory_Energy)
        self.declareTrajectoryVariable_double(
            &se, "spring_energy", "Spring energy: %lf\n",
            energy_unit_name, PyTrajectory_Energy)
        self.declareTrajectoryVariable_double(
            &temperature, "temperature", "Temperature: %lf\n",
            temperature_unit_name, PyTrajectory_Thermodynamic)
        self.declareTrajectoryVariable_double(
            &qe_prim, "quantum_energy_primitive",
            "Primitive quantum energy estimator: %lf\n",
            energy_unit_name, PyTrajectory_Auxiliary)
        self.declareTrajectoryVariable_double(
            &qe_vir, "quantum_energy_virial",
            "Virial quantum energy estimator: %lf\n",
            energy_unit_name, PyTrajectory_Auxiliary)
        self.declareTrajectoryVariable_double(
            &qe_cvir, "quantum_energy_centroid_virial",
            "Centroid virial quantum energy estimator: %lf\n",
            energy_unit_name, PyTrajectory_Auxiliary)
        self.declareTrajectoryVariable_double(
            &qe_rot, "quantum_energy_rotation",
            "Rotation quantum energy estimator: %lf\n",
            energy_unit_name, PyTrajectory_Auxiliary)
        self.initializeTrajectoryActions()

        # Acquire the write lock of the universe. This is necessary to
        # make sure that the integrator's modifications to positions
        # and velocities are synchronized with other threads that
        # attempt to use or modify these same values.
        #
        # Note that the write lock will be released temporarily
        # for trajectory actions. It will also be converted to
        # a read lock temporarily for energy evaluation. This
        # is taken care of automatically by the respective methods
        # of class EnergyBasedTrajectoryGenerator.
        self.acquireWriteLock()
        # Preparation: Calculate initial half-step accelerations
        # and run the trajectory actions on the initial state.
        self.foldCoordinatesIntoBox()

        Ntruemol=0
        for i in range(Nmol):
            print i, self.universe.objectList()[i].numberOfAtoms()
            if (self.universe.objectList()[i].numberOfAtoms()>1):
                Ntruemol+=1
        print Ntruemol

        # Main integration loop
        time = 0.
        self.trajectoryActions(0)
        for step in range(nsteps):
            self.atomtocm(x,v,g,m,xcm,vcm,gcm,mcm,bdcm,Nmol)
            self.cmtoatom(x,v,g,m,xcm,vcm,gcm,mcm,Nmol)
            #######################################
            ### PERFORM MC RIGID BODY ROTATIONS ###
            #######################################
            for a in range(Nmol):
                for t1b in range(P):
                    natomspmol=self.universe.objectList()[a].numberOfAtoms()

                    #if (natomspmol==1):
                    #    atomcount+=natomspmol
                    #    continue

                    t0b=t1b-1
                    t2b=t1b+1

                    if (t0b<0): t0b+=P
                    if (t2b>(P-1)): t2b-=P

                    t0=a*P+t0b
                    t1=a*P+t1b
                    t2=a*P+t2b

                    xold=N.zeros((natomspmol,3),N.float)
                    for i in range(natomspmol):
                        for j in range(3):
                            xold[i,j]=x[(a*natomspmol+i)*P,j]

                    propct=costheta[t1]+rotstep*(N.random.random()-0.5)
                    propphi=phi[t1]+rotstep*(N.random.random()-0.5)

                    if (propct > 1.0):
                        propct=2.0-propct
                    elif (propct < -1.0):
                        propct=-2.0-propct

                    propsint=sqrt(1.0-propct*propct)

                    MCCosprop[0]=propsint*N.cos(propphi)
                    MCCosprop[1]=propsint*N.sin(propphi)
                    MCCosprop[2]=propct

                    pot_old=self.energyCalculator(N.asarray(x))

                    p0=0.0
                    p1=0.0
                    for co in range(3):
                        p0+=MCCosine[t0][co]*MCCosine[t1][co]
                        p1+=MCCosine[t1][co]*MCCosine[t2][co]

                    indexp0=int(N.floor((p0+1.0)*(ndens-1.0)/2.0))
                    indexp1=int(N.floor((p1+1.0)*(ndens-1.0)/2.0))

                    indexp0n=indexp0+1
                    indexp1n=indexp1+1
                    if (indexp0==ndens-1):
                        indexp0n=indexp0
                    if (indexp1==ndens-1):
                        indexp1n=indexp1

                    indexp0val=-1.0+indexp0*2.0/(ndens-1.0)
                    indexp1val=-1.0+indexp1*2.0/(ndens-1.0)

                    dens_old=(densitymatrix[indexp0]+(densitymatrix[indexp0n]-densitymatrix[indexp0])*(p0-indexp0val)/(2.0/(ndens-1.0)))*(densitymatrix[indexp1]+(densitymatrix[indexp1n]-densitymatrix[indexp1])*(p1-indexp1val)/(2.0/(ndens-1.0)))

                    if (fabs(dens_old)<(1.0e-10)):
                        dens_old=0.0
                    if (dens_old < 0.0):
                        print "Rotational Density Negative"
                        raise()

                    ##NEW DENSITY
                    p0=0.0
                    p1=0.0
                    for co in range(3):
                        p0+=MCCosine[t0][co]*MCCosprop[co]
                        p1+=MCCosprop[co]*MCCosine[t2][co]

                    indexp0=int(N.floor((p0+1.0)*(ndens-1.0)/2.0))
                    indexp1=int(N.floor((p1+1.0)*(ndens-1.0)/2.0))

                    indexp0n=indexp0+1
                    indexp1n=indexp1+1
                    if (indexp0==ndens-1):
                        indexp0n=indexp0
                    if (indexp1==ndens-1):
                        indexp1n=indexp1

                    indexp0val=-1.0+indexp0*2.0/(ndens-1.0)
                    indexp1val=-1.0+indexp1*2.0/(ndens-1.0)

                    dens_new=(densitymatrix[indexp0]+(densitymatrix[indexp0n]-densitymatrix[indexp0])*(p0-indexp0val)/(2.0/(ndens-1.0)))*(densitymatrix[indexp1]+(densitymatrix[indexp1n]-densitymatrix[indexp1])*(p1-indexp1val)/(2.0/(ndens-1.0)))

                    if (fabs(dens_new)<(1.0e-10)):
                        dens_new=0.0
                    if (dens_new < 0.0):
                        print "Rotational Density Negative"
                        raise()

                    self.eulertocart(t1b, a, P, x, m, xcm, MCCosprop, bondlength)
                    pot_new=self.energyCalculator(N.asarray(x))

                    rd=1.0
                    if (dens_old>(1.0e-10)):
                        rd=dens_new/dens_old
                    rd*= exp(-(beta)*(pot_new-pot_old))

                    accept=False
                    if (rd>1.0):
                        accept=True
                    elif (rd>N.random.random()):
                        accept=True
                    
                    if (accept):
                        pot_old=pot_new
                        acceptratio+=1.0
                        costheta[t1]=propct
                        phi[t1]=propphi
                        for co in range(3):
                            MCCosine[t1][co]=MCCosprop[co]
                    else:
                        for i in range(natomspmol):
                            for j in range(3):
                                x[(a*natomspmol+i)*P+t1b,j]=xold[i,j]

            qe_prim=self.energyCalculator(N.asarray(x))

            qe_rot=0.0
            for a in range(Nmol):
                for t1b in range(P):
                    #t0b=t1b-1
                    #if (t0b<0): t0b+=P
                    t2b=t1b+1
                    if (t2b>(P-1)): t2b-=P
                    #t0=a*P+t0b
                    t1=a*P+t1b
                    t2=a*P+t2b

                    p0=0.0
                    for co in range(3):
                        #p0+=MCCosine[t0][co]*MCCosine[t1][co]
                        p0+=MCCosine[t1][co]*MCCosine[t2][co]
                    indexp0=int(N.floor((p0+1.0)*(ndens-1.0)/2.0))

                    indexp0n=indexp0+1
                    if (indexp0==ndens-1):
                        indexp0n=indexp0

                    indexp0val=-1.0+indexp0*2.0/(ndens-1.0)

                    qe_rot+=rotenergy[indexp0]+(rotenergy[indexp0n]-rotenergy[indexp0])*(p0-indexp0val)/(2.0/(ndens-1.0))

            # End of time step
            time += delta_t
            self.foldCoordinatesIntoBox()
            self.trajectoryActions(step+1)

        # Release the write lock.
        self.releaseWriteLock()

        # Calculate the Acceptance Ratio for MC step
        acceptratio/=Ntruemol*P*nsteps
        print "Acceptance Ratio: ", acceptratio

        # Finalize all trajectory actions (close files etc.)
        self.finalizeTrajectoryActions(nsteps)

#
# Velocity Verlet integrator in normal-mode coordinates
# with a Langevin thermostat
#
cdef class Rot2DOnly_PILangevinNormalModeIntegrator(Rot2DOnly_PINormalModeIntegrator):

    """
    Molecular dynamics integrator for path integral systems using
    normal-mode coordinates and a Langevin thermostat.

    This integrator works like PINormalModeIntegrator, but has
    an additional option "centroid_friction", which is a ParticleScalar
    (one friction constant per atom) or a plain number.

    """

    cdef N.ndarray gamma

    cdef start(self):
        friction = self.getOption('centroid_friction')
        #self.dipole=self.getOption('dipole')
        self.densmat=self.getOption('densmat')
        self.rotengmat=self.getOption('rotengmat')
        self.rotmove=self.getOption('rotstep')
        if isinstance(friction, ParticleProperties.ParticleScalar):
            self.gamma = friction.array
        else:
            assert isinstance(friction, numbers.Number)
            nbeads = self.universe.numberOfPoints()
            self.gamma = N.zeros((nbeads,), N.float)+friction
        Rot2DOnly_PINormalModeIntegrator.start(self)







