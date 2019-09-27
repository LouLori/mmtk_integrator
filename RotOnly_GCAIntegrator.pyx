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
from libc.stdint cimport int64_t
import numpy as N
cimport numpy as N

from MMTK import Units, ParticleProperties, Features, Environment, Vector
from MMTK.ForceFields.ForceField import CompoundForceField
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
cdef class RotOnlyGCA_PINormalModeIntegrator(MMTK.PIIntegratorSupport.PIIntegrator):

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
    cdef double rotmove
    cdef int rotstepskip

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

    cdef void eulertocart(self, int bindex, int molnum, int P, N.ndarray[double, ndim=2] x, N.ndarray[double, ndim=1] m, N.ndarray[double, ndim=1] eulerangles, N.ndarray[double, ndim=1] bondlength):
        natomspmol=self.universe.objectList()[molnum].numberOfAtoms()
        aindex = molnum*natomspmol
        a1index=(aindex+0)*P+bindex
        a2index=(aindex+1)*P+bindex
        atom1=x[a1index]
        mass1=m[a1index]
        atom2=x[a2index]
        mass2=m[a2index]
        com=(atom1*mass1+atom2*mass2)/(mass1+mass2)
        rel=Vector((atom1-atom2))
        v1=bondlength[molnum]*Vector(eulerangles[0],eulerangles[1],eulerangles[2])
        #print "BEFORE REFLECT2: ", x[a1index], x[a2index]
        for i in range(3):
            x[a1index,i]=com[i]+mass2*v1[i]/(mass1+mass2)
            x[a2index,i]=com[i]-mass1*v1[i]/(mass1+mass2)
        #print "AFTER REFLECT2: ", x[a1index], x[a2index]

    def energyCalculator(self, x):
        cdef energy_data energytemp
        energytemp.gradients = NULL
        energytemp.gradient_fn = NULL
        energytemp.force_constants = NULL
        energytemp.fc_fn = NULL
        self.calculateEnergies(x, &energytemp, 0)
        return energytemp.energy

## Lori adds the potCalculator for energy interaction between the two particles in bead level
    def isopotCalculator(self, x, int Nmol, int Mol1, int Mol2, int P, int bindex):
        #cdef energy_data energytemp
        cdef double potResults, massFpercent, massHpercent, e
        cdef double dipole, bondlength, q, massF, massH, natomspmol1, natomspmol2
        cdef N.ndarray[double, ndim=1] x1,x2,x3,x4,mu1, mu2, cmvec
        #energytemp.gradients = NULL
        #energytemp.gradient_fn = NULL
        #energytemp.force_constants = NULL
        #energytemp.fc_fn = NULL
        #self.calculateEnergies(x, &energytemp, 0)

        massF = 18.998403
        massH = 1.00794
        dipole = 2.0
        bondlength = 0.09255215
        q = (dipole/(bondlength*1.0e-9))*3.33564e-30

        natomspmol1=self.universe.objectList()[Mol1].numberOfAtoms()
        natomspmol2=self.universe.objectList()[Mol2].numberOfAtoms()
        x1=x[(Mol1*natomspmol1+0)*P+bindex]
        x2=x[(Mol1*natomspmol1+1)*P+bindex]
        x3=x[(Mol2*natomspmol2+0)*P+bindex]
        x4=x[(Mol2*natomspmol2+1)*P+bindex]
        mu1=x2-x1
        mu2=x4-x3

        massFpercent=massF/(massF+massH)
        massHpercent=massH/(massF+massH)

        cmvec = (massFpercent*(x3-x1)+massHpercent*(x4-x2))
        r = N.linalg.norm(cmvec)

        e = (q*q/(4.0*N.pi*r*r*r*8.854187817e-12))*(1.0e9*6.022140857e23)/1000.0*(N.dot(mu1,mu2))
        # energy of Viso
        return e

    def extpotCalculator(self, x, int Nmol, int Mol1, int Mol2, int P, int bindex):
        #cdef energy_data energytemp
        cdef double potResults, massFpercent, massHpercent, e
        cdef double dipole, bondlength, q, massF, massH, natomspmol1, natomspmol2
        cdef N.ndarray[double, ndim=1] x1,x2,x3,x4,mu1, mu2, cmvec
        #energytemp.gradients = NULL
        #energytemp.gradient_fn = NULL
        #energytemp.force_constants = NULL
        #energytemp.fc_fn = NULL
        #self.calculateEnergies(x, &energytemp, 0)

        massF = 18.998403
        massH = 1.00794
        dipole = 2.0
        bondlength = 0.09255215
        q = (dipole/(bondlength*1.0e-9))*3.33564e-30

        natomspmol1=self.universe.objectList()[Mol1].numberOfAtoms()
        natomspmol2=self.universe.objectList()[Mol2].numberOfAtoms()
        x1=x[(Mol1*natomspmol1+0)*P+bindex]
        x2=x[(Mol1*natomspmol1+1)*P+bindex]
        x3=x[(Mol2*natomspmol2+0)*P+bindex]
        x4=x[(Mol2*natomspmol2+1)*P+bindex]
        mu1=x2-x1
        mu2=x4-x3

        massFpercent=massF/(massF+massH)
        massHpercent=massH/(massF+massH)

        cmvec = (massFpercent*(x3-x1)+massHpercent*(x4-x2))
        r = N.linalg.norm(cmvec)

        potResults = (q*q/(4.0*N.pi*r*8.854187817e-12))*(1.0e9*6.022140857e23)/1000.0*(-3.0*N.dot(mu1,cmvec)*N.dot(mu2,cmvec))
        ## Sort Mol1 and Mol2, let Mol1 < Mol2
        #if Mol1 > Mol2:
        #    t = Mol1
        #    Mol1 = Mol2
        #    Mol2 = t
        ### Find the index of the interaction between Mol1 and Mol2
        #offset = 0
        #if Mol1 == 0:
        #    offset = 0
        #else:
        #    for i in range(0,Mol1):
        #        offset += (Nmol-i-1)
        #pot_index = (Mol2-Mol1-1+offset)*P+bindex
        ##print "pot_index", Mol1, Mol2, pot_index
        #potResults = energytemp.energy_terms[pot_index]
        # energy of Vext = Vtot-Viso
        return potResults

    def Reflect(self, MCCoord, RefVect, Delthe):#Delcth, Delphi):#,RefVect):
        cdef double phi, cth, sth
        cdef N.ndarray[double, ndim=1] RefMCCoord
        RefMCCoord = MCCoord*N.cos(Delthe)+N.cross(RefVect,MCCoord)*N.sin(Delthe)+RefVect*(N.dot(RefVect,MCCoord))*(1.-N.cos(Delthe))

        return RefMCCoord

    def ClusterGrowth(self, x, m, beta, MCCosine, Nmol, Mol1, Mol2, P, bindex, bondlength, randVector, Delthe):
        cdef N.ndarray[double, ndim=1] reflectMol2
        cdef double pot_n1n2, pot_n1o2
        cdef double factor, linkProb
        cdef int link_active
        ## When calculate the pot_new, the Mol1 has been reflected, and Mol2 hasn't, which is V(ri'-rj)
        ## **Mol1's x is changed
        pot_n1o2 = self.isopotCalculator(x,Nmol,Mol1,Mol2,P,bindex)
        ## When calculte the pot_old, both Mol1 and Mol2 have been reflected, which is V(ri'-rj')=V(ri-rj)
        ## **Mol2's x is changed
        ## Temporarily reflect Mol2
        reflectMol2 = self.Reflect(MCCosine[Mol2*P+bindex], randVector, Delthe)#Delcth, Delphi)
        # change the x corrdinates for mol2
        self.eulertocart(bindex, Mol2, P, x, m, reflectMol2, bondlength)
        pot_n1n2 = self.isopotCalculator(x,Nmol,Mol1,Mol2,P,bindex)
        #exponent = exp(-(beta)*(pot_new-pot_old))
        exponent = exp(-(beta)*(pot_n1o2-pot_n1n2))
        #exponent = exp(-(beta)*(pot_new-pot_old))
        #link_active = pot_new-pot_old
        link_active = 0
        ## p_ij = max[1-exp(-beta*delta_ij),0] ==>
        linkProb = N.maximum((1-exponent), 0)
        if linkProb > N.random.random(): link_active = 1
        return link_active

    def within(self, double value):
        if (value > 1.0):
            value = 1.0
        elif (value < -1.0):
            value = -1.0
        return value

    cdef start(self):
        cdef double acceptratio, rd, MoveProb, sint, pot_old, pot_new, pot_diff, pot_old_0, pot_new_0, pot_old_c, pot_new_c, dens_old_ant, dens_new_ant, dens_old_clu, dens_new_clu, dens_old_a, dens_new_a, dens_old_c,dens_new_c,indexp0val, indexp1val
        cdef int t0b, t1b, t2b, t0, t1, t2, atombead, indexp0, indexp1, indexp0n, indexp1n
        cdef N.ndarray[double, ndim=2] x, v, g, dv, nmc, nmv, xcm, vcm, gcm
        cdef N.ndarray[double, ndim=1] m, mcm
        cdef N.ndarray[double, ndim=1] bondlength,x1,x2
        cdef N.ndarray[N.int32_t, ndim=2] bd, bdcm
        cdef N.ndarray[double, ndim=3] ss
        cdef energy_data energy
        cdef double time, delta_t, ke, ke_nm, se, beta, temperature
        cdef double qe_prim, qe_vir, qe_cvir, qe_rot
        cdef int natoms, nbeads, nsteps, step, df, cdf, nb, Nmol, Ntruemol,rotbdcount,rotbdskip
        cdef Py_ssize_t i, j, k

        cdef double propct, propphi, dcth, delthe
        cdef int P
        cdef N.ndarray[double, ndim=1] costheta, phi
        cdef N.ndarray[double, ndim=2] MCCosine, newcoords
        cdef N.ndarray[double, ndim=1] MCCosprop
        cdef N.ndarray[double, ndim=2] xtemp,xold
        cdef N.ndarray[double, ndim=1] densitymatrix, rotenergy
        cdef double rotstep, ndens
        #cdef int rotskipstep, nrotsteps
        cdef double randVcth, randVphi, randVsin, length_c, RandX
        cdef N.ndarray[double, ndim=1] randVector
        cdef N.ndarray[N.int64_t, ndim=1] clu, buf, anticlu, linkclu
        densitymatrix=self.densmat
        ndens=1.0*len(densitymatrix)
        rotenergy=self.rotengmat
        rotstep=self.rotmove
        #rotskipstep=self.rotstepskip

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

        # For efficiency, the Cython code works at the array
        # level rather than at the ParticleProperty level.
        x = configuration.array
        v = velocities.array
        g = gradients.array
        m = masses.array

	# MATT-Introduce X-COM variable, number of molecules Nmol
    #########################################################
    # nbeads: number of total beads - atom*P+bindex
    # nbeads_mol: number of molecular beads - mol*P+bindex
        acceptratio=0.0
        P=nbeads/natoms
        Nmol = len(self.universe.objectList())
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
        xtemp = N.zeros((nbeads,3),float)


        subspace = self.getOption('frozen_subspace')
        if subspace is None:
            ss = N.zeros((0, nbeads, 3), N.float)
        else:
            ss = subspace.getBasis().array

        #ROTATIONAL VARIABLES
        #nrotsteps=0
        costheta = N.zeros(nbeads_mol, N.float)
        phi = N.zeros(nbeads_mol, N.float)
        MCCosine = N.zeros((nbeads_mol,3), N.float)
        MCCosprop = N.zeros((3), N.float)
        newcoords = N.zeros((nbeads_mol,3), N.float)
        randVector = N.zeros(3, N.float)

        #CLUSTER VARIABLES
        clu = N.array([],N.int64)
        buf = N.array([],N.int64)
        linkclu = N.array([],N.int64)
        anticlu = N.array([],N.int64)
        ##########################################
        ### CALCULATE ANGLES AND FILL MCCosine ###
        ##########################################
        # Filling the bondlength
        for i in range(Nmol):
            bondlength[i]=(self.universe.atomList()[2*i+1].beadPositions()[0]-self.universe.atomList()[2*i].beadPositions()[0]).length()
        for k in range(Nmol):
            for i in range(P):
                natomspmol=self.universe.objectList()[k].numberOfAtoms()
                rel=self.universe.atomList()[2*k+1].beadPositions()[i]-self.universe.atomList()[2*k].beadPositions()[i]
                unirel=rel/rel.length()
                costheta[k*P+i]=unirel[2]
                costheta[k*P+i]=self.within(costheta[k*P+i])
                phi[k*P+i]=N.arctan2(unirel[1],unirel[0])
                if phi[k*P+i] < 0.0:
                    phi[k*P+i]+=2.0*N.pi
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
            #print "Step:",step, "Before MCCosine:",MCCosine
            #######################################
            ### PERFORM MC RIGID BODY ROTATIONS ###
            #######################################
            #######################################
            ########## Wolff's Algorithm ##########
            #######################################
            #print "BEGIN WOLFF ROTATION FOR STEP ", step
            ## Loop over number of beads, middle one, t1b
            for t1b in range(P):
                #store x in xtemp every step
                xtemp=N.asarray(x)

                atomcount=0
                ## Create the anticluster which contains the elements not in the cluster
                anticlu = N.arange(Nmol)
                ## Calculate the old potential energy
                pot_old=self.energyCalculator(N.asarray(x))
                #########################################
                ## Create random molecule index
                randMol = N.random.random_integers(0,Nmol-1)
                ## Create empty cluster and buffer
                clu = N.append(clu,randMol)
                buf = N.append(buf,randMol)
                ## Create random vector for Rodrigues' rotation axis
                randVcth = 2.0*(N.random.random()-0.5)#random number from [-1.0,1.0)
                randVphi = 2.0*N.pi*N.random.random() #random number from [-PI,PI)

                if (randVcth > 1.0):
                    randVcth = 2.0-randVcth
                elif (randVcth < -1.0):
                    randVcth = -2.0-randVcth

                randVsin = sqrt(1.0 - randVcth*randVcth)

                randVector[0] = randVsin*N.cos(randVphi)
                randVector[1] = randVsin*N.sin(randVphi)
                randVector[2] = randVcth

                delthe = rotstep*N.pi*(N.random.random()-0.5)
                ########################################
                ## Reflect the first chosen random molecule based on the hyperplane orthogonal to randVector
                MCCosprop = self.Reflect(MCCosine[randMol*P+t1b], randVector, delthe)#delcth, delphi)#randVector)
                #print "MCCosprop:", MCCosprop
                ## Save the reflected MCCosine in newcoords for Mol1
                newcoords[randMol*P+t1b] = MCCosprop
                ## **Change the randMol Cartesian coordinates based on the newcoords
                self.eulertocart(t1b, randMol, P, x, m, MCCosprop, bondlength)
                ########################################
                ## Grow the cluster
                # anticlu: molecule not in cluster
                anticlu = N.setdiff1d(anticlu,clu)
                while True:
                    ## Save the first Mol in buffer to Mol0
                    Mol0 = buf[0]
                    ## pop off the first Mol in buffer
                    buf = N.delete(buf,0)
                    ## join the nearest neighbor (nnMol) to the linkclu
                    for iM,M in enumerate(anticlu):
                        if (M >= 0 and M < Nmol):
                            Attempt = False
                            if M in clu:
                                Attempt = False
                                break
                            else: Attempt = True
                            if Attempt:
                                link_active = self.ClusterGrowth(x,m,beta,MCCosine,Nmol,Mol0,M,P,t1b,bondlength,randVector,delthe)
                                if link_active == 1:
                                    newcoords[M*P+t1b] = self.Reflect(MCCosine[M*P+t1b],randVector,delthe)
                                    clu = N.append(clu,M)
                                    buf = N.append(buf,M)
                                else:
                                    for co in range(3):
                                        x[(M*2+0)*P+t1b,co] = xtemp[(M*2+0)*P+t1b,co]
                                        x[(M*2+1)*P+t1b,co] = xtemp[(M*2+1)*P+t1b,co]
                    anticlu = N.setdiff1d(anticlu,clu)
                    if (buf.size == 0) or (anticlu.size == 0):
                        break
                clu = N.sort(clu)
                ## After growing the cluster, x coordinates in cluster are all changed, but MCCosine stayed for density calculation
                ########################################
                ## Convert NaN in array to Num
                newcoords = N.nan_to_num(newcoords)
                MCCosine = N.nan_to_num(MCCosine)
                ########################################
                ## Compute the Paccept in cluster
                ########################################
                dens_old_clu = 1.0
                dens_new_clu = 1.0
                pot_old_clu = 0.0
                pot_new_clu = 0.0
                for ic, c in enumerate(clu):
                    t0b=t1b-1
                    t2b=t1b+1

                    if (t0b<0): t0b+=P
                    if (t2b>(P-1)): t2b-=P

                    t0=c*P+t0b
                    t1=c*P+t1b
                    t2=c*P+t2b
                    ###################################
                    ##OLD DENSITY
                    p0=0.0
                    p1=0.0
                    for co in range(3):
                        p0+=MCCosine[t0][co]*MCCosine[t1][co]
                        p1+=MCCosine[t1][co]*MCCosine[t2][co]

                    p0 = N.nan_to_num(p0)
                    p1 = N.nan_to_num(p1)

                    #print "p0floor:", (p0+1.0)*(ndens-1.0)/2.0
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

                    dens_old_c=(densitymatrix[indexp0]+(densitymatrix[indexp0n]-densitymatrix[indexp0])*(p0-indexp0val)/(2.0/(ndens-1.0)))*(densitymatrix[indexp1]+(densitymatrix[indexp1n]-densitymatrix[indexp1])*(p1-indexp1val)/(2.0/(ndens-1.0)))

                    if (fabs(dens_old_c)<(1.0e-10)):
                        dens_old_c=0.0
                    if (dens_old_c < 0.0):
                        print "Rotational Density Negative"
                        raise()
                    ##################################
                    ##NEW DENSITY
                    p0=0.0
                    p1=0.0
                    for co in range(3):
                        p0+=MCCosine[t0][co]*newcoords[t1][co]
                        p1+=newcoords[t1][co]*MCCosine[t2][co]

                    p0 = N.nan_to_num(p0)
                    p1 = N.nan_to_num(p1)

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

                    dens_new_c=(densitymatrix[indexp0]+(densitymatrix[indexp0n]-densitymatrix[indexp0])*(p0-indexp0val)/(2.0/(ndens-1.0)))*(densitymatrix[indexp1]+(densitymatrix[indexp1n]-densitymatrix[indexp1])*(p1-indexp1val)/(2.0/(ndens-1.0)))

                    if (fabs(dens_new_c)<(1.0e-10)):
                        dens_new_c=0.0
                    if (dens_new_c < 0.0):
                        print "Rotational Density Negative"
                        raise()

                    dens_old_clu *= dens_old_c
                    dens_new_clu *= dens_new_c

                    ### Consider about the external field
                    for n in range(Nmol):
                        if n != c:
                            pot_old_clu += self.extpotCalculator(xtemp,Nmol,c,n,P,t1b)
                            pot_new_clu += self.extpotCalculator(x,Nmol,c,n,P,t1b)
                        else: continue

                rd=1.0
                if (dens_old_clu>(1.0e-10)):
                    rd = dens_new_clu/dens_old_clu

                rd*= N.exp(-(beta)*(pot_new_clu-pot_old_clu)/P)

                MoveProb=N.minimum(1.0,rd)
                accept=False
                if MoveProb > N.random.random(): accept=True

                if (accept):
                    acceptratio+=1.0
                    for ic, c in enumerate(clu):
                        for co in range(3):
                            MCCosine[c*P+t1b][co]=newcoords[c*P+t1b][co]
                else:
                    for ic,c in enumerate(clu):
                        natomspmol=self.universe.objectList()[c].numberOfAtoms()
                        for i in range(natomspmol):
                            for j in range(3):
                                x_index = (c*natomspmol+i)*P+t1b
                                x[x_index,j]=xtemp[x_index,j]
                ## Clear all cluster in this bead dimension
                #print "END THIS BEAD STEP FOR ", t1b
                clu = N.array([],N.int64)
                newcoords=N.zeros((nbeads_mol,3),N.float)

            ## Calculate potential energy for this step
            qe_prim=self.energyCalculator(N.asarray(x))

            ## Calculate rotational energy for this step
            qe_rot=0.0
            for a in range(Nmol):
                for t1b in range(P):
                    t2b=t1b+1
                    if (t2b>(P-1)): t2b-=P

                    t1=a*P+t1b
                    t2=a*P+t2b
                    p0=0.0
                    for co in range(3):
                        p0+=MCCosine[t1][co]*MCCosine[t2][co]
                    indexp0=int(N.floor((p0+1.0)*(ndens-1.0)/2.0))

                    indexp0n=indexp0+1
                    if (indexp0==ndens-1):
                        indexp0n=indexp0

                    indexp0val=-1.0+indexp0*2.0/(ndens-1.0)

                    qe_rot+=(rotenergy[indexp0]+(rotenergy[indexp0n]-rotenergy[indexp0])*(p0-indexp0val)/(2.0/(ndens-1.0)))

            # End of time step
            time += delta_t
            self.foldCoordinatesIntoBox()
            self.trajectoryActions(step+1)

        # Release the write lock.
        self.releaseWriteLock()

        # Calculate the Acceptance Ratio for MC step
        acceptratio/=P*nsteps
        #acceptratio/=Ntruemol*float(P*nrotsteps*rotbdcount/rotbdskip)
        print "Acceptance Ratio: ", acceptratio

        # Finalize all trajectory actions (close files etc.)
        self.finalizeTrajectoryActions(nsteps)

#
# Velocity Verlet integrator in normal-mode coordinates
# with a Langevin thermostat
#
cdef class RotOnlyGCA_PILangevinNormalModeIntegrator(RotOnlyGCA_PINormalModeIntegrator):

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
        self.densmat=self.getOption('densmat')
        self.rotengmat=self.getOption('rotengmat')
        self.rotmove=self.getOption('rotstep')
        #self.rotstepskip=self.getOption('rotskipstep')
        if isinstance(friction, ParticleProperties.ParticleScalar):
            self.gamma = friction.array
        else:
            assert isinstance(friction, numbers.Number)
            nbeads = self.universe.numberOfPoints()
            self.gamma = N.zeros((nbeads,), N.float)+friction
        RotOnlyGCA_PINormalModeIntegrator.start(self)

