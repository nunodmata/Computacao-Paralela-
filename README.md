# Parallel Computing 
Work assignments developed in the context of the Parallel Computing course, as part of the Master's degree in Software Engineering at the University of Minho. Bellow there's a brief description of the objective around each phase of the practical assignments.

## Introduction
The program to analyse and optimise is part of a simple molecular dynamics’ simulation code applied to atoms 
of argon gas (original code version in FoleyLab/MolecularDynamics: Simple Molecular Dynamics).
The code follows a generic approach to simulate particle movements over time steps, using the Newton law:


<p align="center">
 <img src="https://github.com/nunodmata/Parallel_Computing/assets/57006792/da8310d4-56fe-4355-8e7b-8cd0662b0e6b" width="600" height="300">
</p>


 The code uses Lennard Jones potential to describe the interactions among two particles (force/potential energy).
The Verlet integration method is used to calculate the particles trajectories over time. Boundary conditions (i.e., 
particles moving outside the simulation space) are managed by elastic walls.

### Work Assignment 1 
This assignment phase aims to explore optimisation techniques applied to a (single threaded) program, using
tools for code analysis/profiling and to improve and evaluate its final performance (execution time).


### Work Assignment 2
This assignment phase aims to explore shared memory parallelism (OpenMP-based) to improve the overall 
execution time.

### Work Assignment 3
This assignment phase aims to understand how to design and implement an efficient parallel version of the case 
study, eventually using accelerators (e.g., GPUs), keeping the main goal of reducing the execution time.
