#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"



// Number of particles
int N;

// Lennard-Jones parameters in natural units!
double sigma = 1.;
double epsilon = 1.;
double m = 1.;
double kB = 1.;

// Avogadro's number and Boltzmann constant in SI units
double NA = 6.022140857e23;
double kBSI = 1.38064852e-23;

// Size of box, which will be specified in natural units
double L;

// Initial Temperature in Natural Units
double Tinit;

// Array sizes
const int MAXPART = 5001;

// Atom type
char atype[10];

// File names
char prefix[1000], tfn[1000], ofn[1000], afn[1000];

// File pointers
FILE* tfp, * ofp, * afp;




//GAUSSDIST function ------------------------------------------------------

// Kernel to setup curand states
__global__ void setupCurandStatesKernel(curandState* state, unsigned long long seed, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void gaussdistKernel(curandState* state, double* out, double* storedGauss, bool* hasStoredGauss, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        if (hasStoredGauss[id]) {
            // If a stored Gaussian number is available, use it
            out[id] = storedGauss[id];
            hasStoredGauss[id] = false;
        }
        else {
            double v1, v2, rsq, fac;
            do {
                v1 = curand_normal(&state[id]);
                v2 = curand_normal(&state[id]);
                rsq = v1 * v1 + v2 * v2;
            } while (rsq >= 1.0 || rsq == 0.0);

            fac = sqrt(-2.0 * log(rsq) / rsq);
            storedGauss[id] = v2 * fac;  // Store the second number for next time
            hasStoredGauss[id] = true;

            out[id] = v1 * fac;
        }
    }
}

// Host function to setup the kernel launch
void generateGaussianNumbers(double* host_out, int N) {
    double* d_out;
    double* d_storedGauss;
    bool* d_hasStoredGauss;
    curandState* d_state;

    // Allocate device memory
    cudaMalloc((void**)&d_out, N * sizeof(double));
    cudaMalloc((void**)&d_storedGauss, N * sizeof(double));
    cudaMalloc((void**)&d_hasStoredGauss, N * sizeof(bool));
    cudaMalloc((void**)&d_state, N * sizeof(curandState));

    // Initialize hasStoredGauss to false for all threads
    cudaMemset(d_hasStoredGauss, 0, N * sizeof(bool));

    // Set up the execution configuration
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Setup the states
    setupCurandStatesKernel << <blocks, threadsPerBlock >> > (d_state, time(NULL), N);

    // Launch the kernel
    gaussdistKernel << <blocks, threadsPerBlock >> > (d_state, d_out, d_storedGauss, d_hasStoredGauss, N);

    // Copy the results back to the host
    cudaMemcpy(host_out, d_out, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_out);
    cudaFree(d_storedGauss);
    cudaFree(d_hasStoredGauss);
    cudaFree(d_state);
}



// InitializeVelocities FUNCTION-----------------------------------------------------------------------------------------------------------------
// 
// Kernel for summing up velocities
__global__ void sumVelocitiesKernel(double* d_v, double* d_vSum, int N) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < N) ? d_v[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) d_vSum[blockIdx.x] = sdata[0];
}

// Kernel for summing up squared velocities
__global__ void sumSquaredVelocitiesKernel(double* d_v, double* d_vSqSum, int N) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    double v = (i < N) ? d_v[i] : 0;
    sdata[tid] = v * v;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) d_vSqSum[blockIdx.x] = sdata[0];
}

// Kernel  used to adjust the velocities of the particles based on the computed center-of-mass velocity
__global__ void adjustVelocitiesKernel(double* d_v, double* vCM, double lambda, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {

        //    Original code: Performs subtraction and scaling in separate steps. This means each velocity component is read from and written to global memory twice
        //d_v[3 * i] -= vCM[0];
        //d_v[3 * i + 1] -= vCM[1];
       // d_v[3 * i + 2] -= vCM[2];
        // Scale velocities to match the desired temperature
        //d_v[3 * i] *= lambda;
        //d_v[3 * i + 1] *= lambda;
        //d_v[3 * i + 2] *= lambda;

        // New: Combines subtractionand scaling, reducing global memory access.Each velocity component is read once, the operation is performed, and then it's written back.
        // In CUDA, reducing memory access is often key to improving performance
        // Subtract center-of-mass velocity and apply normalization factor
        d_v[3 * i] = lambda * (d_v[3 * i] - vCM[0]);
        d_v[3 * i + 1] = lambda * (d_v[3 * i + 1] - vCM[1]);
        d_v[3 * i + 2] = lambda * (d_v[3 * i + 2] - vCM[2]);
    }
}


void initializeVelocitiesCUDA(double* v, double Tinit, int N) {
    double* d_v;
    double* d_vSum, * d_vSqSum;
    size_t size = N * 3 * sizeof(double);

    // Allocate device memory
    cudaMalloc((void**)&d_v, size);
    cudaMalloc((void**)&d_vSum, sizeof(double) * 3);
    cudaMalloc((void**)&d_vSqSum, sizeof(double));

    // Generate Gaussian distributed velocities using the CUDA gaussdist function
    generateGaussianNumbers(d_v, N * 3); // Each particle needs 3 Gaussian numbers

    // Set up kernel execution parameters
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernels to sum velocities and squared velocities
    sumVelocitiesKernel << <blocks, threadsPerBlock, threadsPerBlock * sizeof(double) >> > (d_v, d_vSum, N * 3);
    sumSquaredVelocitiesKernel << <blocks, threadsPerBlock, threadsPerBlock * sizeof(double) >> > (d_v, d_vSqSum, N * 3);

    // Copy the results back to the host
    double vSum[3], vSqSum;
    cudaMemcpy(&vSum, d_vSum, sizeof(double) * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(&vSqSum, d_vSqSum, sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate center-of-mass velocity and normalization factor
    double vCM[3] = { vSum[0] / (N * m), vSum[1] / (N * m), vSum[2] / (N * m) };
    double lambda = sqrt(3 * (N - 1) * Tinit / vSqSum);

    // Launch kernel to adjust velocities
    adjustVelocitiesKernel << <blocks, threadsPerBlock >> > (d_v, vCM, lambda, N);

    // Copy adjusted velocities back to host memory
    cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_v);
    cudaFree(d_vSum);
    cudaFree(d_vSqSum);
}






//Initialize function --------------------------------------------------------------

// CUDA kernel to initialize positions in a cubic lattice
__global__ void initializePositionsKernel(double* d_r, double L, int n, int N) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < N) {
        int x = p % n;
        int y = (p / n) % n;
        int z = p / (n * n);

        double pos = L / n;
        d_r[3 * p] = (x + 0.5) * pos;
        d_r[3 * p + 1] = (y + 0.5) * pos;
        d_r[3 * p + 2] = (z + 0.5) * pos;
    }
}



// Main initialization function using CUDA
void initialize(double* r, double* v, double* a, double L, int N) {
    double* d_r, * d_v, * d_a;
    int n = (int)ceil(cbrt(N));  // Number of atoms in each direction
    size_t size = N * 3 * sizeof(double);

    // Allocate space for device copies of r, v, a
    cudaMalloc((void**)&d_r, size);
    cudaMalloc((void**)&d_v, size);
    cudaMalloc((void**)&d_a, size);

    // Initialize positions on the device
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    initializePositionsKernel << <blocksPerGrid, threadsPerBlock >> > (d_r, L, n, N);

    // Copy positions back to host
    cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);

    // Initialize velocities and accelerations (using previously defined CUDA functions)
    initializeVelocitiesCUDA(v, Tinit, N);  // Ensure this function is adapted for CUDA as well

    // Initialize accelerations (if necessary, depending on your system and potential model)

    // Cleanup
    cudaFree(d_r);
    cudaFree(d_v);
    cudaFree(d_a);
}




// MeanSquaredVelocity function--------------------------------------------------------------

// CUDA kernel to compute squared velocities
__global__ void squaredVelocitiesKernel(double* d_v, double* d_vSqd, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_vSqd[i] = d_v[3 * i] * d_v[3 * i] + d_v[3 * i + 1] * d_v[3 * i + 1] + d_v[3 * i + 2] * d_v[3 * i + 2];
    }
}

// Function to compute mean squared velocity using CUDA
double MeanSquaredVelocityCUDA(double* v, int N) {
    double* d_v, * d_vSqd;
    size_t size = N * sizeof(double);
    size_t size3 = 3 * N * sizeof(double);

    // Allocate space for device copies of v, vSqd
    cudaMalloc((void**)&d_v, size3);
    cudaMalloc((void**)&d_vSqd, size);

    // Copy inputs to device
    cudaMemcpy(d_v, v, size3, cudaMemcpyHostToDevice);

    // Launch squaredVelocitiesKernel() kernel on GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    squaredVelocitiesKernel << <blocksPerGrid, threadsPerBlock >> > (d_v, d_vSqd, N);

    // Copy result back to host
    double* vSqd = new double[N];
    cudaMemcpy(vSqd, d_vSqd, size, cudaMemcpyDeviceToHost);

    // Compute mean squared velocity on host
    double sum_vSqd = 0;
    for (int i = 0; i < N; i++) {
        sum_vSqd += vSqd[i];
    }
    double mean_vSqd = sum_vSqd / N;

    // Cleanup
    cudaFree(d_v);
    cudaFree(d_vSqd);
    delete[] vSqd;

    return mean_vSqd;
}





// Kinetic function---------------------------------------------------------------------

__global__ void kineticEnergyKernel(double* d_v, double* d_kin, double m, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double v2 = d_v[3 * i] * d_v[3 * i] + d_v[3 * i + 1] * d_v[3 * i + 1] + d_v[3 * i + 2] * d_v[3 * i + 2];
        d_kin[i] = 0.5 * m * v2;
    }
}

// Function to compute total kinetic energy using CUDA
double KineticCUDA(double* v, double m, int N) {
    double* d_v, * d_kin;
    size_t size3 = 3 * N * sizeof(double);
    size_t size = N * sizeof(double);

    // Allocate space for device copies of v, kin
    cudaMalloc((void**)&d_v, size3);
    cudaMalloc((void**)&d_kin, size);

    // Copy inputs to device
    cudaMemcpy(d_v, v, size3, cudaMemcpyHostToDevice);

    // Launch kineticEnergyKernel() kernel on GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    kineticEnergyKernel << <blocksPerGrid, threadsPerBlock >> > (d_v, d_kin, m, N);

    // Copy partial sums back to host
    double* kin = new double[N];
    cudaMemcpy(kin, d_kin, size, cudaMemcpyDeviceToHost);

    // Complete the sum on the host to get total kinetic energy
    double totalKinetic = 0;
    for (int i = 0; i < N; i++) {
        totalKinetic += kin[i];
    }

    // Cleanup
    cudaFree(d_v);
    cudaFree(d_kin);
    delete[] kin;

    return totalKinetic;
}






// Potential function ---------------------------------------------------


__global__ void pairwisePotentialKernel(double* d_r, double* d_pot, double sigma, double epsilon, int N) {
    extern __shared__ double sharedPot[]; // Shared memory for block-level reduction
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;
    int i = idx / N;
    int j = idx % N;

    sharedPot[threadId] = 0.0; // Initialize shared memory

    if (i < N && j < i) {
        double dx = d_r[3 * i] - d_r[3 * j];
        double dy = d_r[3 * i + 1] - d_r[3 * j + 1];
        double dz = d_r[3 * i + 2] - d_r[3 * j + 2];

        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 != 0.0) {
            double r6 = r2 * r2 * r2;
            double sig6 = sigma * sigma * sigma * sigma * sigma * sigma;
            double rinv6 = sig6 / r6;
            sharedPot[threadId] = 4 * epsilon * (rinv6 * rinv6 - rinv6);
        }
    }

    __syncthreads(); // Synchronize threads within the block

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadId < s) {
            sharedPot[threadId] += sharedPot[threadId + s];
        }
        __syncthreads();
    }

    // Write block-level result to a unique index in global memory
    if (threadId == 0) {
        d_pot[blockIdx.x] = sharedPot[0];
    }
}



double PotentialCUDA(double* r, double sigma, double epsilon, int N) {
    double* d_r, * d_pot;
    size_t size3 = 3 * N * sizeof(double);
    size_t size = N * N * sizeof(double);  // For all unique pairs

    // Allocate space for device copies of r, pot
    cudaMalloc((void**)&d_r, size3);
    cudaMalloc((void**)&d_pot, size);

    // Initialize d_pot to zero
    cudaMemset(d_pot, 0, size);

    // Copy inputs to device
    cudaMemcpy(d_r, r, size3, cudaMemcpyHostToDevice);

    // Launch pairwisePotentialKernel() kernel on GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(double);

    pairwisePotentialKernel << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (d_r, d_pot, sigma, epsilon, N);
    // Copy partial sums back to host
    double* pot = new double[N * N];
    cudaMemcpy(pot, d_pot, size, cudaMemcpyDeviceToHost);

    // Complete the sum on the host to get total potential energy
    double totalPotential = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        totalPotential += pot[i];
    }

    // Cleanup
    cudaFree(d_r);
    cudaFree(d_pot);
    delete[] pot;

    return totalPotential;
}



// ComputeAcceleration function ---------------------------------------------------


__global__ void computeAccelerationsKernel(double* d_r, double* d_a, double sigma, double epsilon, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Reset acceleration
        for (int k = 0; k < 3; k++) {
            d_a[3 * i + k] = 0;
        }

        // Compute acceleration contribution from all other particles
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double rij[3];
                double rSqd = 0;
                for (int k = 0; k < 3; k++) {
                    rij[k] = d_r[3 * i + k] - d_r[3 * j + k];
                    rSqd += rij[k] * rij[k];
                }

                double f = 24 * epsilon * (2 * pow(sigma, 12) / pow(rSqd, 7) - pow(sigma, 6) / pow(rSqd, 4));

                for (int k = 0; k < 3; k++) {
                    d_a[3 * i + k] += rij[k] * f;
                }
            }
        }
    }
}


void computeAccelerationsCUDA(double* h_r, double* h_a, double sigma, double epsilon, int N) {
    double* d_r, * d_a;
    size_t size = 3 * N * sizeof(double);

    // Allocate memory for device copies of r and a
    cudaMalloc((void**)&d_r, size);
    cudaMalloc((void**)&d_a, size);

    // Copy inputs to device
    cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Set up kernel execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch computeAccelerationsKernel kernel on GPU
    computeAccelerationsKernel << <blocksPerGrid, threadsPerBlock >> > (d_r, d_a, sigma, epsilon, N);

    // Copy result back to host
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_r);
    cudaFree(d_a);
}



// VelocityVerlet function ---------------------------------------------------------------

// Kernel to update positions
__global__ void updatePositionsKernel(double* d_r, double* d_v, double* d_a, double L, double dt, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        for (int k = 0; k < 3; k++) {
            d_r[3 * i + k] += d_v[3 * i + k] * dt + 0.5 * d_a[3 * i + k] * dt * dt;
            // Boundary conditions
            if (d_r[3 * i + k] < 0.0 || d_r[3 * i + k] >= L) {
                d_v[3 * i + k] *= -1.0;
            }
        }
    }
}

// Kernel to update velocities
__global__ void updateVelocitiesAndCalculatePressureKernel(double* d_v, double* d_a, double* d_pressure, double m, double dt, double L, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double psum = 0.0;

    if (i < N) {
        for (int k = 0; k < 3; k++) {
            double old_v = d_v[3 * i + k];
            d_v[3 * i + k] += 0.5 * d_a[3 * i + k] * dt;
            // Boundary conditions
            if (old_v * d_v[3 * i + k] < 0.0) { // Velocity direction changed due to collision
                psum += 2 * m * fabs(d_v[3 * i + k]) / dt;
            }
        }
    }
    d_pressure[i] = psum / (6 * L * L);
}



// Function to perform a Velocity Verlet update using CUDA
void VelocityVerletCUDA(double* r, double* v, double* a, double dt, int N, double* pressure, double L) {
    double* d_r, * d_v, * d_a, * d_pressure;
    size_t size = N * 3 * sizeof(double);

    // Allocate space and copy to device
    cudaMalloc((void**)&d_r, size);
    cudaMalloc((void**)&d_v, size);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_pressure, N * sizeof(double)); // Allocate memory for pressure

    cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    // Set up kernel execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Step 1: Update positions
    updatePositionsKernel << <blocksPerGrid, threadsPerBlock >> > (d_r, d_v, d_a, dt, N, L);

    // Step 2: Calculate new accelerations based on updated positions
    computeAccelerationsKernel << <blocksPerGrid, threadsPerBlock >> > (d_r, d_a, sigma, epsilon, N);

    // Step 3: Update velocities and calculate pressure
    updateVelocitiesAndCalculatePressureKernel << <blocksPerGrid, threadsPerBlock >> > (d_v, d_a, d_pressure, m, dt, L, N);
    // Copy results back to host
    cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pressure, d_pressure, N * sizeof(double), cudaMemcpyDeviceToHost); // Copy back pressure

    // Cleanup
    cudaFree(d_r);
    cudaFree(d_v);
    cudaFree(d_a);
    cudaFree(d_pressure); // Free pressure array
}



// main code---------------------------------------------------------------------------------------------------------------------------

int main() {

    // Variables for simulation parameters that might change in each run
    int i;
    double dt, Vol, Temp, Press, Pavg, Tavg, rho;
    double VolFac, TempFac, PressFac, timefac;
    double KE, PE, mvs, gc, Z;
    double* d_pressure;
    double* pressure = new double[N];

    int NumTime;
    clock_t start_time = clock();

    // Allocate memory for position, velocity, and acceleration vectors (flattened for CUDA)
    double* r = new double[3 * MAXPART];
    double* v = new double[3 * MAXPART];
    double* a = new double[3 * MAXPART];


    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("                  WELCOME TO WILLY P CHEM MD!\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n  ENTER A TITLE FOR YOUR CALCULATION!\n");
    scanf("%s", prefix);
    strcpy(tfn, prefix);
    strcat(tfn, "_traj.xyz");
    strcpy(ofn, prefix);
    strcat(ofn, "_output.txt");
    strcpy(afn, prefix);
    strcat(afn, "_average.txt");

    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("                  TITLE ENTERED AS '%s'\n", prefix);
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("  WHICH NOBLE GAS WOULD YOU LIKE TO SIMULATE? (DEFAULT IS ARGON)\n");
    printf("\n  FOR HELIUM,  TYPE 'He' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR NEON,    TYPE 'Ne' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR ARGON,   TYPE 'Ar' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR KRYPTON, TYPE 'Kr' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR XENON,   TYPE 'Xe' THEN PRESS 'return' TO CONTINUE\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    scanf("%s", atype);

    if (strcmp(atype, "He") == 0) {

        VolFac = 1.8399744000000005e-29;
        PressFac = 8152287.336171632;
        TempFac = 10.864459551225972;
        timefac = 1.7572698825166272e-12;

    }
    else if (strcmp(atype, "Ne") == 0) {

        VolFac = 2.0570823999999997e-29;
        PressFac = 27223022.27659913;
        TempFac = 40.560648991243625;
        timefac = 2.1192341945685407e-12;

    }
    else if (strcmp(atype, "Ar") == 0) {

        VolFac = 3.7949992920124995e-29;
        PressFac = 51695201.06691862;
        TempFac = 142.0950000000000;
        timefac = 2.09618e-12;
        //strcpy(atype,"Ar");

    }
    else if (strcmp(atype, "Kr") == 0) {

        VolFac = 4.5882712000000004e-29;
        PressFac = 59935428.40275003;
        TempFac = 199.1817584391428;
        timefac = 8.051563913585078e-13;

    }
    else if (strcmp(atype, "Xe") == 0) {

        VolFac = 5.4872e-29;
        PressFac = 70527773.72794868;
        TempFac = 280.30305642163006;
        timefac = 9.018957925790732e-13;

    }
    else {

        VolFac = 3.7949992920124995e-29;
        PressFac = 51695201.06691862;
        TempFac = 142.0950000000000;
        timefac = 2.09618e-12;
        strcpy(atype, "Ar");

    }
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n                     YOU ARE SIMULATING %s GAS! \n", atype);
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n  YOU WILL NOW ENTER A FEW SIMULATION PARAMETERS\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n\n  ENTER THE INTIAL TEMPERATURE OF YOUR GAS IN KELVIN\n");
    scanf("%lf", &Tinit);
    // Make sure temperature is a positive number!
    if (Tinit < 0.) {
        printf("\n  !!!!! ABSOLUTE TEMPERATURE MUST BE A POSITIVE NUMBER!  PLEASE TRY AGAIN WITH A POSITIVE TEMPERATURE!!!\n");
        exit(0);
    }
    // Convert initial temperature from kelvin to natural units
    Tinit /= TempFac;


    printf("\n\n  ENTER THE NUMBER DENSITY IN moles/m^3\n");
    printf("  FOR REFERENCE, NUMBER DENSITY OF AN IDEAL GAS AT STP IS ABOUT 40 moles/m^3\n");
    printf("  NUMBER DENSITY OF LIQUID ARGON AT 1 ATM AND 87 K IS ABOUT 35000 moles/m^3\n");

    scanf("%lf", &rho);

    N = 5000;
    Vol = N / (rho * NA);

    Vol /= VolFac;

    //  Limiting N to MAXPART for practical reasons
    if (N >= MAXPART) {

        printf("\n\n\n  MAXIMUM NUMBER OF PARTICLES IS %i\n\n  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY \n\n", MAXPART);
        exit(0);

    }

    if (Vol < N) {

        printf("\n\n\n  YOUR DENSITY IS VERY HIGH!\n\n");
        printf("  THE NUMBER OF PARTICLES IS %i AND THE AVAILABLE VOLUME IS %f NATURAL UNITS\n", N, Vol);
        printf("  SIMULATIONS WITH DENSITY GREATER THAN 1 PARTCICLE/(1 Natural Unit of Volume) MAY DIVERGE\n");
        printf("  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY AND RETRY\n\n");
        exit(0);
    }


    // Vol = L*L*L;
    // Length of the box in natural units:
    L = cbrt(Vol);

    //  Files that we can write different quantities to
    tfp = fopen(tfn, "w");     //  The MD trajectory, coordinates of every particle at each timestep
    ofp = fopen(ofn, "w");     //  Output of other quantities (T, P, gc, etc) at every timestep
    afp = fopen(afn, "w");    //  Average T, P, gc, etc from the simulation


    if (strcmp(atype, "He") == 0) {


        dt = 0.2e-14 / timefac;

        NumTime = 50000;
    }
    else {
        dt = 0.5e-14 / timefac;
        NumTime = 200;

    }

    auto start = std::chrono::high_resolution_clock::now();

    // Initialize the system with CUDA
    initialize(r, v, a, L, N);

    // Update accelerations - initial call
    computeAccelerationsCUDA(r, a, sigma, epsilon, N); // Ensure this function is adapted for CUDA


    // Print number of particles to the trajectory file
    fprintf(tfp, "%i\n", N);

    Pavg = 0;
    Tavg = 0;



    double* h_pressure = new double[N]; // Host array

    cudaMalloc(&d_pressure, N * sizeof(double));


    int tenp = floor(NumTime / 10);
    fprintf(ofp, "  time (s)              T(t) (K)              P(t) (Pa)           Kinetic En. (n.u.)     Potential En. (n.u.) Total En. (n.u.)\n");
    printf("  PERCENTAGE OF CALCULATION COMPLETE:\n  [");



    for (int i = 0; i < NumTime + 1; i++) {


        //  This just prints updates on progress of the calculation for the users convenience
        if (i == tenp) printf(" 10 |");
        else if (i == 2 * tenp) printf(" 20 |");
        else if (i == 3 * tenp) printf(" 30 |");
        else if (i == 4 * tenp) printf(" 40 |");
        else if (i == 5 * tenp) printf(" 50 |");
        else if (i == 6 * tenp) printf(" 60 |");
        else if (i == 7 * tenp) printf(" 70 |");
        else if (i == 8 * tenp) printf(" 80 |");
        else if (i == 9 * tenp) printf(" 90 |");
        else if (i == 10 * tenp) printf(" 100 ]\n");
        fflush(stdout);



        VelocityVerletCUDA(r, v, a, dt, N, pressure, L);
        cudaMemcpy(h_pressure, d_pressure, N * sizeof(double), cudaMemcpyDeviceToHost);

        double totalPressure = 0.0;
        for (int i = 0; i < N; ++i) {
            totalPressure += h_pressure[i];
        }

        // Use totalPressure in your calculations
        Press = totalPressure;  // Assuming Press is the total pressure variable
        Press *= PressFac;      // Scale pressure if needed



        KE = KineticCUDA(v, m, N);
        PE = PotentialCUDA(r, sigma, epsilon, N);
        mvs = MeanSquaredVelocityCUDA(v, N);
        Temp = m * mvs / (3 * kB) * TempFac;


        gc = NA * Press * (Vol * VolFac) / (N * Temp);
        Z = Press * (Vol * VolFac) / (N * kBSI * Temp);

        Tavg += Temp;
        Pavg += Press;

        fprintf(ofp, "  %8.4e  %20.8f  %20.8f %20.8f  %20.8f  %20.8f \n", i * dt * timefac, Temp, Press, KE, PE, KE + PE);

    }

    Pavg /= NumTime;
    Tavg /= NumTime;

    cudaFree(d_pressure);


    // Because we have calculated the instantaneous temperature and pressure,
    // we can take the average over the whole simulation here
    Pavg /= NumTime;
    Tavg /= NumTime;
    Z = Pavg * (Vol * VolFac) / (N * kBSI * Tavg);
    gc = NA * Pavg * (Vol * VolFac) / (N * Tavg);
    fprintf(afp, "  Total Time (s)      T (K)               P (Pa)      PV/nT (J/(mol K))         Z           V (m^3)              N\n");
    fprintf(afp, " --------------   -----------        ---------------   --------------   ---------------   ------------   -----------\n");
    fprintf(afp, "  %8.4e  %15.5f       %15.5f     %10.5f       %10.5f        %10.5e         %i\n", i * dt * timefac, Tavg, Pavg, gc, Z, Vol * VolFac, N);

    printf("\n  TO ANIMATE YOUR SIMULATION, OPEN THE FILE \n  '%s' WITH VMD AFTER THE SIMULATION COMPLETES\n", tfn);
    printf("\n  TO ANALYZE INSTANTANEOUS DATA ABOUT YOUR MOLECULE, OPEN THE FILE \n  '%s' WITH YOUR FAVORITE TEXT EDITOR OR IMPORT THE DATA INTO EXCEL\n", ofn);
    printf("\n  THE FOLLOWING THERMODYNAMIC AVERAGES WILL BE COMPUTED AND WRITTEN TO THE FILE  \n  '%s':\n", afn);
    printf("\n  AVERAGE TEMPERATURE (K):                 %15.5f\n", Tavg);
    printf("\n  AVERAGE PRESSURE  (Pa):                  %15.5f\n", Pavg);
    printf("\n  PV/nT (J * mol^-1 K^-1):                 %15.5f\n", gc);
    printf("\n  PERCENT ERROR of pV/nT AND GAS CONSTANT: %15.5f\n", 100 * fabs(gc - 8.3144598) / 8.3144598);
    printf("\n  THE COMPRESSIBILITY (unitless):          %15.5f \n", Z);
    printf("\n  TOTAL VOLUME (m^3):                      %10.5e \n", Vol * VolFac);
    printf("\n  NUMBER OF PARTICLES (unitless):          %i \n", N);


    // Stop the timer
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;


    // Cleanup dynamically allocated host memory
    delete[] r;
    delete[] v;
    delete[] a;
    delete[] pressure; // Delete if i dont use

    // Close any open files
    fclose(tfp);
    fclose(ofp);
    fclose(afp);

    // Destroy CURAND generator and other resources if i neeed it! prob not
    // curandDestroyGenerator(gen);
    // cudaStreamDestroy(myStream);
    // cudaEventDestroy(myEvent);


    return 0;

}