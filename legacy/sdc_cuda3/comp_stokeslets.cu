// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
//#include <cutil.h>
#include "cuda.h"
#include "mex.h"

#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

#define BLOCKDIM 256
#define LOOP_UNROLL 4

__constant__ float softeningSquared;

// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]

// stokeslet version
__device__ float3 
bodyBodyInteraction(float3 Vi, float4 bi, float4 bj) 
{
 //[matlab]
 //	  dx1 = targets(:,1)-sources(i,1)
 //   dx2 = targets(:,2)-sources(i,2);
 //   dx3 = targets(:,3)-sources(i,3);
	// 3 flops
    float3 dx;
    dx.x = bi.x - bj.x;
    dx.y = bi.y - bj.y;
    dx.z = bi.z - bj.z;
/*
{
    float3 tmpVi;
// -255;
//    tmpVi.x = 1;
//tmpVi.y = 20;
//tmpVi.z = 53;

tmpVi.x = bj.x;
tmpVi.y = bj.y;
tmpVi.z = bj.z;

// -1
//    tmpVi.x = Vi.x + dx.y;
// -1
//    tmpVi.x = Vi.x + dx.z;

    return tmpVi;
}
*/

	// 18 flops
// [matlab] r2  = dx1.^2 + dx2.^2 + dx3.^2;
    // distSqr
    float r2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
// [matlab] H1 = 1/2 ./(r2+d2(i)).^(1/2) + d2(i)/2 ./(r2+d2(i)).^(3/2);
	// assume delta is 1 => d2 is 1
	float d2 = 1.0f;
	float H1 = 0.5f / sqrtf(r2 + d2) + d2 / 2 / powf( sqrtf(r2 + d2), 3);
// [matlab] H2 = 1/2 ./(r2+d2(i)).^(3/2);
	float H2 = 0.5f / powf( sqrtf(r2 + d2), 3);

 //[matlab]
 //   for j = 1:Ntargets
 //     A(1,1) = H1(j) + dx1(j).*dx1(j).*H2(j);
 //     A(1,2) = dx1(j).*dx2(j).*H2(j) ;
 //     A(1,3) = dx1(j).*dx3(j).*H2(j) ;
 //     
 //     A(2,1) = dx2(j).*dx1(j).*H2(j) ;
 //     A(2,2) = H1(j) + dx2(j).*dx2(j).*H2(j) ;
 //     A(2,3) = dx2(j).*dx3(j).*H2(j) ;
 //     
 //     A(3,1) = dx3(j).*dx1(j).*H2(j) ;
 //     A(3,2) = dx3(j).*dx2(j).*H2(j) ;
 //     A(3,3) = H1(j) + dx3(j).*dx3(j).*H2(j) ;

 //		V(j,:) = V(j,:) + (A*strengths(i,:)')'/(4*pi);    
 //   end

	// 21 flops
	float A11 = H1 + dx.x * dx.x * H2;
    float A12 = dx.x * dx.y * H2;
    float A13 = dx.x * dx.z * H2;
      
	float A21 = dx.y * dx.x * H2;
    float A22 = H1 + dx.y * dx.y * H2;
    float A23 = dx.y * dx.z * H2;
      
    float A31 = dx.z * dx.x * H2;
    float A32 = dx.z * dx.y * H2;
    float A33 = H1 + dx.z * dx.z * H2;

// [matlab] V(j,:) = V(j,:) + (A*strengths(i,:)')'/(4*pi);   
	// TODO: figure out if the location of the sources are unchanged.
	float3 strength;
	strength.x = 1;
	strength.y = 1;
	strength.z = 1;

	// 0.154 if 32 thread
	float3 tmpVi;
	tmpVi.x = Vi.x + (A11 * strength.x + A12 * strength.y + A13 * strength.z) / 12.566370614359172;
	tmpVi.y = Vi.y + (A21 * strength.x + A22 * strength.y + A23 * strength.z) / 12.566370614359172;
	tmpVi.z = Vi.z + (A31 * strength.x + A32 * strength.y + A33 * strength.z) / 12.566370614359172;
	return tmpVi;
}

// This is the "tile_calculation" function from the GPUG3 article.
__device__ float3 gravitation(float4 myPos, float3 accel)
{
    extern __shared__ float4 sharedPos[];
    
    // The CUDA 1.1 compiler cannot determine that i is not going to 
    // overflow in the loop below.  Therefore if int is used on 64-bit linux 
    // or windows (or long instead of long long on win64), the compiler
    // generates suboptimal code.  Therefore we use long long on win64 and
    // long on everything else. (Workaround for Bug ID 347697)
#ifdef _Win64
    unsigned long long i = 0;
#else
    unsigned long i = 0;
#endif

    // Here we unroll the loop

    // Note that having an unsigned int loop counter and an unsigned
    // long index helps the compiler generate efficient code on 64-bit
    // OSes.  The compiler can't assume the 64-bit index won't overflow
    // so it incurs extra integer operations.  This is a standard issue
    // in porting 32-bit code to 64-bit OSes.

    for (unsigned int counter = 0; counter < blockDim.x; ) 
    {
        accel = bodyBodyInteraction(accel, SX(i++), myPos); 
	counter++;
#if LOOP_UNROLL > 1
        accel = bodyBodyInteraction(accel, SX(i++), myPos); 
	counter++;
#endif
#if LOOP_UNROLL > 2
        accel = bodyBodyInteraction(accel, SX(i++), myPos); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos); 
	counter += 2;
#endif
#if LOOP_UNROLL > 4
        accel = bodyBodyInteraction(accel, SX(i++), myPos); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos); 
        accel = bodyBodyInteraction(accel, SX(i++), myPos); 
	counter += 4;
#endif
    }

    return accel;
}

// WRAP is used to force each block to start working on a different 
// chunk (and wrap around back to the beginning of the array) so that
// not all multiprocessors try to read the same memory locations at 
// once.
#define WRAP(x,m) (((x)<m)?(x):(x-m))  // Mod without divide, works on values from 0 up to 2m

__device__ float3
computeBodyAccel(float4 bodyPos, float4* positions, int numBodies)
{
    extern __shared__ float4 sharedPos[];

    float3 acc = {0.0f, 0.0f, 0.0f};
    
    int p = blockDim.x;
    int q = blockDim.y;
    int n = numBodies;
    int numTiles = n / (p * q);

    for (int tile = blockIdx.y; tile < numTiles + blockIdx.y; tile++) 
    {
        sharedPos[threadIdx.x+blockDim.x*threadIdx.y] = positions[WRAP(blockIdx.x + tile, gridDim.x) * p + threadIdx.x];
       
        __syncthreads();

        // This is the "tile_calculation" function from the GPUG3 article.
        acc = gravitation(bodyPos, acc);
        
        __syncthreads();
    }

    return acc;
}

__global__ void
integrateBodies(float4* newPos, float4* newVel, 
                float4* oldPos, float4* oldVel,
                float deltaTime, float damping,
                int numBodies)
{
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    float4 pos = oldPos[index];   

	// for stokeslets, we get velocity directly.
    float3 velocity = computeBodyAccel(pos, oldPos, numBodies);

	float4 vel;
	vel.x = velocity.x;
	vel.y = velocity.y;
	vel.z = velocity.z;
        
    // store new position and velocity
    newPos[index].x = pos.x + deltaTime * vel.x;
    newPos[index].y = pos.y + deltaTime * vel.y;
    newPos[index].z = pos.z + deltaTime * vel.z;

    newVel[index] = vel;
    //newVel[index].x = 0;
    //newVel[index].y = 0;
    //newVel[index].z = 0;
}

extern "C"
{
void setDeviceSoftening(float softening)
{
    float softeningSq = softening*softening;

//    CUDA_SAFE_CALL(cudaMemcpyToSymbol("softeningSquared", &softeningSq, sizeof(float), 0, cudaMemcpyHostToDevice));
    cudaMemcpyToSymbol("softeningSquared", &softeningSq, sizeof(float), 0, cudaMemcpyHostToDevice);
}

void copyArrayFromDevice(float* host, const float* device, int numBodies)
{   
//    CUDA_SAFE_CALL(cudaMemcpy(host, device, numBodies*4*sizeof(float), cudaMemcpyDeviceToHost));
    cudaMemcpy(host, device, numBodies*4*sizeof(float), cudaMemcpyDeviceToHost);
}

void copyArrayToDevice(float* device, const float* host, int numBodies)
{
//    CUDA_SAFE_CALL(cudaMemcpy(device, host, numBodies*4*sizeof(float), cudaMemcpyHostToDevice));
    cudaMemcpy(device, host, numBodies*4*sizeof(float), cudaMemcpyHostToDevice);
}

void threadSync() { cudaThreadSynchronize(); }

void 
integrateNbodySystem(float* newPos, float* newVel, 
                     float* oldPos, float* oldVel, 
                     float deltaTime, float damping, int numBodies, 
					 int p, int q)
{
    int sharedMemSize = p * q * sizeof(float4); // 4 floats for pos
    
    dim3 threads(p,q,1);
    dim3 grid(numBodies/p, 1, 1);

	integrateBodies<<< grid, threads, sharedMemSize >>>
		((float4*)newPos, (float4*)newVel,
		(float4*)oldPos, (float4*)oldVel,
		deltaTime, damping, numBodies);
}

}

//--------------------------------------------------------------------------------------------

/* Kernel to square elements of the array on the GPU */
__global__ void square_elements(float* in, float* out, int N)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < N) out[idx]=in[idx]*in[idx];
}

void allocateNBodyArrays(float* vel[2], int numBodies)
{
    // 4 floats each for alignment reasons
    unsigned int memSize = sizeof( float) * 4 * numBodies;
    
    cudaMalloc((void**)&vel[0], memSize);
    cudaMalloc((void**)&vel[1], memSize);
}

void deleteNBodyArrays(float* vel[2])
{
    cudaFree((void**)vel[0]);
    cudaFree((void**)vel[1]);
}

float dot(float3 v0, float3 v1)
{
    return v0.x*v1.x+v0.y*v1.y+v0.z*v1.z;
}


// CPU data (host)
float* m_hPos;
float* m_hVel;
// GPU data (device)
float* m_dPos[2];
float* m_dVel[2];
unsigned int m_currentRead = 0;
unsigned int m_currentWrite = 1;

struct NBodyParams
{       
    float m_timestep;
    float m_clusterScale;
    float m_velocityScale;
    float m_softening;
    float m_damping;
    float m_pointSize;
    float m_x, m_y, m_z;
};

NBodyParams demoParams[] = 
{
    { 0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    { 0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    { 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    { 0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    { 0.016f, 6.04f, 0.0f, 1.0f, 1.0f, 0.76f, 0, 0, -50},
};

NBodyParams activeParams = demoParams[0];

int m_numBodies = 4096;
int m_p = 32;
//int m_p = 32;
int m_q = 1;

void computePerfStats(double &interactionsPerSecond, double &gflops, float milliseconds, int iterations)
{
    const int flopsPerInteraction = 20;
    interactionsPerSecond = (float)m_numBodies * (float)m_numBodies;
    interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds;
    gflops = interactionsPerSecond * (float)flopsPerInteraction;
}

void nbody_update(float deltaTime)
{
     integrateNbodySystem(
		m_dPos[m_currentWrite], m_dVel[m_currentWrite],
		m_dPos[m_currentRead], m_dVel[m_currentRead],
		deltaTime, activeParams.m_damping, m_numBodies, 
		m_p, m_q);

	 unsigned int tmp = m_currentRead;
	 m_currentRead = m_currentWrite;
	 m_currentWrite = tmp;

//    std::swap(m_currentRead, m_currentWrite);
}

void nbodySimulation(int iterations, float dt)
{
    // once without timing to prime the GPU
    //nbody_update(activeParams.m_timestep);
    nbody_update(dt);

	//unsigned int timer = 0;
	//CUT_SAFE_CALL(cutCreateTimer(&timer));
	//CUT_SAFE_CALL(cutStartTimer(timer)); 

    //for (int i = 0; i < iterations-1; ++i)
    //{
    //    nbody_update(activeParams.m_timestep);
    //}

	threadSync();
}

/* Gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
				 int nrhs, const mxArray *prhs[])
{
	int numIterations = 1;
	{
		// get number of SMs on this GPU
		int devID;
		cudaDeviceProp props;
//		CUDA_SAFE_CALL(cudaGetDevice(&devID));
//		CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, devID));
		cudaGetDevice(&devID);
		cudaGetDeviceProperties(&props, devID);
		
		printf("\n\nname[%s],global:%d,process:%d\n\n", props.name, props.totalGlobalMem,props.multiProcessorCount);

	    /* Find the dimensions of the data */
        // layout is [xxxxxxxyyyyyyyzzzzzz]
		int m= mxGetN(prhs[0]) / 3; // targets

		// default number of bodies is #SMs * 4 * CTA size
		//m_numBodies = m_p*m_q*4*props.multiProcessorCount;
		//m_numBodies = 2048;
		m_numBodies = m;
		// get number of bodies
		//cutGetCmdLineArgumenti(argc, (const char**) argv, "n", &m_numBodies);
		//// get number of iterations
		//cutGetCmdLineArgumenti(argc, (const char**) argv, "i", &numIterations);
		//// get number of threads
		//cutGetCmdLineArgumenti(argc, (const char**) argv, "p", &m_p);

		printf("n=%d,i=%d,p=%d\n", m_numBodies, numIterations, m_p);
	}

    // time step
    float dt=0.1;

	// initilize
	{
		setDeviceSoftening(0.00125f);

		// allocate host memory (CPU data)
		m_hPos = new float[m_numBodies*4];
		m_hVel = new float[m_numBodies*4];
		memset(m_hPos, 0, m_numBodies*4*sizeof(float));
		memset(m_hVel, 0, m_numBodies*4*sizeof(float));

		/* Retrieve the input data */
		{
			/* Check if the input array is single or double precision */
			//category = mxGetClassID(prhs[0]);
			//if( category == mxSINGLE_CLASS);
			//if( category == mxDOUBLE_CLASS);

			// assume data type is double
			double *data1 = mxGetPr(prhs[0]);   // target

			for (int k=0;k<m_numBodies;k++)
			{
				float *ptrPos = m_hPos + k*4;
				double *ptrData1 = data1 + k;

				ptrPos[0] = *ptrData1;

                ptrData1 += m_numBodies;
				ptrPos[1] = *ptrData1;

                ptrData1 += m_numBodies;
				ptrPos[2] = *ptrData1;
			}
        
            double *data2 = mxGetPr(prhs[1]);
            dt = *data2;
		}

		// initalize the memory
		//randomizeBodies(m_hPos, m_hVel, activeParams.m_clusterScale, activeParams.m_velocityScale, m_numBodies);

		// allocate device memory (GPU data)
		allocateNBodyArrays(m_dPos, m_numBodies);
		allocateNBodyArrays(m_dVel, m_numBodies);
	}

	// copy memory from host to device
	{
		copyArrayToDevice(m_dPos[m_currentRead], m_hPos, m_numBodies);
		copyArrayToDevice(m_dVel[m_currentRead], m_hVel, m_numBodies);
	}

	nbodySimulation(numIterations, dt);

	// copy memory from device to host
	copyArrayFromDevice(m_hVel, m_dVel[m_currentRead], m_numBodies);

	{
		//plhs[0] = mxCreateDoubleMatrix(m_numBodies, 3, mxREAL);
        plhs[0] = mxCreateDoubleMatrix(1, m_numBodies*3, mxREAL);

		/* Create a pointer to the output data */
		double *dataOutput = mxGetPr(plhs[0]);

		/* Convert from single to double before returning */
		for (int j = 0; j < m_numBodies; j++)
		{
			double *ptrMATLAB = dataOutput + j;
			float *ptrHost = m_hVel + j * 4;

            // note MATLAB is COLUMN-MAJOR!!!
			*ptrMATLAB = ptrHost[0];

            ptrMATLAB += m_numBodies; 
			*ptrMATLAB = ptrHost[1];

            ptrMATLAB += m_numBodies; 
			*ptrMATLAB = ptrHost[2];
		}
	}

	// uninitialize
	{
		delete [] m_hPos;
		delete [] m_hVel;
		deleteNBodyArrays(m_dPos);
		deleteNBodyArrays(m_dVel);	
	}
}