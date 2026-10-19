// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
//#include <cutil.h>
#include "cuda.h"
#include "mex.h"

#define BLOCKDIM 256

// stokeslet version
__device__ float3 
bodyBodyInteraction(float3 Vi, float4 strength, float4 bi, float4 bj, float delta) 
{
//  First make the sources the image
//    h0 =sources(:,3);    % Height of sources
//    sources(:,3)= -h0;   % Image of sources
    float h0 = bi.z;
    bi.z = -h0;

 //[matlab]
 //	  dx1 = targets(:,1)-sources(i,1)
 //   dx2 = targets(:,2)-sources(i,2);
 //   dx3 = targets(:,3)-sources(i,3);
	// 3 flops
    float3 dx;
    dx.x = bj.x - bi.x;
    dx.y = bj.y - bi.y;
    dx.z = bj.z - bi.z;	// image part

	// 18 flops
// [matlab] r2  = dx1.^2 + dx2.^2 + dx3.^2;
    // distSqr
    float r2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;

// [matlab] Stokeslets stuff
// H1 = 1/2 ./(r2+d2(i)).^(1/2) + d2(i)/2 ./(r2+d2(i)).^(3/2);

	float d2 = delta * delta;

	float H1 = 0.5f / sqrtf(r2 + d2) + d2 / 2 / powf( sqrtf(r2 + d2), 3);

// [matlab] H2 = 1/2 ./(r2+d2(i)).^(3/2);
	float H2 = 0.5f / powf( sqrtf(r2 + d2), 3);

// [matlab]
//     % Dipole stuff
//     Hdp1 =  -h0(i)^2*( 1./(r2+d2(i)).^(3/2) - 3*d2(i)./(r2+d2(i)).^(5/2) );
//     Hdp2 = -h0(i)^2*( -3./(r2+d2(i)).^(5/2) );
//     %  Doublet stuff
//     Hdb1 = 2*h0(i)*( -3/2*d2(i)./(r2+d2(i)).^(5/2) - 1/2 ./(r2+d2(i)).^(3/2) );
//     Hdb2 = 2*h0(i)*( 1/2./(r2+d2(i)).^(3/2) );
//     Hdb3 = 2*h0(i)*( -3/2./(r2+d2(i)).^(5/2) );
//     %  Rotlet stuff
//     Hr1 = (h0(i))*3*d2(i) ./(r2+d2(i)).^(5/2);

	float Hdp1 = -h0*h0*( 1.0f/powf( sqrtf(r2 + d2), 3) - 3*d2/powf( sqrtf(r2 + d2), 5) );
	float Hdp2 = -h0*h0*( -3.0f/powf( sqrtf(r2 + d2), 5) );

	float Hdb1 = 2*h0*( -3.0f / 2 * d2 / powf( sqrtf(r2 + d2), 5) - 1.0f / 2 / powf( sqrtf(r2 + d2), 3) );
	float Hdb2 = 2*h0*( 1.0f / 2 / powf( sqrtf(r2 + d2), 3) );
	float Hdb3 = 2*h0*( -3.0f / 2 / powf( sqrtf(r2 + d2), 5) );

	float Hr1 = h0 * 3 * d2 / powf( sqrtf(r2 + d2), 5);

 //[matlab]

//       % A neg Stokeslet
// 	A(1,1) = -(H1(j) + dx1(j).*dx1(j).*H2(j));
// 	A(1,2) = -(dx1(j).*dx2(j).*H2(j)) ;
// 	A(1,3) = -(dx1(j).*dx3(j).*H2(j)) ;
// 
// 	A(2,1) = -(dx2(j).*dx1(j).*H2(j)) ;
// 	A(2,2) = -(H1(j) + dx2(j).*dx2(j).*H2(j)) ;
// 	A(2,3) = -(dx2(j).*dx3(j).*H2(j)) ;
// 
// 	A(3,1) = -(dx3(j).*dx1(j).*H2(j)) ;
// 	A(3,2) = -(dx3(j).*dx2(j).*H2(j)) ;
// 	A(3,3) = -(H1(j) + dx3(j).*dx3(j).*H2(j)) ;

	// 21 flops
	float A11 = -(H1 + dx.x * dx.x * H2);
	float A12 = -(dx.x * dx.y * H2);
	float A13 = -(dx.x * dx.z * H2);
	  
	float A21 = -(dx.y * dx.x * H2);
	float A22 = -(H1 + dx.y * dx.y * H2);
	float A23 = -(dx.y * dx.z * H2);
	  
	float A31 = -(dx.z * dx.x * H2);
	float A32 = -(dx.z * dx.y * H2);
	float A33 = -(H1 + dx.z * dx.z * H2);

      //% A pos Dipole
      A11 = A11-( Hdp1 + dx.x*dx.x*Hdp2 ); // %i=1 j=1
      A12 = A12-( dx.x*dx.y*Hdp2 );          //  %i=1 j=2
      A13 = A13+( dx.x*dx.z*Hdp2 );          //  %i=1 j=3
      
      A21 = A21-( dx.y*dx.x*Hdp2 );          // %i=2 j=1
      A22 = A22-( Hdp1 + dx.y*dx.y*Hdp2 ); //%i=2 j=2
      A23 = A23+ ( dx.y*dx.z*Hdp2 );        //  %i=2 j=3
      
      A31 = A31-( dx.z*dx.x*Hdp2 );         //  %i=3 j=1
      A32 = A32-( dx.z*dx.y*Hdp2 );         //  %i=3 j=2
      A33 = A33+( Hdp1 + dx.z*dx.z*Hdp2 ); //%i=3 j=3

      //% A pos Doublet
      A11 = A11-( dx.z*(Hdb2 + dx.x*dx.x*Hdb3) ) ;
      A12 = A12-( dx.x*dx.y*dx.z*Hdb3 );     //% i=1 j=2
      A13 = A13+dx.x*(Hdb2 + dx.z*dx.z*Hdb3 );  // % i=1 j=3
      
      A21 = A21-( dx.y*dx.x*dx.z*Hdb3 ) ;          // % i=2 j=1
      A22 = A22-( dx.z*(Hdb2 + dx.y*dx.y*Hdb3) ) ; 
      A23 = A23+dx.y*(Hdb2+ dx.z*dx.z*Hdb3 ) ; //% i=2 j=3
      
      A31 = A31-( dx.x*(Hdb1 + dx.z*dx.z*Hdb3 ) ); 
      A32 = A32-( dx.y*(Hdb1 + dx.z*dx.z*Hdb3 ) );
      A33 = A33+ dx.z*(Hdb1 + 2*Hdb2 + dx.z*dx.z*Hdb3);

      //% A pos Rotlet
      A11 = A11 + dx.z*Hr1;
      A22 = A22 + dx.z*Hr1;
      A31 = A31-dx.x*Hr1;
      A32 = A32-dx.y*Hr1;

// [matlab] V(j,:) = V(j,:) + (A*strengths(i,:)')'/(4*pi);   
	// 0.154 if 32 thread
	float3 tmpVi;
 	tmpVi.x = Vi.x + (A11 * strength.x + A12 * strength.y + A13 * strength.z) / 12.566370614359172;
 	tmpVi.y = Vi.y + (A21 * strength.x + A22 * strength.y + A23 * strength.z) / 12.566370614359172;
 	tmpVi.z = Vi.z + (A31 * strength.x + A32 * strength.y + A33 * strength.z) / 12.566370614359172;
	return tmpVi;
}


// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i]
#define SX_strength(i) sharedPos[i+blockDim.x]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]

// This is the "tile_calculation" function from the GPUG3 article.
__device__ float3 gravitation(float4 myPos, float4* pStrength, float3 accel, int numBody, float delta)
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

    for (unsigned int counter = 0; counter < numBody; ) 
    {
        accel = bodyBodyInteraction(accel, SX_strength(i), SX(i), myPos, delta); i++;
		//accel = bodyBodyInteraction(accel, pStrength[counter], SX(i++), myPos); 
		counter++;
    }

    return accel;
}

__device__ float3
computeBodyAccel(float4 bodyPos, float4* pStrength, float4* positions, int numBodiesTarget, int numBodiesSource, float delta)
{
    extern __shared__ float4 sharedPos[];

    float3 acc = {0.0f, 0.0f, 0.0f};
    
    int p = blockDim.x;
    int q = blockDim.y;
    int n = numBodiesSource;
    int numTiles = n / (p * q);

	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    for (int tile = 0; tile < numTiles; tile++) 
    {
		// all blocks star from index 0.
		int posIdx = tile * blockDim.x + threadIdx.x;

		// the first half is pos
		sharedPos[threadIdx.x] = positions[posIdx];
		// the second half is strength
		sharedPos[threadIdx.x + blockDim.x] = pStrength[posIdx];
   
        __syncthreads();

		if (index < numBodiesTarget)
		{
			// This is the "tile_calculation" function from the GPUG3 article.
			acc = gravitation(bodyPos, pStrength, acc, blockDim.x, delta);
		}
        
        __syncthreads();
    }

	// handle if numBodiesTarget != 32 * n  (the last tile)
	int bodyLeft = n - numTiles * (p * q);
	if (bodyLeft > 0)
	{
		// only 'bodyLeft' threads have to write the entries in share memory, (32-bodyLeft) threads are idle.
		if (threadIdx.x < bodyLeft)
		{
			int posIdx = numTiles * blockDim.x + threadIdx.x;
			// the first half is pos
			sharedPos[threadIdx.x] = positions[posIdx];
			// the second half is strength
			sharedPos[threadIdx.x + blockDim.x] = pStrength[posIdx];
		}
   
        __syncthreads();

		if (index < numBodiesTarget)
		{
			// This is the "tile_calculation" function from the GPUG3 article.
			acc = gravitation(bodyPos, pStrength, acc, bodyLeft, delta);
		}
        
        __syncthreads();
	}

    return acc;
}

__global__ void
integrateBodies(float4* newPos, float4* newVel, 
                float4* oldPos, float4* oldVel, float4* pStrength, float4* oldPosSrc,
                float deltaTime, float delta,
                int numBodiesTarget, int numBodiesSource)
{
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	// no idea why this code doesn't work (because computeBodyAccel is in if statement?).	-ksl100312
/*
	if (index >= numBodiesTarget)
	{
		//return;	// note we can't return here since we need other threads to write share mem.
		float4 pos = oldPos[0]; // some dummy index.
		computeBodyAccel(pos, pStrength, oldPos, numBodiesTarget);
	}
	else
	{
		float4 pos = oldPos[index];
		// for stokeslets, we get velocity directly.
		float3 velocity = computeBodyAccel(pos, pStrength, oldPos, numBodiesTarget);
		float4 vel;
		vel.x = velocity.x;
		vel.y = velocity.y;
		vel.z = velocity.z;
	        
		// store new position and velocity
		newPos[index] = pos;	// actually pos will not be used.
		newVel[index] = vel;
	}
*/

	if (index >= numBodiesTarget)
//		return;	// note we can't return here since we need other thread to write share mem.
		index = 0;	// assign dummy index.

    float4 pos = oldPos[index];

	// for stokeslets, we get velocity directly.
	float3 velocity = computeBodyAccel(pos, pStrength, oldPosSrc, numBodiesTarget, numBodiesSource,delta);

	index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if (index < numBodiesTarget)
	{
		float4 vel;
		vel.x = velocity.x;
		vel.y = velocity.y;
		vel.z = velocity.z;
	        
		// store new position and velocity
		newPos[index] = pos;	// actually pos will not be used.
		newVel[index] = vel;
	}
}

extern "C"
{
void copyArrayFromDevice(float* host, const float* device, int numBodiesTarget)
{   
//    CUDA_SAFE_CALL(cudaMemcpy(host, device, numBodiesTarget*4*sizeof(float), cudaMemcpyDeviceToHost));
    cudaMemcpy(host, device, numBodiesTarget*4*sizeof(float), cudaMemcpyDeviceToHost);
}

void copyArrayToDevice(float* device, const float* host, int numBodiesTarget)
{
//    CUDA_SAFE_CALL(cudaMemcpy(device, host, numBodiesTarget*4*sizeof(float), cudaMemcpyHostToDevice));
    cudaMemcpy(device, host, numBodiesTarget*4*sizeof(float), cudaMemcpyHostToDevice);
}

float g_delta;

void threadSync() { cudaThreadSynchronize(); }

void 
integrateNbodySystem(float* newPos, float* newVel, 
                     float* oldPos, float* oldVel, float* str, float* oldPosSrc,
                     float deltaTime, int numBodiesTarget, int numBodiesSource, 
					 int p, int q)
{
	// 4 floats for pos and 4 flosts for strength
    int sharedMemSize = p * q * sizeof(float4)*2; 
    
    dim3 threads(p,q,1);
    dim3 grid(ceil(float(numBodiesTarget)/p), 1, 1);

	integrateBodies<<< grid, threads, sharedMemSize >>>
		((float4*)newPos, (float4*)newVel,
		(float4*)oldPos, (float4*)oldVel, (float4*)str,(float4*)oldPosSrc,
		deltaTime, g_delta, numBodiesTarget, numBodiesSource);
}

}

//--------------------------------------------------------------------------------------------

void allocateNBodyArrays(float* vel[2], int numBodiesTarget)
{
    // 4 floats each for alignment reasons
    unsigned int memSize = sizeof( float) * 4 * numBodiesTarget;
    
    int ret=cudaMalloc((void**)&vel[0], memSize);
	//printf("allocate array: %d, size:%d\n",ret, memSize);
    ret=cudaMalloc((void**)&vel[1], memSize);
	//printf("allocate array: %d, size:%d\n",ret, memSize);
}

void deleteNBodyArrays(float* vel[2])
{
    cudaFree((void**)vel[0]);
    cudaFree((void**)vel[1]);
}

// CPU data (host)
float* m_hPos;
float* m_hPosSource;
float* m_hVel;
float* m_hStr;
// GPU data (device)
float* m_dPosSource;
float* m_dPos[2];
float* m_dVel[2];
float* m_dStr;
unsigned int m_currentRead = 0;
unsigned int m_currentWrite = 1;

int g_numTargets = 4096;
int g_numSources = 4096;
int g_p = 32;
//int g_p = 32;
int g_q = 1;

void nbody_update(float deltaTime)
{
     integrateNbodySystem(
		m_dPos[m_currentWrite], m_dVel[m_currentWrite],
		m_dPos[m_currentRead], m_dVel[m_currentRead],m_dStr,m_dPosSource,
		deltaTime, g_numTargets, g_numSources,
		g_p, g_q);

	 unsigned int tmp = m_currentRead;
	 m_currentRead = m_currentWrite;
	 m_currentWrite = tmp;

//    std::swap(m_currentRead, m_currentWrite);
}

void nbodySimulation(int iterations)
{
	float timeStep=1;
    // once without timing to prime the GPU
    nbody_update(timeStep);

	threadSync();
}

/* Gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
				 int nrhs, const mxArray *prhs[])
{
    // test for matlab + glut   -ksl091129
    //static int a=1;
    //printf("iteration:%d\n", a++);

	int numIterations = 1;

	// get basic parameters
	{
		// get number of SMs on this GPU
		int devID;
		cudaDeviceProp props;
		cudaGetDevice(&devID);
		cudaGetDeviceProperties(&props, devID);

	    // num of targets = rows of parameter 0.
		g_numTargets = mxGetM(prhs[0]); // targets
		g_numSources = mxGetM(prhs[1]); // sources
//		printf("target:%d, source:%d\n", g_numTargets,g_numSources);
	}

	// initilize
	{
		// allocate host memory (CPU data)
		m_hPos = new float[g_numTargets*4];
		m_hVel = new float[g_numTargets*4];
		m_hStr = new float[g_numTargets*4];
		m_hPosSource = new float[g_numSources*4];
		memset(m_hPos, 0, g_numTargets*4*sizeof(float));
		memset(m_hVel, 0, g_numTargets*4*sizeof(float));
		memset(m_hStr, 0, g_numTargets*4*sizeof(float));
		memset(m_hPosSource, 0, g_numSources*4*sizeof(float));

		/* Retrieve the input data */
		{
			/* Check if the input array is single or double precision */
			//category = mxGetClassID(prhs[0]);
			//if( category == mxSINGLE_CLASS);
			//if( category == mxDOUBLE_CLASS);

			// assume data type is double
			double *dataTarget = mxGetPr(prhs[0]);   // 0: target
			double *dataSource = mxGetPr(prhs[1]);   // 1: source
			double *dataStr = mxGetPr(prhs[2]);   // 2: strength
			double *dataDelta = mxGetPr(prhs[3]);   // 3: delta
			g_delta = *dataDelta;

			// target position
			for (int k=0;k<g_numTargets;k++)
			{
				float *ptrPos = m_hPos + k*4;
				double *ptrData1 = dataTarget + k;

				ptrPos[0] = *ptrData1;
                ptrData1 += g_numTargets;
				ptrPos[1] = *ptrData1;
                ptrData1 += g_numTargets;
				ptrPos[2] = *ptrData1;
			}
			// strength
			for (int k=0;k<g_numSources;k++)
			{
				float *ptrPos = m_hStr + k*4;
				double *ptrData1 = dataStr + k;

				ptrPos[0] = *ptrData1;
                ptrData1 += g_numSources;
				ptrPos[1] = *ptrData1;
                ptrData1 += g_numSources;
				ptrPos[2] = *ptrData1;
			}
			// source position
			for (int k=0;k<g_numSources;k++)
			{
				float *ptrPos = m_hPosSource + k*4;
				double *ptrData1 = dataSource + k;

				ptrPos[0] = *ptrData1;
                ptrData1 += g_numSources;
				ptrPos[1] = *ptrData1;
                ptrData1 += g_numSources;
				ptrPos[2] = *ptrData1;
			}
		}

// initalize the memory
	// allocate device memory (GPU data)
		// target position
		allocateNBodyArrays(m_dPos, g_numTargets);
		// source position
		cudaMalloc((void**)&m_dPosSource, sizeof( float) * 4 * g_numSources);
		// target velocity
		allocateNBodyArrays(m_dVel, g_numTargets);
		// strength
		cudaMalloc((void**)&m_dStr, sizeof( float) * 4 * g_numSources);
		//printf("allocate str: %d, size:%d\n",ret, memSize);
	}

	// copy memory from host to device
	{
		copyArrayToDevice(m_dPos[m_currentRead], m_hPos, g_numTargets);
		copyArrayToDevice(m_dVel[m_currentRead], m_hVel, g_numTargets);
		cudaMemcpy(m_dStr, m_hStr, g_numSources*4*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(m_dPosSource, m_hPosSource, g_numSources*4*sizeof(float), cudaMemcpyHostToDevice);
		//printf("copy str: %d, size:%d\n",ret, g_numTargets*4*sizeof(float));
	}

	nbodySimulation(numIterations);

	// copy memory from device to host
	copyArrayFromDevice(m_hVel, m_dVel[m_currentRead], g_numTargets);

	{
		plhs[0] = mxCreateDoubleMatrix(g_numTargets, 3, mxREAL);

		/* Create a pointer to the output data */
		double *dataOutput = mxGetPr(plhs[0]);

		/* Convert from single to double before returning */
		for (int j = 0; j < g_numTargets; j++)
		{
			double *ptrMATLAB = dataOutput + j;
			float *ptrHost = m_hVel + j * 4;

            // note MATLAB is COLUMN-MAJOR!!!
			*ptrMATLAB = ptrHost[0];

            ptrMATLAB += g_numTargets; 
			*ptrMATLAB = ptrHost[1];

            ptrMATLAB += g_numTargets; 
			*ptrMATLAB = ptrHost[2];
		}
	}

	// uninitialize
	{
		delete [] m_hPos;
		delete [] m_hPosSource;
		delete [] m_hVel;
		delete [] m_hStr;
		deleteNBodyArrays(m_dPos);
		deleteNBodyArrays(m_dVel);
		cudaFree((void**)m_dStr);
		cudaFree((void**)m_dPosSource);
	}
}