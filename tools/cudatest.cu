


#include <stdio.h>




__global__ void advanceParticles(float dt, int nParticles)
{
  /*
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < nParticles)
	{
		pArray[idx].advance(dt);
	}
  */
}


int main()
{
  cudaError_t error;

  error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    printf("0 %s\n",cudaGetErrorString(error));
    exit(1);
  }

    int nDevices;
  cudaGetDeviceCount(&nDevices);

  printf("Number of devices: %d\n", nDevices);

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    printf("  Warp-size: %d\n", prop.warpSize);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
  }


#if 0
  particle * pArray = new particle[n];
  particle * devPArray = NULL;
  cudaMalloc(&devPArray, n*sizeof(particle));
  cudaDeviceSynchronize(); error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    printf("1 %s\n",cudaGetErrorString(error));
    exit(1);
  }

  cudaMemcpy(devPArray, pArray, n*sizeof(particle), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize(); error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    printf("2 %s\n",cudaGetErrorString(error));
    exit(1);
  }

  for(int i=0; i<100; i++)
  {
    float dt = (float)rand()/(float) RAND_MAX; // Random distance each step
    advanceParticles<<< 1 +  n/256, 256>>>(dt, devPArray, n);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
      printf("3 %s\n",cudaGetErrorString(error));
      exit(1);
    }

    cudaDeviceSynchronize();
  }
  cudaMemcpy(pArray, devPArray, n*sizeof(particle), cudaMemcpyDeviceToHost);

  v3 totalDistance(0,0,0);
  v3 temp;
  for(int i=0; i<n; i++)
  {
    temp = pArray[i].getTotalDistance();
    totalDistance.x += temp.x;
    totalDistance.y += temp.y;
    totalDistance.z += temp.z;
  }
  float avgX = totalDistance.x /(float)n;
  float avgY = totalDistance.y /(float)n;
  float avgZ = totalDistance.z /(float)n;
  float avgNorm = sqrt(avgX*avgX + avgY*avgY + avgZ*avgZ);
  printf(	"Moved %d particles 100 steps. Average distance traveled is |(%f, %f, %f)| = %f\n",
      n, avgX, avgY, avgZ, avgNorm);

#endif
  return 0;
}

