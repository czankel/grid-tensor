
Cuda::Cuda()
{
  /*
  static cudaDeviceProp GetDeviceProp();
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
  &numBlocks,
  MyKernel,
  blockSize,
  0);

*/
}

Cuda& Cuda::GetDevice()
{
  if (g_device_ == nullptr)
    g_device_ = new Cuda();

  return *g_device_;
}

grid::device::Cuda* grid::device::Cuda::g_device_;
