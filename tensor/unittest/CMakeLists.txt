if (BUILD_CUDA)

grid_add_sources(gridtensor_test
  addition_cuda.cc
)

endif()

grid_add_sources(gridtensor_test
  tensor_base.cc
  addition_base.cc
  multiplication_base.cc
  rms_norm_base.cc
  softmax_base.cc
)


