# Copyright (C) Chris Zankel. All rights reserved.
# This code is subject to U.S. and other copyright laws and
# intellectual property protections.
#
# The contents of this file are confidential and proprietary to Chris Zankel.

# set_property(GLOBAL PROPERTY gridtensor_cuda_PATH ${CMAKE_CURRENT_BINARY_DIR})
# set(gridtensor_cuda_PATH ${CMAKE_CURRENT_BINARY_DIR})

#set(
#  KERNELS
#  "binary"
#)
### set_property(TARGET myexe PROPERTY CUDA_SEPARABLE_COMPILATION ON)
grid_add_sources(gridtensor
	binary.cu
	matmul.cu
	rms_norm.cu
	wrapper.cu
)

#add_custom_target(
# gridtensor_cudalib
  # ?? DEPENDS ${gridtensor_metal_PATH}/gridtensor.metallib
  #)
