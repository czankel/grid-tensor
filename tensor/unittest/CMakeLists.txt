grid_add_sources(gridtensor_test
  tensor_base.cc
  addition_base.cc
  multiplication_base.cc
  rms_norm_base.cc
  softmax_base.cc
)

if (BUILD_METAL)
grid_add_sources(gridtensor_test
  tensor_metal.cc
  addition_metal.cc
)
endif()
