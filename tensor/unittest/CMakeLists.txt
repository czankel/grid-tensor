grid_add_sources(gridtensor_test
  multiplication_base.cc
  rms_norm_base.cc
  softmax_base.cc
  unary_base.cc
)

grid_add_sources(gridtensor_test
  tensor.cc
  addition.cc
  rope.cc
)

if (BUILD_METAL)
grid_add_sources(gridtensor_test
  multiplication_metal.cc
  rms_norm_metal.cc
  softmax_metal.cc
  unary_metal.cc
)
endif()
