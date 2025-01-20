grid_add_sources(gridtensor_test
  softmax_base.cc
)

grid_add_sources(gridtensor_test
  tensor.cc
  unary.cc
  addition.cc
  multiplication.cc
  rms_norm.cc
  rope.cc
)

if (BUILD_METAL)
grid_add_sources(gridtensor_test
  softmax_metal.cc
)
endif()
