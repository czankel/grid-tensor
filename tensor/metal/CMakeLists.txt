
set(GRIDTENSOR_METAL_PATH ${CMAKE_CURRENT_BINARY_DIR})
set_property(GLOBAL PROPERTY GRIDTENSOR_METAL_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(
  KERNELS
  "binary"
  "gemm"
)

function(build_kernel TARGET SRCFILE)
  message("BUILD ${TARGET}.air ${SRCFILE}")
  add_custom_command(
    COMMAND xcrun -sdk macosx metal -Wall -Wextra
                  -fno-fast-math
                  -c ${SRCFILE}
                  -I${PROJECT_SOURCE_DIR}
                  -o ${TARGET}.air
    DEPENDS ${SRCFILE} ## FIXME ${HEADERS}
    OUTPUT ${TARGET}.air
    COMMENT "Building ${TARGET}.air"
    VERBATIM
  )
endfunction(build_kernel)

foreach(KERNEL ${KERNELS})
  set(SRCFILE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${KERNEL}.metal)
  build_kernel(${KERNEL} ${SRCFILE})
  set(KERNELS_AIR ${KERNEL}.air ${KERNELS_AIR})
endforeach()

add_custom_command(
  OUTPUT ${GRIDTENSOR_METAL_PATH}/gridtensor.metallib
  COMMAND xcrun -sdk macosx metallib ${KERNELS_AIR} -o ${GRIDTENSOR_METAL_PATH}/gridtensor.metallib
  DEPENDS ${KERNELS_AIR}
  COMMENT "Building gridtensor.metallib"
  VERBATIM
)

add_custom_target(
  gridtensor_metallib
  DEPENDS ${GRIDTENSOR_METAL_PATH}/gridtensor.metallib
)

# FIXME remove?
# Metal cpp library (linker)
#target_link_libraries(METAL_CPP
#        "-framework Metal"
#        "-framework MetalKit"
#        "-framework AppKit"
#        "-framework Foundation"
#        "-framework QuartzCore"
#        )
