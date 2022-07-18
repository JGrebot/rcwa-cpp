# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.


set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include)
set(HEADER_LIST "${HEADER_DIR}/CubicSplineInterpolation.h"
                    "${HEADER_DIR}/LinearInterpolation.h"
                    "${HEADER_DIR}/ParseMatrix.h"
                    "${HEADER_DIR}/rcwa-1d.h"
                    "${HEADER_DIR}/si_green.h"
                    "${HEADER_DIR}/write-utils.h"
                )


# Make an automatic library - will be static or dynamic based on user setting
add_library(RCWA_CORE STATIC
                rcwa-1d.cpp 
			    ${HEADER_LIST})


# We need this directory, and users of our library will need it too
target_include_directories(RCWA_CORE PUBLIC ../include)
target_link_libraries(RCWA_CORE PUBLIC Threads::Threads)
target_link_libraries(RCWA_CORE PUBLIC fmt::fmt)
target_link_libraries(RCWA_CORE PUBLIC fftw3)
target_link_libraries(RCWA_CORE PUBLIC Eigen3::Eigen)


# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})


