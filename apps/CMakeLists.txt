
add_executable(epsilon-fourier  epsilon-fourier.cpp)
target_link_libraries(epsilon-fourier PRIVATE Threads::Threads fmt::fmt  fftw3  Eigen3::Eigen RCWA_CORE)

add_executable(rcwa-1d  main-1d.cpp)
target_link_libraries(rcwa-1d PRIVATE Threads::Threads fmt::fmt  fftw3  Eigen3::Eigen RCWA_CORE)

add_executable(test-interp  test-interp.cpp)
target_link_libraries(test-interp PRIVATE Threads::Threads fmt::fmt  fftw3  Eigen3::Eigen RCWA_CORE)

add_executable(rcwa-benchmark main-benchmark.cpp)
target_link_libraries(rcwa-benchmark PRIVATE Threads::Threads fmt::fmt  fftw3  Eigen3::Eigen RCWA_CORE)



