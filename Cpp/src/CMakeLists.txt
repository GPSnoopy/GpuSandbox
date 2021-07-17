
set(exe_name ${MAIN_PROJECT})

set(src_files_cuda
	add_vector.cu
	add_vector.hpp
	intra_return.cu
	intra_return.hpp
	squared_distance.cu
	squared_distance.hpp
)

set(src_files
	cuda_utils.cpp
	cuda_utils.hpp
	real.hpp
	utils.hpp
	main.cpp
)

source_group("Kernels" FILES ${src_files_cuda})
source_group("Main" FILES ${src_files})

add_executable(${exe_name} ${src_files_cuda} ${src_files})
set_target_properties(${exe_name} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
