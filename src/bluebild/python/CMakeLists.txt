

# Creates bluebild library with given name. All common target modifications should be done here.
macro(bluebild_create_library _TARGET_NAME)
	add_library(${_TARGET_NAME} ${BLUEBILD_LIBRARY_TYPE}
		${BLUEBILD_SOURCE_FILES}
		)

	set_property(TARGET ${_TARGET_NAME} PROPERTY VERSION ${BLUEBILD_VERSION})
	set_property(TARGET ${_TARGET_NAME} PROPERTY SOVERSION ${BLUEBILD_SO_VERSION})


	# Don't export any symbols of external static libaries. Only works on linux.
	if(UNIX AND NOT APPLE)
		if(${CMAKE_VERSION} VERSION_LESS "3.13.5") 
			target_link_libraries(${_TARGET_NAME} PRIVATE "-Wl,--exclude-libs,ALL")
		else()
			target_link_options(${_TARGET_NAME} PRIVATE "-Wl,--exclude-libs,ALL")
		endif()
	endif()

	# All .cu files are self-contained. Device linking can have issues with propageted linker flags of other targets like MPI.
	if(BLUEBILD_CUDA)
		set_property(TARGET ${_TARGET_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
		set_property(TARGET ${_TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
	endif()

	target_compile_options(${_TARGET_NAME} PRIVATE ${BLUEBILD_DEFINITIONS} ${BLUEBILD_EXTERNAL_COMPILE_OPTIONS})
	target_include_directories(${_TARGET_NAME} PRIVATE ${BLUEBILD_EXTERNAL_INCLUDE_DIRS} ${BLUEBILD_INCLUDE_DIRS})


	target_link_libraries(${_TARGET_NAME} PRIVATE ${BLUEBILD_EXTERNAL_LIBS})

	target_include_directories(${_TARGET_NAME} INTERFACE ${BLUEBILD_INTERFACE_INCLUDE_DIRS})
	target_include_directories(${_TARGET_NAME} INTERFACE $<INSTALL_INTERFACE:include>) # for install(EXPORT ...)
	target_include_directories(${_TARGET_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>) # for export(...)

endmacro()


pybind11_add_module(bluebild_module bluebild_module.cpp)
target_link_libraries(bluebild_module PRIVATE bluebild)
set_target_properties(bluebild_module PROPERTIES OUTPUT_NAME bluebild)


# add_library(bluebild_module MODULE bluebild_module.cpp)
# target_link_libraries(bluebild_module PRIVATE pybind11::module bluebild)
# set_target_properties(bluebild_module PROPERTIES OUTPUT_NAME bluebild)

# set_target_properties(bluebild_module PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)
# The output name of the pyarb .so file is "_bluebild"
# With this, the full name of the library will be something like:
#   bluebild.cpython-36m-x86_64-linux-gnu.so
# set_target_properties(bluebild_module PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

