if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/BLUEBILDSharedConfig.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/BLUEBILDSharedConfig.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/BLUEBILDStaticConfig.cmake")
endif()