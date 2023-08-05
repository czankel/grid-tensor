# Copyright (C) Chris Zankel. All rights reserved.
# This code is subject to U.S. and other copyright laws and
# intellectual property protections.
#
# The contents of this file are confidential and proprietary to Chris Zankel.

include(CheckCXXCompilerFlag)

# Function to list all header files in the current directory,
# recursing into sub-directories
function(grid_list_header_files HEADER_FILES HEADER_DIRS)
	message ("FOREACH " ${HEADER_DIRS})
	foreach(dir ${HEADER_DIRS})
		file(GLOB_RECURSE HEADER_FILES_TMP RELATIVE "${CMAKE_SOURCE_DIR}/${dir}" "${dir}/*.h")
		list(APPEND HEADER_FILES_ ${HEADER_FILES_TMP})
	endforeach()
	set(${HEADER_FILES} ${HEADER_FILES_} PARENT_SCOPE)
endfunction()


# Function to list all source files in the current directory
function(grid_list_source_files SOURCE_DIR SOURCE_FILES)
	file(GLOB_RECURSE SOURCE_FILES_TMP "${SOURCE_DIR}/*.c" "${SOURCE_DIR}/*.cc")
	set(SOURCE_FILES ${SOURCE_FILES_TMP} PARENT_SCOPE)
endfunction()


function(grid_add_sources target)
	# define the <target>_SOURCES property if necessary
	get_property(prop_defined GLOBAL PROPERTY ${target}_SOURCES DEFINED)
	if (NOT prop_defined)
		define_property(GLOBAL PROPERTY ${target}_SOURCES
			BRIEF_DOCS "Sources for the ${target} target"
			FULL_DOCS "List of source files for the ${target} target")
	endif()
	# create list of source (absolute paths)
	set (SOURCES)
	foreach(source IN LISTS ARGN)
		if (NOT IS_ABSOLUTE "${source}")
			get_filename_component(source "${source}" ABSOLUTE)
			#FILE(RELATIVE_PATH path ${CMAKE_SOURCE_DIR} "${source}")
			#set(source "${path}/${source}")
		endif()
		list(APPEND SOURCES "${source}")
	endforeach()
	# append to global property
	set_property(GLOBAL APPEND PROPERTY "${target}_SOURCES" "${SOURCES}")
endfunction()

# Function to check if a given compiler flag is supported and add it to the
# compiler flags.
function(check_cxx_compiler_flag_and_append flag)
	string(REGEX REPLACE "[=-]" "_" _FLAG_NAME ${flag})
	string(TOUPPER HAS_${_FLAG_NAME} _FLAG_NAME)
	check_cxx_compiler_flag("${flag}" ${_FLAG_NAME})
	if(${_FLAG_NAME})
	  string (APPEND  CMAKE_CXX_FLAGS " ${flag}")
        endif()
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
endfunction()
