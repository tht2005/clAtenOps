cmake_minimum_required(VERSION 3.15)
project(clAtenOps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

add_library(main main.cpp)

target_include_directories(main PRIVATE
  ${OpenCL_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
)
target_link_libraries(main PRIVATE
  OpenCL::OpenCL
  ${TORCH_LIBRARIES}
  Python3::Python
)
target_compile_definitions(main PRIVATE
  ${TORCH_CXX_FLAGS}
)

# Generate kernel registry header
set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
set(KERNEL_OUT "${CMAKE_CURRENT_BINARY_DIR}/generated/kernel_registry.hpp")
file(GLOB KERNEL_FILES "${KERNEL_DIR}/*.cl")
set(KERNEL_REGISTRY_HPP_CONTENT
"#pragma once
#include <string>
#include <unordered_map>

namespace ht {
	struct KernelRegistry {
		static const std::unordered_map<std::string, std::string>& get_sources() {
			static const std::unordered_map<std::string, std::string> sources = {
")

foreach(FILE_PATH ${KERNEL_FILES})
  get_filename_component(FILE_NAME ${FILE_PATH} NAME_WE)
  file(READ ${FILE_PATH} FILE_CONTENT)
  string(APPEND KERNEL_REGISTRY_HPP_CONTENT
	"
				{ \"${FILE_NAME}\", R\"CLC(
${FILE_CONTENT}
)CLC\" },
")
endforeach()

string(APPEND KERNEL_REGISTRY_HPP_CONTENT
"
			};
			return sources;
		}
	};
} // namespace ht
")
file(WRITE ${KERNEL_OUT} "${KERNEL_REGISTRY_HPP_CONTENT}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/generated")
