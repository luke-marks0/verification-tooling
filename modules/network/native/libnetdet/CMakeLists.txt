cmake_minimum_required(VERSION 3.16)
project(libnetdet C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -O2")

# NOTE: This project only builds on Linux with DPDK and OpenSSL installed.
# It will not build on macOS.

# DPDK discovery via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)

# OpenSSL for SHA-256
find_package(OpenSSL REQUIRED)

add_library(netdet SHARED src/netdet.c)
target_include_directories(netdet PUBLIC src ${DPDK_INCLUDE_DIRS})
target_link_libraries(netdet ${DPDK_LINK_LIBRARIES} OpenSSL::Crypto)
target_compile_options(netdet PRIVATE ${DPDK_CFLAGS_OTHER})

# Install
install(TARGETS netdet LIBRARY DESTINATION lib)
install(FILES src/netdet.h DESTINATION include)

# Test binary (requires DPDK hardware to do anything useful)
add_executable(test_netdet tests/test_netdet.c)
target_link_libraries(test_netdet netdet)
target_include_directories(test_netdet PRIVATE src)
