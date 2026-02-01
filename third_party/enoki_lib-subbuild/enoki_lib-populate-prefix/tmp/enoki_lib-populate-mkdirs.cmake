# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "E:/workspace/NexDynSDF/third_party/enoki_lib-src")
  file(MAKE_DIRECTORY "E:/workspace/NexDynSDF/third_party/enoki_lib-src")
endif()
file(MAKE_DIRECTORY
  "E:/workspace/NexDynSDF/third_party/enoki_lib-build"
  "E:/workspace/NexDynSDF/third_party/enoki_lib-subbuild/enoki_lib-populate-prefix"
  "E:/workspace/NexDynSDF/third_party/enoki_lib-subbuild/enoki_lib-populate-prefix/tmp"
  "E:/workspace/NexDynSDF/third_party/enoki_lib-subbuild/enoki_lib-populate-prefix/src/enoki_lib-populate-stamp"
  "E:/workspace/NexDynSDF/third_party/enoki_lib-subbuild/enoki_lib-populate-prefix/src"
  "E:/workspace/NexDynSDF/third_party/enoki_lib-subbuild/enoki_lib-populate-prefix/src/enoki_lib-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "E:/workspace/NexDynSDF/third_party/enoki_lib-subbuild/enoki_lib-populate-prefix/src/enoki_lib-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "E:/workspace/NexDynSDF/third_party/enoki_lib-subbuild/enoki_lib-populate-prefix/src/enoki_lib-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
