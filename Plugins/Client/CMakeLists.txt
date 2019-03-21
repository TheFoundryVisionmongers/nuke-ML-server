# CMakeLists.txt for Deep Learning Plug-in: DLClient

# Setting up DLClient sources and dependencies
set (DL_CLIENT_SOURCES
        DLClient.cpp
        message.pb.cc
)

find_package(Protobuf REQUIRED)

if(NOT NUKE_INSTALL_PATH)
    message(FATAL_ERROR "Nuke install path not set.")
endif()
find_library(DDIMAGE_LIBRARY NAME DDImage libDDImage PATHS ${NUKE_INSTALL_PATH})
if(NOT DDIMAGE_LIBRARY)
    message(FATAL_ERROR "DDImage library not found.")
endif()


# Create DLClient.so shared library
add_library(DLClient SHARED
    ${DL_CLIENT_SOURCES}
)

set_target_properties (DLClient PROPERTIES PREFIX "")
target_include_directories(DLClient PRIVATE
    ${NUKE_INSTALL_PATH}/include)
target_link_libraries(DLClient
    ${PROTOBUF_LIBRARY}
    ${DDIMAGE_LIBRARY}
)

install(TARGETS DLClient DESTINATION ${CMAKE_BINARY_DIR}/lib)