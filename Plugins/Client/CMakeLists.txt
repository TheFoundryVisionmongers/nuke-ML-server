# CMakeLists.txt for Deep Learning Plug-in: DLClient

# Setting up DLClient sources and dependencies
set (DL_CLIENT_SOURCES
        DLClient.cpp
        DLClientComms.cpp
)

find_package(Protobuf REQUIRED)
# Compile protobuf .cpp and .h files out of message.proto
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS message.proto)
list(APPEND DL_CLIENT_SOURCES ${PROTO_SRCS}) # add message.pb.cc

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
    ${NUKE_INSTALL_PATH}/include
    ${CMAKE_CURRENT_BINARY_DIR} # include message.pb.h
)
target_link_libraries(DLClient
    ${PROTOBUF_LIBRARY}
    ${DDIMAGE_LIBRARY}
)