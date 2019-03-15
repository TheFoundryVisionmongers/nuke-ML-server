# CMakeLists.txt for Deep Learning Plug-in: DLClient

# Setting up DLClient sources and dependencies
set (DL_CLIENT_SOURCES
        DLClient.cpp
        message.pb.cc
)

find_package(Protobuf REQUIRED)

# Create DLClient.so shared library
add_library(DLClient SHARED
    ${DL_CLIENT_SOURCES}
)
set_target_properties (DLClient PROPERTIES PREFIX "")
target_include_directories(DLClient PRIVATE
    "/usr/local/Nuke11.3v1/include")
target_link_libraries(DLClient
    ${PROTOBUF_LIBRARY}
)