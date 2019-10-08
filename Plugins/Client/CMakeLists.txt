# CMakeLists.txt for Machine Learning Plug-ins: MLClient and MLClientGeo

# Setting up MLClient sources and dependencies
set (ML_CLIENT_SOURCES
        MLClientComms.cpp
        MLClientModelManager.cpp
)

find_package(Protobuf REQUIRED)
# Compile protobuf .cpp and .h files out of message.proto
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS message.proto)
list(APPEND ML_CLIENT_SOURCES ${PROTO_SRCS}) # add message.pb.cc

set (ML_CLIENT_IOP_SOURCES
        MLClient.cpp
        ${ML_CLIENT_SOURCES}
)
set (ML_CLIENT_GEOOP_SOURCES
        MLClientGeo.cpp
        ${ML_CLIENT_SOURCES}
)

if(NOT NUKE_INSTALL_PATH)
    message(FATAL_ERROR "Nuke install path not set.")
endif()
find_library(DDIMAGE_LIBRARY NAME DDImage libDDImage PATHS ${NUKE_INSTALL_PATH})
if(NOT DDIMAGE_LIBRARY)
    message(FATAL_ERROR "DDImage library not found.")
endif()

# Create MLClient.so shared library
add_library(MLClient SHARED
    ${ML_CLIENT_IOP_SOURCES}
)
# Create MLClientGeo.so shared library
add_library(MLClientGeo SHARED
    ${ML_CLIENT_GEOOP_SOURCES}
)

set(ML_CLIENT_INCLUDE_DIRS
    ${NUKE_INSTALL_PATH}/include
    ${CMAKE_CURRENT_BINARY_DIR} # include message.pb.h
    ${PROTOBUF_INCLUDE_DIRS}
)

set_target_properties (MLClient PROPERTIES PREFIX "")
target_include_directories(MLClient PRIVATE
    ${ML_CLIENT_INCLUDE_DIRS}
)
target_link_libraries(MLClient
    ${PROTOBUF_LIBRARY}
    ${DDIMAGE_LIBRARY}
)

set_target_properties (MLClientGeo PROPERTIES PREFIX "")
target_include_directories(MLClientGeo PRIVATE
    ${ML_CLIENT_INCLUDE_DIRS}
)
target_link_libraries(MLClientGeo
    ${PROTOBUF_LIBRARY}
    ${DDIMAGE_LIBRARY}
)
