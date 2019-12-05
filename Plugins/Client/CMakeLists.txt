# CMakeLists.txt for Machine Learning Plug-in: MLClient

# Setting up MLClient sources and dependencies
set (ML_CLIENT_SOURCES
        MLClient.cpp
        MLClientComms.cpp
        MLClientModelManager.cpp
)

find_package(Protobuf REQUIRED)
if (WIN32)
	find_library(PROTOBUF_LIBRARY NAME libprotobuf PATHS ${Protobuf_LIBRARIES})
endif()

# Compile protobuf .cpp and .h files out of message.proto
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS message.proto)
list(APPEND ML_CLIENT_SOURCES ${PROTO_SRCS}) # add message.pb.cc

if(NOT NUKE_INSTALL_PATH)
    message(FATAL_ERROR "Nuke install path not set.")
endif()
find_library(DDIMAGE_LIBRARY NAME DDImage libDDImage PATHS ${NUKE_INSTALL_PATH})
if(NOT DDIMAGE_LIBRARY)
    message(FATAL_ERROR "DDImage library not found.")
endif()

# Create MLClient.so shared library
add_library(MLClient SHARED
    ${ML_CLIENT_SOURCES}
)

set_target_properties (MLClient PROPERTIES PREFIX "")
target_include_directories(MLClient PRIVATE
    ${NUKE_INSTALL_PATH}/include
    ${CMAKE_CURRENT_BINARY_DIR} # include message.pb.h
    ${Protobuf_INCLUDE_DIR}
)

target_link_libraries(MLClient
    ${PROTOBUF_LIBRARY}
    ${DDIMAGE_LIBRARY}
)

if (WIN32)
    target_link_libraries(MLClient
        ws2_32.lib  # include windows socket library
    )
endif (WIN32)