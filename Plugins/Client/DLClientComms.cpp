// Copyright (c) 2019 The Foundry Visionmongers Ltd.  All Rights Reserved.

// Includes for sockets and protobuf
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <sstream>
#include <arpa/inet.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "DLClientComms.h"

DLClientComms::DLClientComms()
: _isConnected(false)
, _socket(0)
, _verbose(true)
{
}

DLClientComms::~DLClientComms()
{
}

//! Returns whether this object is connected to the specified server.
bool DLClientComms::isConnected() const
{
  return _isConnected;
}

//! Tests if a given hostname is valid, returning true if it is, false otherwise
bool DLClientComms::validateHostName(const std::string& hostStr)
{
  // check if correct ipv4 or ipv6 addresses
  struct sockaddr_in sa;
  struct sockaddr_in6 sa6;
  bool is_ipv4 = inet_pton(AF_INET, hostStr.c_str(), &(sa.sin_addr)) != 0;
  bool is_ipv6 = inet_pton(AF_INET6, hostStr.c_str(), &(sa6.sin6_addr)) != 0;

  return is_ipv4 || is_ipv6;
}

//! Tries to connect to the server with the specified hostStr & port. After it
//! returns, you can test if it was successful by calling isConnected().
void DLClientComms::connectLoop(const std::string& hostStr, int port)
{
  const int kTimeout = 500000;
  const int kMaxNumberOfTry = 5;
  int i = 0;
  std::string errorStr;
  while (!setupConnection(hostStr, port, errorStr)) {
    usleep(kTimeout);
    i++;
    if (i >= kMaxNumberOfTry) {
      std::cerr << "Client -> Error setting up connection:" << std::endl;
      std::cerr << "\t" << errorStr.c_str() << std::endl;
      vprint("-----------------------------------------------");
      _isConnected = false;
      return;
    }
    std::cerr << "Client -> Failing to connect. Attempts: " << i << std::endl;
  }
  _isConnected = true;
}

//! Create a socket to connect to the server specified by hostStr and port
bool DLClientComms::setupConnection(const std::string& hostStr, int port, std::string& errorStr)
{
  try {
    int status;
    struct addrinfo hints;
    struct addrinfo* aiResult;
    // Before using hint you have to make sure that the data structure is empty
    memset(&hints, 0, sizeof hints);
    // Set the attribute for hint
    hints.ai_family = AF_INET;       // IPV4 AF_INET
    hints.ai_socktype = SOCK_STREAM; // TCP Socket SOCK_DGRAM
    hints.ai_flags = 0;
    hints.ai_protocol = IPPROTO_TCP;
    char s[INET_ADDRSTRLEN]; // to store the network address as a char

    // Fill the res data structure and make sure that the results make sense.
    status = getaddrinfo(hostStr.c_str(), std::to_string(port).c_str(), &hints, &aiResult);
    inet_ntop(aiResult->ai_family, get_in_addr((struct sockaddr *)aiResult->ai_addr), s, sizeof s);
    if (_verbose) {
      std::cerr << "Client -> Trying to connect to " << s << std::endl;
    }
    if (status != 0) {
      std::stringstream ss;
      ss << "Client -> getaddrinfo error: " << gai_strerror(status);
      errorStr = ss.str();
      std::cerr << errorStr.c_str() << std::endl;
      return false;
    }

    // Create Socket and check if error occured afterwards
    _socket = socket(aiResult->ai_family, aiResult->ai_socktype, aiResult->ai_protocol);
    if (_socket < 0) {
      std::stringstream ss;
      ss << "Client -> socket error: " << gai_strerror(_socket);
      errorStr = ss.str();
      std::cerr << errorStr.c_str() << std::endl;
      return false;
    }

    // Add socket option
    int enable = 1;
    if (setsockopt(_socket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
      errorStr = "setsockopt(SO_REUSEADDR) failed";
      return false;
    }

    long socket_flags;
    // Set non-blocking connect socket
    if ((socket_flags = fcntl(_socket, F_GETFL, NULL)) < 0) { // get socket flag argument
      std::cerr << "Client -> socket error fcntl(..., F_GETFL) (" << strerror(errno) << ")" << std::endl;
      return false;
    }
    socket_flags |= O_NONBLOCK; // add non-blocking flag to the socket flags
    if (fcntl(_socket, F_SETFL, socket_flags) < 0) { // update socket flags
      std::stringstream ss;
      ss << "Client -> socket error fcntl(..., F_SETFL) (" << strerror(errno) << ")";
      errorStr = ss.str();
      std::cerr << errorStr.c_str() << std::endl;
      return false;
    }

    // Connect to the server using the socket
    status = connect(_socket, aiResult->ai_addr, aiResult->ai_addrlen);

    int valopt;
    fd_set myset;
    struct timeval tv;
    // Trying to connect with timeout
    if (status < 0) {
      if (errno == EINPROGRESS) {
        tv.tv_sec = 1; // timeout in seconds to wait before failing to connect to host and port
        tv.tv_usec = 0;
        // Re-enable file descriptors fd that were cleared after last select() return
        FD_ZERO(&myset);
        FD_SET(_socket, &myset);
        status = select(_socket + 1, NULL, &myset, NULL, &tv);
        if (status > 0) {
          // Socket selected for write
          getsockopt(_socket, SOL_SOCKET, SO_ERROR, (void *)(&valopt), &aiResult->ai_addrlen);
          if (valopt) {
            std::stringstream ss;
            ss << "Error in socket connection " << valopt << " - " << strerror(valopt);
            errorStr = ss.str();
            std::cerr << errorStr.c_str() << std::endl;
            return false;
          }
        }
        else { // Unable to select socket
          return false;
        }
      }
      else {
        std::stringstream ss;
        ss << "Client -> socket error connecting " << errno << " " << strerror(errno);
        errorStr = ss.str();
        std::cerr << errorStr.c_str() << std::endl;
        return false;
      }
    }
    // Set to blocking mode again
    if ((socket_flags = fcntl(_socket, F_GETFL, NULL)) < 0) { // get socket flag argument
      std::stringstream ss;
      ss << "Client -> socket error fcntl(..., F_GETFL) (" << strerror(errno) << ")";
      errorStr = ss.str();
      std::cerr << errorStr.c_str() << std::endl;
      return false;
    }
    socket_flags &= (~O_NONBLOCK); // remove non-blocking flag from the socket
    if (fcntl(_socket, F_SETFL, socket_flags) < 0) { // update socket flags
      std::stringstream ss;
      ss << "Client -> socket error fcntl(..., F_SETFL) (" << strerror(errno) << ")" << std::endl;
      errorStr = ss.str();
      std::cerr << errorStr.c_str() << std::endl;
      return false;
    }

    inet_ntop(aiResult->ai_family, get_in_addr((struct sockaddr *)aiResult->ai_addr), s, sizeof s);
    if (_verbose) {
      std::cerr << "Client -> Connected to " << s << std::endl;
    }

    // Free the aiResult linked list after we are done with it
    freeaddrinfo(aiResult);
  }
  catch (const std::exception &e) {
    std::stringstream ss;
    ss << e.what();
    errorStr = ss.str();
    std::cerr << errorStr.c_str() << std::endl;
    return false;
  }
  return true;
}

//! Requests the server to return a future message about its models. This is used
//! to instruct the server that it should set itself up.
bool DLClientComms::sendInfoRequest()
{
  int bytecount;
  vprint("Sending info request");

  // Create message
  dlserver::RequestWrapper req_wrapper;
  req_wrapper.set_info(true);
  dlserver::RequestInfo* req_info = new dlserver::RequestInfo;
  req_info->set_info(true);
  req_wrapper.set_allocated_r1(req_info);
  vprint("Created message");

  // Generate the data which should be sent over the network
  std::string request_s;
  req_wrapper.SerializeToString(&request_s);
  int length = request_s.size();
  vprint("Serialized message");

  // Creating header
  char hdr_send[12];
  std::ostringstream ss;
  ss << std::setw(12) << std::setfill('0') << length;
  ss.str().copy(hdr_send, 12);
  vprint("Created char array of length " + std::to_string(length));

  // Copy to char array
  char* to_send = new char[12 + length];
  for (int i = 0; i < 12; ++i) {
    to_send[i] = hdr_send[i];
  }

  for (int i = 0; i < length; ++i) {
    char val = request_s[i];
    to_send[i + 12] = val;
  }
  vprint("Copied to char array");

  // Send header with number of bytes
  if ((bytecount = send(_socket, (void *)to_send, 12 + length, 0)) == -1) {
    std::cerr << "Client -> Error sending data " << errno << std::endl;
  }

  vprint("Message sent");

  delete[] to_send;

  return true;
}

//! Retrieve the response from the server and store it in resp_wrapper, to be parsed
//! elsewhere.
bool DLClientComms::readInfoResponse(dlserver::RespondWrapper& resp_wrapper)
{
  int bytecount;

  // Read header first
  vprint("Reading header data");
  char buffer_hdr[12];
  if ((bytecount = recv(_socket, buffer_hdr, 12, 0)) == -1) {
    std::cerr << "Client -> Error receiving data " << std::endl;
  }
  google::protobuf::uint32 siz = readHdr(buffer_hdr);

  return readInfoResponse(siz, resp_wrapper);
}

//! Helper to the above 'readInfoResponse' function, pulls the data after
//! determining the size 'siz' from the header.
bool DLClientComms::readInfoResponse(google::protobuf::uint32 siz, dlserver::RespondWrapper& resp_wrapper)
{
  // Reading message data
  vprint("Reading data of size: " + std::to_string(siz));
  int bytecount;
  char buffer[siz];

  resp_wrapper.set_info(true);

  //Read the entire buffer
  if ((bytecount = recv(_socket, (void *)buffer, siz, 0)) == -1) {
    std::cerr << "Client -> Error receiving data " << errno << std::endl;
  }

  // Deserialize using protobuf functions
  vprint("Deserializing message");
  google::protobuf::io::ArrayInputStream ais(buffer, siz);
  google::protobuf::io::CodedInputStream coded_input(&ais);
  google::protobuf::io::CodedInputStream::Limit msgLimit = coded_input.PushLimit(siz);
  resp_wrapper.ParseFromCodedStream(&coded_input);
  coded_input.PopLimit(msgLimit);

  return false;
}

//! Sends a messaged image to to the server.
bool DLClientComms::sendInferenceRequest(dlserver::RequestInference* req_inference) {
  int bytecount;

  // Create message
  dlserver::RequestWrapper req_wrapper;
  req_wrapper.set_info(true);

  req_wrapper.set_allocated_r2(req_inference);

  // Serialize message
  std::string request_s;
  req_wrapper.SerializeToString(&request_s);
  int length = request_s.size();
  vprint("Serialized message");

  // Creating header
  char hdr_send[12];
  std::ostringstream ss;
  ss << std::setw(12) << std::setfill('0') << length;
  ss.str().copy(hdr_send, 12);
  vprint("Created char array of length " + std::to_string(length));

  // Copy to char array
  char* to_send = new char[12 + length];
  for (int i = 0; i < 12; ++i) {
    to_send[i] = hdr_send[i];
  }

  for (int i = 0; i < length; ++i) {
    char val = request_s[i];
    to_send[i + 12] = val;
  }
  vprint("Copied to char array");

  // Send header with number of bytes
  if ((bytecount = send(_socket, (void *)to_send, 12 + length, 0)) == -1) {
    std::cerr << "Client -> Error sending data " << errno << std::endl;
  }

  vprint("Message sent");

  delete[] to_send;

  return true;
}

//! Marshalls the returned image into a float buffer of the original image size. Note, this
//! expects the size of result to have been set to the same size as the image that was
//! previously sent to the server.
bool DLClientComms::readInferenceResponse(std::vector<float>& result)
{
  int bytecount;
  
  // Read header first
  vprint("Reading header data");
  char buffer_hdr[12];
  if ((bytecount = recv(_socket, buffer_hdr, 12, 0)) == -1) {
    std::cerr << "Client -> Error receiving data " << std::endl;
  }
  google::protobuf::uint32 siz = readHdr(buffer_hdr);

  return readInferenceResponse(siz, result);
}

//! Helper to the above 'readInferenceResponse' function, pulls the data after
//! determining the size 'siz' from the header.
bool DLClientComms::readInferenceResponse(google::protobuf::uint32 siz, std::vector<float>& result)
{
  vprint("Reading data of size: " + std::to_string(siz));
  dlserver::RespondWrapper resp_wrapper;
  resp_wrapper.set_info(true);

  // Read the buffer
  std::string output;
  char buffer[1024];
  int n;
  while ((errno = 0, (n = recv(_socket, buffer, sizeof(buffer), 0)) > 0) ||
         errno == EINTR) {
    if (n > 0) {
      output.append(buffer, n);
    }
  }

  if (n < 0) {
    std::cerr << "Client -> Error receiving data " << std::endl;
  }

  // Deserialize using protobuf functions
  vprint("Deserializing message");
  google::protobuf::io::ArrayInputStream ais(output.c_str(), siz);
  google::protobuf::io::CodedInputStream coded_input(&ais);
  google::protobuf::io::CodedInputStream::Limit msgLimit = coded_input.PushLimit(siz);
  resp_wrapper.ParseFromCodedStream(&coded_input);
  coded_input.PopLimit(msgLimit);

  const dlserver::Image &img = resp_wrapper.r2().image(0);

  const char* imdata = img.image().c_str();
  std::memcpy(&result[0], imdata, result.size() * sizeof(float));

  return false;
}

google::protobuf::uint32 DLClientComms::readHdr(char* buf)
{
  google::protobuf::uint32 size;
  char tmp[13];
  std::memcpy(tmp, buf, 12);
  tmp[12] = '\0';
  size = atoi(tmp);
  return size;
}

void* DLClientComms::get_in_addr(struct sockaddr* sa)
{
  if (sa->sa_family == AF_INET) {
    return &(((struct sockaddr_in *)sa)->sin_addr);
  }
  return &(((struct sockaddr_in6 *)sa)->sin6_addr);
}

void DLClientComms::vprint(std::string msg)
{
  if (_verbose) {
    std::cerr << "Client -> " << msg << std::endl;
  }
}
