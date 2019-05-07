// Copyright (c) 2019 The Foundry Visionmongers Ltd.  All Rights Reserved.
// This is strictly non-commercial.

// Includes for sockets and protobuf
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <cstring>
#include <sstream>
#include <arpa/inet.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "MLClientComms.h"

// Static consts

/*static*/ const int MLClientComms::kNumberOfBytesHeaderSize = 12;

/*static*/ const int MLClientComms::kTimeout = 500000;
/*static*/ const int MLClientComms::kMaxNumberOfTry = 5;

// Static non-const variables
/*static*/ bool MLClientComms::Verbose = true;

//! Constructor. Initialize user controls to their default values, then try to
//! connect to the specified host / port. Following the c-tor, you can test for
//! a valid connection by calling isConnected().
MLClientComms::MLClientComms(const std::string& hostStr, int port)
: _isConnected(false)
, _socket(0)
, _hostStr(hostStr)
, _port(port)
{
  // On construction, try to connect with the given host & port
  connectLoop();
}

//! Destructor. Tear down any existing connection.
MLClientComms::~MLClientComms()
{
  // On destruction, we need to close the socket and reset our connection variable
  closeConnection();
}

//! Test if a given hostname is valid, returning true if it is, false otherwise
/*static*/ bool MLClientComms::ValidateHostName(const std::string& hostStr)
{
  // Check if correct ipv4 or ipv6 addresses
  struct sockaddr_in sa;
  struct sockaddr_in6 sa6;
  bool isIPv4 = inet_pton(AF_INET, hostStr.c_str(), &(sa.sin_addr)) != 0;
  bool isIPv6 = inet_pton(AF_INET6, hostStr.c_str(), &(sa6.sin6_addr)) != 0;

  return isIPv4 || isIPv6;
}

//! Print debug related information to std::cout, when ::Verbose is set to true.
/*static*/ void MLClientComms::Vprint(std::string msg)
{
  if (MLClientComms::Verbose) {
    std::cerr << "Client -> " << msg << std::endl;
  }
}

//! Return whether this object is connected to the specified server.
bool MLClientComms::isConnected() const
{
  return _isConnected;
}

//! Function for discovering & negotiating the available models and their parameters.
//! Return true on success, false otherwise with the errorMsg filled in.
bool MLClientComms::sendInfoRequestAndReadInfoResponse(mlserver::RespondWrapper& responseWrapper, std::string& errorMsg)
{
  // Try connect if we haven't already
  connectLoop();
  if(!isConnected()) {
    errorMsg = "Unable to connect to server.";
    return false;
  }

  // Send the request for server & model info, and parse the response.
  if(!sendInfoRequest()) {
    errorMsg = "Error sending info request.";
    return false;
  }

  // Check that the comms read the response OK
  if(!readInfoResponse(responseWrapper)) {
    errorMsg = "Error reading info response.";
    return false;
  }
  // Check if error occured in the server
  if (responseWrapper.has_error()) {
    errorMsg = responseWrapper.error().msg();
    return false;
  }

  // Return true for success if we reached here
  return true;
}

bool MLClientComms::sendInferenceRequestAndReadInferenceResponse(mlserver::RequestInference& requestInference, mlserver::RespondWrapper& responseWrapper, std::string& errorMsg)
{
  // Try connect if we haven't already
  connectLoop();
  if(!isConnected()) {
    errorMsg = "Unable to connect to server.";
    return false;
  }

  // Send the inference request.
  if(!sendInferenceRequest(requestInference)) {
    errorMsg = "Error sending inference request.";
    return false;
  }

  // Await and process the response.
  if(!readInferenceResponse(responseWrapper)) {
    errorMsg = "Error reading inference response.";
    return false;
  }

  // Check if an error occured in the server
  if (responseWrapper.has_error()) {
    errorMsg = responseWrapper.error().msg();
    return false;
  }

  // Return true for success if we reached here
  return true;
}

//! Try to connect to the server with the specified _hostStr & _port. After it
//! returns, you can test if it was successful by calling isConnected().
void MLClientComms::connectLoop()
{
  // First test if there's an existing connected, if so return immediately.
  if ( isConnected() ) {
    return;
  }

  // Otherwise establish a new connection.
  int i = 0;
  std::string errorStr;
  while (!setupConnection(errorStr)) {
    usleep(kTimeout);
    i++;
    if (i >= kMaxNumberOfTry) {
      std::stringstream strStrm;
      strStrm << "Error setting up connection:" << std::endl;
      strStrm << "\t" << errorStr.c_str();
      MLClientComms::Vprint(strStrm.str());
      _isConnected = false;
      return;
    }
    std::stringstream strStrm;
    strStrm << "Failing to connect. Attempts: " << i;
    MLClientComms::Vprint(strStrm.str());
  }
  _isConnected = true;
}

//! Create a socket to connect to the server specified by _hostStr and _port
bool MLClientComms::setupConnection(std::string& errorStr)
{
  // This assumes there is no prior connection. Before calling this any existing connection
  // should be closed.
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
    status = getaddrinfo(_hostStr.c_str(), std::to_string(_port).c_str(), &hints, &aiResult);
    inet_ntop(aiResult->ai_family, getInAddr((struct sockaddr *)aiResult->ai_addr), s, sizeof s);

    std::stringstream strStrm;
    strStrm << "Trying to connect to " << s;
    MLClientComms::Vprint(strStrm.str());

    if (status != 0) {
      std::stringstream ss;
      ss << "getaddrinfo error: " << gai_strerror(status);
      errorStr = ss.str();
      MLClientComms::Vprint(errorStr);
      return false;
    }

    // Create Socket and check if error occured afterwards
    _socket = socket(aiResult->ai_family, aiResult->ai_socktype, aiResult->ai_protocol);
    if (_socket < 0) {
      std::stringstream ss;
      ss << "socket error: " << gai_strerror(_socket);
      errorStr = ss.str();
      MLClientComms::Vprint(errorStr);
      return false;
    }

    // Add socket option
    int enable = 1;
    if (setsockopt(_socket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
      errorStr = "setsockopt(SO_REUSEADDR) failed";
      MLClientComms::Vprint(errorStr);
      return false;
    }

    long socket_flags;
    // Set non-blocking connect socket
    if ((socket_flags = fcntl(_socket, F_GETFL, NULL)) < 0) { // get socket flag argument
      std::stringstream ss;
      ss << "socket error fcntl(..., F_GETFL) (" << strerror(errno) << ")";
      errorStr = ss.str();
      MLClientComms::Vprint(errorStr);
      return false;
    }
    socket_flags |= O_NONBLOCK; // add non-blocking flag to the socket flags
    if (fcntl(_socket, F_SETFL, socket_flags) < 0) { // update socket flags
      std::stringstream ss;
      ss << "socket error fcntl(..., F_SETFL) (" << strerror(errno) << ")";
      errorStr = ss.str();
      MLClientComms::Vprint(errorStr);
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
            MLClientComms::Vprint(errorStr);
            return false;
          }
        }
        else { // Unable to select socket
          return false;
        }
      }
      else {
        std::stringstream ss;
        ss << "socket error connecting " << errno << " " << strerror(errno);
        errorStr = ss.str();
        MLClientComms::Vprint(errorStr);
        return false;
      }
    }
    // Set to blocking mode again
    if ((socket_flags = fcntl(_socket, F_GETFL, NULL)) < 0) { // get socket flag argument
      std::stringstream ss;
      ss << "socket error fcntl(..., F_GETFL) (" << strerror(errno) << ")";
      errorStr = ss.str();
      MLClientComms::Vprint(errorStr);
      return false;
    }
    socket_flags &= (~O_NONBLOCK); // remove non-blocking flag from the socket
    if (fcntl(_socket, F_SETFL, socket_flags) < 0) { // update socket flags
      std::stringstream ss;
      ss << "socket error fcntl(..., F_SETFL) (" << strerror(errno) << ")";
      errorStr = ss.str();
      MLClientComms::Vprint(errorStr);
      return false;
    }

    inet_ntop(aiResult->ai_family, getInAddr((struct sockaddr *)aiResult->ai_addr), s, sizeof s);

    std::stringstream ss;
    ss << "Connected to " << s;
    MLClientComms::Vprint(ss.str());

    // Free the aiResult linked list after we are done with it
    freeaddrinfo(aiResult);
  }
  catch (const std::exception &e) {
    std::stringstream ss;
    ss << e.what();
    errorStr = ss.str();
    MLClientComms::Vprint(errorStr);
    return false;
  }
  return true;
}

//! Request the server to return a future message about its models. This is used
//! to instruct the server that it should set itself up.
bool MLClientComms::sendInfoRequest()
{
  int bytecount;
  MLClientComms::Vprint("Sending info request");

  // Create message
  mlserver::RequestWrapper requestWrapper;
  requestWrapper.set_info(true);
  mlserver::RequestInfo* requestInfo = new mlserver::RequestInfo;
  requestInfo->set_info(true);
  requestWrapper.set_allocated_r1(requestInfo);
  MLClientComms::Vprint("Created message");

  // Generate the data which should be sent over the network
  std::string requestStr;
  requestWrapper.SerializeToString(&requestStr);
  int length = requestStr.size();
  MLClientComms::Vprint("Serialized message");

  // Creating header
  char hdrSend[kNumberOfBytesHeaderSize];
  std::ostringstream ss;
  ss << std::setw(kNumberOfBytesHeaderSize) << std::setfill('0') << length;
  ss.str().copy(hdrSend, kNumberOfBytesHeaderSize);
  MLClientComms::Vprint("Created char array of length " + std::to_string(length));

  // Copy to char array
  std::vector<char> toSend(kNumberOfBytesHeaderSize + length);
  for (int i = 0; i < kNumberOfBytesHeaderSize; ++i) {
    toSend[i] = hdrSend[i];
  }

  for (int i = 0; i < length; ++i) {
    char val = requestStr[i];
    toSend[i + kNumberOfBytesHeaderSize] = val;
  }
  MLClientComms::Vprint("Copied to char array");

  // Send header with number of bytes
  if ((bytecount = send(_socket, (void *)&toSend[0], kNumberOfBytesHeaderSize + length, 0)) == -1) {
    std::stringstream ss;
    ss << "Error sending data " << errno;
    MLClientComms::Vprint(ss.str());
    return false;
  }

  MLClientComms::Vprint("Message sent");

  return true;
}

//! Retrieve the response from the server and store it in responseWrapper, to be parsed
//! elsewhere.
bool MLClientComms::readInfoResponse(mlserver::RespondWrapper& responseWrapper)
{
  int bytecount;

  // Read header first
  MLClientComms::Vprint("Reading header data");
  char hdrBuffer[kNumberOfBytesHeaderSize];
  if ((bytecount = recv(_socket, hdrBuffer, kNumberOfBytesHeaderSize, 0)) == -1) {
    MLClientComms::Vprint("Error receiving data.");
    return false;
  }
  google::protobuf::uint32 siz = readHdr(hdrBuffer);

  return readInfoResponse(siz, responseWrapper);
}

//! Pull the data after determining the size 'siz' from the header.
//! Helper to the above 'readInfoResponse' function.
bool MLClientComms::readInfoResponse(google::protobuf::uint32 siz, mlserver::RespondWrapper& responseWrapper)
{
  // Reading message data
  MLClientComms::Vprint("Reading data of size: " + std::to_string(siz));
  int bytecount;
  char buffer[siz];

  responseWrapper.set_info(true);

  // Read the entire buffer
  if ((bytecount = recv(_socket, (void *)buffer, siz, 0)) == -1) {
    std::stringstream ss;
    ss << "Error receiving data " << errno;
    MLClientComms::Vprint(ss.str());
    return false;
  }

  // Deserialize using protobuf functions
  MLClientComms::Vprint("Deserializing message");
  google::protobuf::io::ArrayInputStream ais(buffer, siz);
  google::protobuf::io::CodedInputStream codedInput(&ais);
  google::protobuf::io::CodedInputStream::Limit msgLimit = codedInput.PushLimit(siz);
  // Fill the message responseWrapper with a protocol buffer parsed from codedInput
  responseWrapper.ParseFromCodedStream(&codedInput);
  codedInput.PopLimit(msgLimit);

  // Return true on success
  return true;
}

//! Send a messaged image to to the server.
bool MLClientComms::sendInferenceRequest(mlserver::RequestInference& requestInference) {
  int bytecount;

  // Create message
  mlserver::RequestWrapper requestWrapper;
  requestWrapper.set_info(true);

  requestWrapper.set_allocated_r2(&requestInference);

  // Serialize message
  std::string requestStr;
  requestWrapper.SerializeToString(&requestStr);
  int length = requestStr.size();
  MLClientComms::Vprint("Serialized message");

  // Creating header
  char hdrSend[kNumberOfBytesHeaderSize];
  std::ostringstream ss;
  ss << std::setw(kNumberOfBytesHeaderSize) << std::setfill('0') << length;
  ss.str().copy(hdrSend, kNumberOfBytesHeaderSize);
  MLClientComms::Vprint("Created char array of length " + std::to_string(length));

  // Copy to char array
  std::vector<char> toSend(kNumberOfBytesHeaderSize + length);
  for (int i = 0; i < kNumberOfBytesHeaderSize; ++i) {
    toSend[i] = hdrSend[i];
  }

  for (int i = 0; i < length; ++i) {
    char val = requestStr[i];
    toSend[i + kNumberOfBytesHeaderSize] = val;
  }
  MLClientComms::Vprint("Copied to char array");

  // Send header with number of bytes
  if ((bytecount = send(_socket, (void *)&toSend[0], kNumberOfBytesHeaderSize + length, 0)) == -1) {
    isConnected();
    std::stringstream ss;
    ss << "Error sending data " << errno;
    MLClientComms::Vprint(ss.str());
    return false;
  }

  MLClientComms::Vprint("Message sent");

  // Return true on success
  return true;
}

//! Marshall the returned image into a float buffer of the original image size. Note, this
//! expects the size of result to have been set to the same size as the image that was
//! previously sent to the server.
bool MLClientComms::readInferenceResponse(mlserver::RespondWrapper& responseWrapper)
{
  int bytecount;
  
  // Read header first
  MLClientComms::Vprint("Reading header data");
  char hdrBuffer[kNumberOfBytesHeaderSize];
  if ((bytecount = recv(_socket, hdrBuffer, kNumberOfBytesHeaderSize, 0)) == -1) {
    MLClientComms::Vprint("Error receiving data.");
    return false;
  }
  google::protobuf::uint32 siz = readHdr(hdrBuffer);

  return readInferenceResponse(siz, responseWrapper);
}

//! Pull the data after determining the size 'siz' from the header.
//! Helper to the above 'readInferenceResponse' function.
bool MLClientComms::readInferenceResponse(google::protobuf::uint32 siz, mlserver::RespondWrapper& responseWrapper)
{
  MLClientComms::Vprint("Reading data of size: " + std::to_string(siz));
  responseWrapper.set_info(true);

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
    MLClientComms::Vprint("Error receiving data.");
    return false;
  }

  // Deserialize using protobuf functions
  MLClientComms::Vprint("Deserializing message");
  google::protobuf::io::ArrayInputStream ais(output.c_str(), siz);
  google::protobuf::io::CodedInputStream codedInput(&ais);
  google::protobuf::io::CodedInputStream::Limit msgLimit = codedInput.PushLimit(siz);
  responseWrapper.ParseFromCodedStream(&codedInput);
  codedInput.PopLimit(msgLimit);

  // Return true on success
  return true;
}

//! Close the current connection if one is open.
void MLClientComms::closeConnection()
{
  // Check if a valid socket is open
  if(_socket) {
    // If so, close & reset the state variables
    close(_socket);
    _socket = 0;
    MLClientComms::Vprint("Closed connection\n"
      "-----------------------------------------------");
  }
  _isConnected = false;
}

google::protobuf::uint32 MLClientComms::readHdr(char* buf)
{
  google::protobuf::uint32 size;
  char tmp[kNumberOfBytesHeaderSize+1];
  std::memcpy(tmp, buf, kNumberOfBytesHeaderSize);
  tmp[kNumberOfBytesHeaderSize] = '\0';
  size = atoi(tmp);
  return size;
}

void* MLClientComms::getInAddr(struct sockaddr* sa)
{
  if (sa->sa_family == AF_INET) {
    return &(((struct sockaddr_in *)sa)->sin_addr);
  }
  return &(((struct sockaddr_in6 *)sa)->sin6_addr);
}