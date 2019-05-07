// Copyright (c) 2019 The Foundry Visionmongers Ltd.  All Rights Reserved.
// This is strictly non-commercial.

#ifndef MLCLIENTCOMMS_H
#define MLCLIENTCOMMS_H

// Includes for sockets and protobuf
#include <netdb.h>
#include "message.pb.h"

using byte = unsigned char;


//! The Machine Learning (ML) Client plug-in connects Nuke to a Python server to apply ML models to images.
/*! This plug-in can connect to a server (given a host and port), which responds
    with a list of available Machine Learning (ML) models and options.
    On every /a renderStripe() call, the image and model options are sent from Nuke to the server,
    there the server can process the image by doing Machine Learning inference,
    finally the resulting image is sent back to Nuke.
*/
class MLClientComms
{
public:
  // Static consts
  static const int kNumberOfBytesHeaderSize;

  static const int kTimeout;
  static const int kMaxNumberOfTry;

  // Static non-conts
  static bool Verbose;

public:
  //! Constructor. Initialize user controls to their default values, then try to
  //! connect to the specified host / port. Following the c-tor, you can test for
  //! a valid connection by calling isConnected().
  MLClientComms(const std::string& hostStr, int port);

  //! Destructor. Tear down any existing connection.
  virtual ~MLClientComms();

public:
  // Public static methods for client-server communication

  //! Test if a given hostname is valid, returning true if it is, false otherwise
  static bool ValidateHostName(const std::string& hostStr);

  //! Print debug related information to std::cout, when ::Verbose is set to true.
  static void Vprint(std::string msg);

public:
  // Public methods for client-server communication

  //! Return whether this object is connected to the specified server.
  bool isConnected() const;

  //! Function for discovering & negotiating the available models and their parameters.
  //! Return true on success, false otherwise with the errorMsg filled in.
  bool sendInfoRequestAndReadInfoResponse(mlserver::RespondWrapper& responseWrapper, std::string& errorMsg);

  //! Function for performing the inference on a selected model.
  //! Return true on success, false otherwise with the errorMsg filled in.
  bool sendInferenceRequestAndReadInferenceResponse(mlserver::RequestInference& requestInference, mlserver::RespondWrapper& responseWrapper, std::string& errorMsg);

private:
  // Private client / server comms functions

  //! Try to connect to the server with the specified hostStr & port, by repeatedly
  //! calling setupConnection() below until a connection is made or times out. After it
  //! returns, you can test if it was successful by calling isConnected().
  void connectLoop();

  //! Create a socket to connect to the server specified by hostStr and port.
  //! Return true on success, false otherwise with a message filled in errorStr.
  bool setupConnection(std::string& errorStr);

  //! Request the server to return a future message about its models. This is used
  //! to instruct the server that it should set itself up.
  //! Return true on success, false otherwise.
  bool sendInfoRequest();

  //! Retrieve the response from the server and store it in responseWrapper, to be parsed
  //! elsewhere. Return true on success, false otherwise.
  bool readInfoResponse(mlserver::RespondWrapper& responseWrapper);

  //! Send a messaged image to to the server. Return true on success, false otherwise.
  bool sendInferenceRequest(mlserver::RequestInference& requestInference);

  //! Marshall the returned image into a float buffer of the original image size. Note, this
  //! expects the size of result to have been set to the same size as the image that was
  //! previously sent to the server. Return true on success, false otherwise.
  bool readInferenceResponse(mlserver::RespondWrapper& responseWrapper);

  //! Pull the data after determining the size 'siz' from the header.
  //! Helper to the above 'readInfoResponse' function.
  bool readInfoResponse(google::protobuf::uint32 siz, mlserver::RespondWrapper& responseWrapper);

  //! Pull the data after determining the size 'siz' from the header.
  //! Helper to the above 'readInferenceResponse' function.
  bool readInferenceResponse(google::protobuf::uint32 siz, mlserver::RespondWrapper& responseWrapper);

  //! Close the current connection if one is open.
  void closeConnection();

private:
  // Private helper functions
  google::protobuf::uint32 readHdr(char* buf);
  void* getInAddr(struct sockaddr* sa);

private:
  // Private member variables
  std::string _hostStr;
  int _port;

  bool _isConnected;
  int _socket;
};

#endif // MLCLIENTCOMMS_H
