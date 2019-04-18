// Copyright (c) 2019 The Foundry Visionmongers Ltd.  All Rights Reserved.
// This is strictly non-commercial.

#ifndef DLCLIENTCOMMS_H
#define DLCLIENTCOMMS_H

// Includes for sockets and protobuf
#include <netdb.h>
#include "message.pb.h"

using byte = unsigned char;


//! The Deep Learning (DL) Client plug-in connects Nuke to a Python server to apply DL models to images.
/*! This plug-in can connect to a server (given a host and port), which responds
    with a list of available Deep Learning (DL) models and options.
    On every /a engine() call, the image and model options are sent from Nuke to the server,
    there the server can process the image by doing Deep Learning inference,
    finally the resulting image is sent back to Nuke.
*/
class DLClientComms
{
public:
  //! Constructor. Initialize user controls to their default values.
  DLClientComms();
  virtual ~DLClientComms();

public:
  // Public getter / setter functions

  //! Returns whether this object is connected to the specified server.
  bool isConnected() const;

public:
  // Public methods for client-server communication

  //! Tests if a given hostname is valid, returning true if it is, false otherwise
  bool validateHostName(const std::string& hostStr);

  //! Tries to connect to the server with the specified hostStr & port. After it
  //! returns, you can test if it was successful by calling isConnected().
  void connectLoop(const std::string& hostStr, int port);

  //! Requests the server to return a future message about its models. This is used
  //! to instruct the server that it should set itself up.
  bool sendInfoRequest();

  //! Retrieve the response from the server and store it in resp_wrapper, to be parsed
  //! elsewhere.
  bool readInfoResponse(dlserver::RespondWrapper& resp_wrapper);

  //! Sends a messaged image to to the server.
  bool sendInferenceRequest(dlserver::RequestInference* req_inference);

  //! Marshalls the returned image into a float buffer of the original image size. Note, this
  //! expects the size of result to have been set to the same size as the image that was
  //! previously sent to the server.
  bool readInferenceResponse(dlserver::RespondWrapper& resp_wrapper);

private:
  // Private client / server comms functions

  //! Create a socket to connect to the server specified by hostStr and port. Returns
  //! true on success, false otherwise with a message filled in errorStr.
  bool setupConnection(const std::string& hostStr, int port, std::string& errorStr);

  //! Helper to the above 'readInfoResponse' function, pulls the data after
  //! determining the size 'siz' from the header.
  bool readInfoResponse(google::protobuf::uint32 siz, dlserver::RespondWrapper& resp_wrapper);

  //! Helper to the above 'readInferenceResponse' function, pulls the data after
  //! determining the size 'siz' from the header.
  bool readInferenceResponse(google::protobuf::uint32 siz, dlserver::RespondWrapper& resp_wrapper);

private:
  // Private helper functions
  google::protobuf::uint32 readHdr(char* buf);
  void* get_in_addr(struct sockaddr* sa);
  void vprint(std::string msg);

private:
  // Private member variables
  bool _isConnected;
  int _socket;
  bool _verbose;
};

#endif // DLCLIENTCOMMS_H
