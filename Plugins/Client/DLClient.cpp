// Copyright (c) 2018 The Foundry Visionmongers Ltd.  All Rights Reserved.

// Includes for sockets and protobuf
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <arpa/inet.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "DLClient.h"

using namespace DD::Image;

/*! This is a function that creates an instance of the operator, and is
   needed for the Iop::Description to work.
 */
static Iop* DLClientCreate(Node* node)
{
  return new DLClient(node);
}

/*! The Iop::Description is how NUKE knows what the name of the operator is,
   how to create one, and the menu item to show the user. The menu item may be
   0 if you do not want the operator to be visible.
 */
const Iop::Description DLClient::description(CLASS, "Merge/DLClient",
                                                        DLClientCreate);

//! Constructor. Initialize user controls to their default values.
DLClient::DLClient(Node* node)
: DD::Image::Iop(node)
, _firstTime(true)
, _isConnected(false)
, _host("172.17.0.2")
, _hostIsValid(true)
, _port(55555)
, _portIsValid(true)
, _chosenModel(0)
, _modelSelected(false)
, _showDynamic(false)
, _numNewKnobs(0)
, _verbose(true)
{ }

DLClient::~DLClient() {}

//! The maximum number of input connections the operator can have.
int DLClient::maximum_inputs() const
{
  if (_modelSelected) {
    return _numInputs[_chosenModel];
  }
  else {
    return 1;
  }
}

//! The minimum number of input connections the operator can have.
int DLClient::minimum_inputs() const
{ 
  if (_modelSelected) {
    return _numInputs[_chosenModel];
  }
  else {
    return 1; 
  }
}

/*! Return the text Nuke should draw on the arrow head for input \a input
    in the DAG window. This should be a very short string, one letter
    ideally. Return null or an empty string to not label the arrow.
*/
const char* DLClient::input_label(int input, char* buffer) const
{
  if (!_modelSelected) {
    return "";
  }
  else {
    if ((input < _inputNames[_chosenModel].size()) && (_chosenModel < _inputNames.size())) {
      return _inputNames[_chosenModel][input].c_str();
    }
    else {
      return "";
    }
  }
}

void DLClient::_validate(bool for_real)
{
  copy_info(); // copy bbox channels etc from input0, which will validate it.
}

void DLClient::_request(int x, int y, int r, int t, ChannelMask channels, int count)
{
  // request all input input as we are going to search the whole input area
  // ChannelSet readChannels = input0().info().channels();
  // input(0)->request( readChannels, count );
  for (int i = 0; i < getInputs().size(); i++) {
    ChannelSet readChannels = input(i)->info().channels();
    input(i)->request(readChannels, count);
  }
}

void DLClient::_open()
{
  _firstTime = true;
}

/*! For each line in the area passed to request(), this will be called. It must
   calculate the image data for a region at vertical position \a y, and between
   horizontal positions \a x and \a r, and write it to the passed row
   structure. Usually this works by asking the input for data, and modifying
   it.
 */
void DLClient::engine(int y, int x, int r,
                      ChannelMask channels, Row &row)
{
  Format masterFormat = input(0)->format();

  const int masterFx = masterFormat.x();
  const int masterFy = masterFormat.y();
  const int masterFr = masterFormat.r();
  const int masterFt = masterFormat.t();

  {
    Guard guard(_lock);
    if (_firstTime) {
      _inputs.resize(getInputs().size());
      _w.resize(getInputs().size());
      _h.resize(getInputs().size());
      _c.resize(getInputs().size());
      _result.resize(masterFr * masterFt * 3);
      for (int i = 0; i < getInputs().size(); i++) {
        Format format = input(i)->format();

        const int fx = format.x();
        const int fy = format.y();
        const int fr = format.r();
        const int ft = format.t();

        ChannelSet readChannels = input(i)->info().channels();

        Interest interest(*input(i), fx, fy, fr, ft, readChannels, true);
        interest.unlock();

        _inputs[i].resize(fr * ft * 3);
        _w[i] = fr;
        _h[i] = ft;
        float* fullImagePtr = &_inputs[i][0];

        for (int ry = fy; ry < ft; ry++) {
          progressFraction(ry, ft - fy);
          Row row(fx, fr);
          row.get(*input(i), ry, fx, fr, readChannels);
          if (aborted()) {
            return;
          }
          foreach (z, readChannels) {
            if (strcmp(getLayerName(z), "rgb") == 0) {
              const float* CUR = row[z] + fx;
              const float* END = row[z] + fr;
              int currentPos = 0;
              while (CUR < END) {
                int fullPos = colourIndex(z) * fr * ft + ry * fr + currentPos++;
                fullImagePtr[fullPos] = (*CUR++);
              }
            }
          }
        }
      }
      _firstTime = false;
      if (_isConnected && _modelSelected) {
        processImage();
      }
    }
    Row in(x, r);
    in.get(*input(0), y, x, r, channels);
    if (aborted()) {
      return;
    }
    float* fullImagePtr = &_result[0];

    foreach (z, channels) {
      float* CUR = row.writable(z) + x;
      const float* inptr = in[z] + x;
      const float* END = row[z] + r;
      int currentPos = x;
      if (strcmp(getLayerName(z), "rgb") == 0) {
        while (CUR < END) {
          int fullPos = colourIndex(z) * masterFr * masterFt + y * masterFr + currentPos++;
          *CUR++ = fullImagePtr[fullPos];
        }
      }
      else {
        while (CUR < END) {
          *CUR++ = *inptr++;
        }
      }
    }
  }
}

void DLClient::vprint(std::string msg)
{
  if (_verbose) {
    std::cerr << "Client -> " << msg << std::endl;
  }
}

void* get_in_addr(struct sockaddr* sa)
{
  if (sa->sa_family == AF_INET) {
    return &(((struct sockaddr_in *)sa)->sin_addr);
  }
  return &(((struct sockaddr_in6 *)sa)->sin6_addr);
}

//! Create a socket to connect to the server specified by _host and _port
bool DLClient::setupConnection()
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
    status = getaddrinfo(_host.c_str(), std::to_string(_port).c_str(), &hints, &aiResult);
    inet_ntop(aiResult->ai_family, get_in_addr((struct sockaddr *)aiResult->ai_addr), s, sizeof s);
    if (_verbose) {
      std::cerr << "Client -> Trying to connect to " << s << std::endl;
    }
    if (status != 0) {
      std::cerr << "Client -> getaddrinfo error: " << gai_strerror(status) << std::endl;
      return 0;
    }

    // Create Socket and check if error occured afterwards
    _socket = socket(aiResult->ai_family, aiResult->ai_socktype, aiResult->ai_protocol);
    if (_socket < 0) {
      std::cerr << "Client -> socket error: " << gai_strerror(_socket) << std::endl;
      return 0;
    }

    // Add socket option
    int enable = 1;
    if (setsockopt(_socket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
      error("setsockopt(SO_REUSEADDR) failed");
    }

    long socket_flags;
    // Set non-blocking connect socket
    if ((socket_flags = fcntl(_socket, F_GETFL, NULL)) < 0) { // get socket flag argument
      std::cerr << "Client -> socket error fcntl(..., F_GETFL) (" << strerror(errno) << ")" << std::endl;
      return 0;
    }
    socket_flags |= O_NONBLOCK; // add non-blocking flag to the socket flags
    if (fcntl(_socket, F_SETFL, socket_flags) < 0) { // update socket flags
      std::cerr << "Client -> socket error fcntl(..., F_SETFL) (" << strerror(errno) << ")" << std::endl;
      return 0;
    }

    // Connect to the server using the socket
    status = connect(_socket, aiResult->ai_addr, aiResult->ai_addrlen);

    int valopt;
    fd_set myset;
    struct timeval tv;
    // Trying to connect with timeout
    if (status < 0) {
      if (errno == EINPROGRESS) {
        tv.tv_sec = 0.25; // timeout in seconds to wait before failing to connect to host and port
        tv.tv_usec = 0;
        // Re-enable file descriptors fd that were cleared after last select() return
        FD_ZERO(&myset);
        FD_SET(_socket, &myset);
        status = select(_socket + 1, NULL, &myset, NULL, &tv);
        if (status > 0) {
          // Socket selected for write
          getsockopt(_socket, SOL_SOCKET, SO_ERROR, (void *)(&valopt), &aiResult->ai_addrlen);
          if (valopt) {
            std::cerr << "Error in socket connection " << valopt << " - " << strerror(valopt) << std::endl;
            return 0;
          }
        }
        else { // Unable to select socket
          return 0;
        }
      }
      else {
        std::cerr << "Client -> socket error connecting " << errno << " " << strerror(errno) << std::endl;
        return 0;
      }
    }
    // Set to blocking mode again
    if ((socket_flags = fcntl(_socket, F_GETFL, NULL)) < 0) { // get socket flag argument
      std::cerr << "Client -> socket error fcntl(..., F_GETFL) (" << strerror(errno) << ")" << std::endl;
      return 0;
    }
    socket_flags &= (~O_NONBLOCK); // remove non-blocking flag from the socket
    if (fcntl(_socket, F_SETFL, socket_flags) < 0) { // update socket flags
      std::cerr << "Client -> socket error fcntl(..., F_SETFL) (" << strerror(errno) << ")" << std::endl;
      return 0;
    }

    inet_ntop(aiResult->ai_family, get_in_addr((struct sockaddr *)aiResult->ai_addr), s, sizeof s);
    if (_verbose) {
      std::cerr << "Client -> Connected to " << s << std::endl;
    }

    // Free the aiResult linked list after we are done with it
    freeaddrinfo(aiResult);
  }
  catch (const std::exception &e) {
    std::cerr << e.what();
    return 0;
  }
  return 1;
}

void DLClient::connectLoop()
{
  const int kTimeout = 500000;
  const int kMaxNumberOfTry = 5;
  int i = 0;
  while (!setupConnection()) {
    usleep(kTimeout);
    i++;
    if (i >= kMaxNumberOfTry) {
      std::cerr << "Client -> Error setting up connection" << std::endl;
      vprint("-----------------------------------------------");
      _isConnected = false;
      return;
    }
    std::cerr << "Client -> Failing to connect. Attempts: " << i << std::endl;
  }
  _isConnected = true;
}

bool DLClient::processImage()
{
  try {
    connectLoop();
    sendInferenceRequest();
    readInferenceResponse();
    vprint("-----------------------------------------------");
  }
  catch (...) {
    std::cerr << "Client -> Error receiving message" << std::endl;
  }
  return 0;
}

google::protobuf::uint32 DLClient::readHdr(char* buf)
{
  google::protobuf::uint32 size;
  char tmp[13];
  memcpy(tmp, buf, 12);
  tmp[12] = '\0';
  size = atoi(tmp);
  return size;
}

bool DLClient::sendInfoRequest()
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

bool DLClient::readInfoResponse()
{
  int bytecount;

  // Read header first
  vprint("Reading header data");
  char buffer_hdr[12];
  if ((bytecount = recv(_socket, buffer_hdr, 12, 0)) == -1) {
    std::cerr << "Client -> Error receiving data " << std::endl;
  }
  google::protobuf::uint32 siz = readHdr(buffer_hdr);

  return readInfoResponse(siz);
}

bool DLClient::readInfoResponse(google::protobuf::uint32 siz)
{
  // Reading message data
  vprint("Reading data of size: " + std::to_string(siz));
  int bytecount;
  char buffer[siz];
  dlserver::RespondWrapper resp_wrapper;
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

  // Parse message and fill in menu items for enumeration knob
  _serverModels.clear();
  _numInputs.clear();
  _inputNames.clear();
  std::vector<std::string> modelNames;
  int numModels = resp_wrapper.r1().nummodels();
  vprint("Server can serve " + std::to_string(numModels) + " models");
  vprint("-----------------------------------------------");
  for (int i = 0; i < numModels; i++) {
    dlserver::Model m;
    m = resp_wrapper.r1().models()[i];
    modelNames.push_back(m.label());
    _serverModels.push_back(m);
    _numInputs.push_back(m.inputs_size());
    std::vector<std::string> names;
    for (int j = 0; j < m.inputs_size(); j++) {
      dlserver::ImagePrototype p;
      p = m.inputs()[j];
      names.push_back(p.name());
    }
    _inputNames.push_back(names);
  }

  // Change enumeration knob choices
  Enumeration_KnobI* pSelectModelEnum = _selectedModelknob->enumerationKnob();
  pSelectModelEnum->menu(modelNames);

  if (_chosenModel >= (int)numModels) {
    _selectedModelknob->set_value(0);
  }

  _modelSelected = true;
  _showDynamic = true;

  return false;
}

bool DLClient::sendInferenceRequest() {
  int bytecount;
  vprint("Sending inference request for model \"" + _serverModels[_chosenModel].name() + "\"");

  // Create message
  dlserver::RequestWrapper req_wrapper;
  req_wrapper.set_info(true);
  dlserver::RequestInference* req_inference = new dlserver::RequestInference;

  dlserver::Model* m = new dlserver::Model(_serverModels[_chosenModel]);
  updateOptions(m);
  req_inference->set_allocated_model(m);

  // Parse image. TODO: Check for multiple inputs, different channel size
  for (int i = 0; i < getInputs().size(); i++) {
    dlserver::Image* image = req_inference->add_image();
    image->set_width(_w[i]);
    image->set_height(_h[i]);
    image->set_channels(3);

    int size = _inputs[i].size() * sizeof(float);
    byte* bytes = new byte[size];
    std::memcpy(bytes, _inputs[i].data(), size);
    image->set_image(bytes, size);
    delete[] bytes;
  }

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

bool DLClient::readInferenceResponse()
{
  int bytecount;
  
  // Read header first
  vprint("Reading header data");
  char buffer_hdr[12];
  if ((bytecount = recv(_socket, buffer_hdr, 12, 0)) == -1) {
    std::cerr << "Client -> Error receiving data " << std::endl;
  }
  google::protobuf::uint32 siz = readHdr(buffer_hdr);

  return readInferenceResponse(siz);
}

bool DLClient::readInferenceResponse(google::protobuf::uint32 siz)
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
  std::memcpy(&_result[0], imdata, _result.size() * sizeof(float));

  return false;
}

void DLClient::parseOptions()
{
  dlserver::Model m = _serverModels[_chosenModel];
  _dynamicBoolValues.clear();
  _dynamicIntValues.clear();
  _dynamicFloatValues.clear();
  _dynamicStringValues.clear();

  _dynamicBoolNames.clear();
  _dynamicIntNames.clear();
  _dynamicFloatNames.clear();
  _dynamicStringNames.clear();

  for (int i = 0; i < m.booloptions_size(); i++) {
    dlserver::BoolOption o;
    o = m.booloptions()[i];
    if (o.value()) {
      _dynamicBoolValues.push_back(1);
    }
    else {
      _dynamicBoolValues.push_back(0);
    }
    _dynamicBoolNames.push_back(o.name());
  }

  for (int i = 0; i < m.intoptions_size(); i++) {
    dlserver::IntOption o;
    o = m.intoptions()[i];
    _dynamicIntValues.push_back(o.value());
    _dynamicIntNames.push_back(o.name());
  }

  for (int i = 0; i < m.floatoptions_size(); i++) {
    dlserver::FloatOption o;
    o = m.floatoptions()[i];
    _dynamicFloatValues.push_back(o.value());
    _dynamicFloatNames.push_back(o.name());
  }

  for (int i = 0; i < m.stringoptions_size(); i++) {
    dlserver::StringOption o;
    o = m.stringoptions()[i];
    _dynamicStringValues.push_back(o.value());
    _dynamicStringNames.push_back(o.name());
  }
}

void DLClient::updateOptions(dlserver::Model* model)
{
  model->clear_booloptions();
  for (int i = 0; i < _dynamicBoolValues.size(); i++) {
    ::dlserver::BoolOption* opt = model->add_booloptions();
    opt->set_name(_dynamicBoolNames[i]);
    opt->set_value(_dynamicBoolValues[i]);
  }

  model->clear_intoptions();
  for (int i = 0; i < _dynamicIntValues.size(); i++) {
    ::dlserver::IntOption* opt = model->add_intoptions();
    opt->set_name(_dynamicIntNames[i]);
    opt->set_value(_dynamicIntValues[i]);
  }

  model->clear_floatoptions();
  for (int i = 0; i < _dynamicFloatValues.size(); i++) {
    ::dlserver::FloatOption* opt = model->add_floatoptions();
    opt->set_name(_dynamicFloatNames[i]);
    opt->set_value(_dynamicFloatValues[i]);
  }

  model->clear_stringoptions();
  for (int i = 0; i < _dynamicStringValues.size(); i++) {
    ::dlserver::StringOption* opt = model->add_stringoptions();
    opt->set_name(_dynamicStringNames[i]);
    opt->set_value(_dynamicStringValues[i]);
  }
}

void DLClient::addDynamicKnobs(void* p, Knob_Callback f)
{
  if (((DLClient *)p)->getShowDynamic()) {
    for (int i = 0; i < ((DLClient *)p)->getNumOfInts(); i++) {
      std::string name = ((DLClient *)p)->getDynamicIntName(i);
      std::string label = ((DLClient *)p)->getDynamicIntName(i);
      Int_knob(f, ((DLClient *)p)->getDynamicIntValue(i), name.c_str(), label.c_str());
      Newline(f, " ");
    }
    for (int i = 0; i < ((DLClient *)p)->getNumOfFloats(); i++) {
      std::string name = ((DLClient *)p)->getDynamicFloatName(i);
      std::string label = ((DLClient *)p)->getDynamicFloatName(i);
      Float_knob(f, ((DLClient *)p)->getDynamicFloatValue(i), name.c_str(), label.c_str());
      ClearFlags(f, Knob::SLIDER);
      Newline(f, " ");
    }
    for (int i = 0; i < ((DLClient *)p)->getNumOfBools(); i++) {
      std::string name = ((DLClient *)p)->getDynamicBoolName(i);
      std::string label = ((DLClient *)p)->getDynamicBoolName(i);
      Bool_knob(f, ((DLClient *)p)->getDynamicBoolValue(i), name.c_str(), label.c_str());
      Newline(f, " ");
    }
    for (int i = 0; i < ((DLClient *)p)->getNumOfStrings(); i++) {
      std::string name = ((DLClient *)p)->getDynamicStringName(i);
      std::string label = ((DLClient *)p)->getDynamicStringName(i);
      String_knob(f, ((DLClient *)p)->getDynamicStringValue(i), name.c_str(), label.c_str());
      Newline(f, " ");
    }
  }
}

void DLClient::knobs(Knob_Callback f)
{
  String_knob(f, &_host, "host");
  Int_knob(f, &_port, "port");
  Button(f, "connect", "Connect");
  Divider(f, "  ");
  static const char* static_choices[] = {
      0};
  Knob* knob = Enumeration_knob(f, &_chosenModel, static_choices, "models", "Models");
  if (knob) {
    _selectedModelknob = knob;
  }
  SetFlags(f, Knob::SAVE_MENU);

  if (!f.makeKnobs()) {
    DLClient::addDynamicKnobs(this->firstOp(), f);
  }
}

int DLClient::knob_changed(Knob* knobChanged)
{
  if (knobChanged->is("host")) {
    struct sockaddr_in sa;
    struct sockaddr_in6 sa6;
    bool is_ipv4 = inet_pton(AF_INET, _host.c_str(), &(sa.sin_addr)) != 0;
    bool is_ipv6 = inet_pton(AF_INET6, _host.c_str(), &(sa6.sin6_addr)) != 0;
    // check if correct ipv4 or ipv6 addresses
    if (!is_ipv4 && !is_ipv6) {
      error("Please insert a valid host ipv4 or ipv6 address.");
      _hostIsValid = false;
    }
    else {
      _hostIsValid = true;
    }
    return 1;
  }

  if (knobChanged->is("port")) {
    if (_port > 65535 || _port < 0) {
      error("Port out of range.");
      _portIsValid = false;
    }
    else {
      _portIsValid = true;
    }
    return 1;
  }

  if (knobChanged->is("connect")) {
    if (_portIsValid && _hostIsValid) {
      connectLoop();
      if (!_isConnected) {
        error("Could not connect to python server.");
      }
      else {
        sendInfoRequest();
        readInfoResponse();
        _firstTime = true;
        parseOptions();
        _numNewKnobs = replace_knobs(knob("models"), _numNewKnobs, addDynamicKnobs, this->firstOp());
      }
    }
    return 1;
  }

  if (knobChanged->is("models")) {
    parseOptions();
    _numNewKnobs = replace_knobs(knob("models"), _numNewKnobs, addDynamicKnobs, this->firstOp());
    return 1;
  }
  return 0;
}

//! Return the name of the class.
const char* DLClient::Class() const
{ 
  return CLASS;
}

const char* DLClient::node_help() const
{ 
  return HELP;
}
  
int DLClient::getNumOfFloats() const
{ 
  return _dynamicFloatValues.size();
}

int DLClient::getNumOfInts() const
{ 
  return _dynamicIntValues.size();
}

int DLClient::getNumOfBools() const
{ 
  return _dynamicBoolValues.size();
}

int DLClient::getNumOfStrings() const
{ 
  return _dynamicStringValues.size();
}

std::string DLClient::getDynamicBoolName(int idx)
{ 
  return _dynamicBoolNames[idx];
}

std::string DLClient::getDynamicFloatName(int idx)
{ 
  return _dynamicFloatNames[idx];
}

std::string DLClient::getDynamicIntName(int idx)
{ 
  return _dynamicIntNames[idx];
}

std::string DLClient::getDynamicStringName(int idx)
{ 
  return _dynamicStringNames[idx];
}

float* DLClient::getDynamicFloatValue(int idx)
{ 
  return &_dynamicFloatValues[idx];
}

int* DLClient::getDynamicIntValue(int idx)
{ 
  return &_dynamicIntValues[idx];
}

bool* DLClient::getDynamicBoolValue(int idx)
{ 
  return (bool* )&_dynamicBoolValues[idx];
}

std::string* DLClient::getDynamicStringValue(int idx)
{ 
  return &_dynamicStringValues[idx];
}

bool DLClient::getShowDynamic() const
{ 
  return _showDynamic && _isConnected;
}