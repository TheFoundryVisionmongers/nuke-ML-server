// Copyright (c) 2018 The Foundry Visionmongers Ltd.  All Rights Reserved.

#include <cstring>

#include "DLClient.h"

const char* const DLClient::kClassName = "DLClient";
const char* const DLClient::kHelpString = "Connects to a Python server for Deep Learning inference.";

const char* const DLClient::kDefaultHostName = "172.17.0.2";
const int         DLClient::kDefaultPortNumber = 55555;

const int DLClient::kDefaultNumberOfChannels = 3;

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
const Iop::Description DLClient::description(DLClient::kClassName, 0, DLClientCreate);

//! Constructor. Initialize user controls to their default values.
DLClient::DLClient(Node* node)
: DD::Image::PlanarIop(node)
, _host(DLClient::kDefaultHostName)
, _hostIsValid(true)
, _port(DLClient::kDefaultPortNumber)
, _portIsValid(true)
, _chosenModel(0)
, _modelSelected(false)
, _showDynamic(false)
, _numNewKnobs(0)
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

bool DLClient::useStripes() const
{
  return false;
}

bool DLClient::renderFullPlanes() const
{
  return true;
}

void DLClient::getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput &reqData) const
{
    // request all input input as we are going to search the whole input area
  for (int i = 0, endI = getInputs().size(); i < endI; i++) {
    const ChannelSet readChannels = input(i)->info().channels();
    input(i)->request(readChannels, count);
  }
}

void DLClient::renderStripe(ImagePlane& imagePlane)
{
  if (aborted() || cancelled()) {
    return;
  }
  input0().fetchPlane(imagePlane);
  imagePlane.makeUnique();

  initBuffers(imagePlane);
  if (_comms.isConnected() && _modelSelected) {
    processImage(_host, _port);
  }
  renderOutputBuffer(imagePlane);
}

void DLClient::initBuffers(ImagePlane& imagePlane)
{
  const Box box = imagePlane.bounds();
  const int numberOfInputs = getInputs().size();
  _inputs.resize(numberOfInputs);
  _w.resize(numberOfInputs);
  _h.resize(numberOfInputs);
  _c.resize(numberOfInputs);
  _result.resize(box.w() * box.h() * kDefaultNumberOfChannels);
  initInputs(imagePlane);
}

void DLClient::initInputs(ImagePlane &imagePlane)
{
  for (int i = 0, endI = getInputs().size(); i < endI; i++) {
    initInput(i, imagePlane);
  }
}

void DLClient::initInput(int i, ImagePlane &imagePlane)
{
  const Box box = imagePlane.bounds();
  const int fx = box.x();
  const int fy = box.y();
  const int fr = box.r();
  const int ft = box.t();

  _inputs[i].resize(box.w() * box.h() * kDefaultNumberOfChannels);
  _w[i] = fr;
  _h[i] = ft;
  float* fullImagePtr = &_inputs[i][0];

  for (int z = 0; z < kDefaultNumberOfChannels; z++) {
    for(int ry = fy; ry < ft; ry++) {
      progressFraction(ry, ft - fy);
      if (aborted()) {
        return;
      }
      ImageTileReadOnlyPtr tile = imagePlane.readableAt(ry, z);
      int currentPos = 0;
      for(int rx = fx; rx < fr; rx++) {
        int fullPos = z * fr * ft + ry * fr + currentPos++;
        fullImagePtr[fullPos] = tile[rx];
      }
    }
  }
}

void DLClient::renderOutputBuffer(ImagePlane& imagePlane)
{
  const Box box = imagePlane.bounds();
  const int fx = box.x();
  const int fy = box.y();
  const int fr = box.r();
  const int ft = box.t();

  float* fullImagePtr = &_result[0];
  for (int z = 0; z < kDefaultNumberOfChannels; z++) {
    for(int ry = fy; ry < ft; ry++) {
      if (aborted()) {
        return;
      }
      int currentPos = 0;
      for(int rx = fx; rx < fr; rx++) {
        int fullPos = z * fr * ft + ry * fr + currentPos++;
        imagePlane.writableAt(rx, ry, z)  = fullImagePtr[fullPos];
      }
    }
  }
}

bool DLClient::processImage(const std::string& hostStr, int port)
{
  try {
    _comms.connectLoop(hostStr, port);
    if (!_comms.isConnected()) {
      error("Could not connect to python server.");
    }
    std::cerr << "Sending inference request for model \"" << _serverModels[_chosenModel].name() << "\"" << std::endl;

    // Create inference message
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

    _comms.sendInferenceRequest(req_inference);
    dlserver::RespondWrapper resp_wrapper;
    _comms.readInferenceResponse(resp_wrapper);
    // Check if error occured in the server
    if (resp_wrapper.has_error()) {
      const char * errorMsg = resp_wrapper.error().msg().c_str();
      error(errorMsg);
    }
    // Get the resulting image data
    const dlserver::Image &img = resp_wrapper.r2().image(0);
    const char* imdata = img.image().c_str();
    std::memcpy(&_result[0], imdata, _result.size() * sizeof(float));
  }
  catch (...) {
    std::cerr << "Client -> Error receiving message" << std::endl;
  }
  return 0;
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

  for (int i = 0, endI = m.bool_options_size(); i < endI; i++) {
    dlserver::BoolOption o;
    o = m.bool_options(i);
    if (o.value()) {
      _dynamicBoolValues.push_back(1);
    }
    else {
      _dynamicBoolValues.push_back(0);
    }
    _dynamicBoolNames.push_back(o.name());
  }
  for (int i = 0, endI = m.int_options_size(); i < endI; i++) {
    dlserver::IntOption o;
    o = m.int_options(i);
    _dynamicIntValues.push_back(o.value());
    _dynamicIntNames.push_back(o.name());
  }
  for (int i = 0, endI = m.float_options_size(); i < endI; i++) {
    dlserver::FloatOption o;
    o = m.float_options(i);
    _dynamicFloatValues.push_back(o.value());
    _dynamicFloatNames.push_back(o.name());
  }
  for (int i = 0, endI = m.string_options_size(); i < endI; i++) {
    dlserver::StringOption o;
    o = m.string_options(i);
    _dynamicStringValues.push_back(o.value());
    _dynamicStringNames.push_back(o.name());
  }
}

void DLClient::updateOptions(dlserver::Model* model)
{
  model->clear_bool_options();
  for (int i = 0; i < _dynamicBoolValues.size(); i++) {
    ::dlserver::BoolOption* opt = model->add_bool_options();
    opt->set_name(_dynamicBoolNames[i]);
    opt->set_value(_dynamicBoolValues[i]);
  }

  model->clear_int_options();
  for (int i = 0; i < _dynamicIntValues.size(); i++) {
    ::dlserver::IntOption* opt = model->add_int_options();
    opt->set_name(_dynamicIntNames[i]);
    opt->set_value(_dynamicIntValues[i]);
  }

  model->clear_float_options();
  for (int i = 0; i < _dynamicFloatValues.size(); i++) {
    ::dlserver::FloatOption* opt = model->add_float_options();
    opt->set_name(_dynamicFloatNames[i]);
    opt->set_value(_dynamicFloatValues[i]);
  }

  model->clear_string_options();
  for (int i = 0; i < _dynamicStringValues.size(); i++) {
    ::dlserver::StringOption* opt = model->add_string_options();
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
    if (!_comms.validateHostName(_host)) {
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
      _comms.connectLoop(_host, _port);
      if (!_comms.isConnected()) {
        error("Could not connect to python server.");
      }
      else {
        _comms.sendInfoRequest();
        dlserver::RespondWrapper resp_wrapper;
        _comms.readInfoResponse(resp_wrapper);
        // Check if error occured in the server
        if (resp_wrapper.has_error()) {
          const char * errorMsg = resp_wrapper.error().msg().c_str();
          error(errorMsg);
          return 0;
        }
        // Parse message and fill in menu items for enumeration knob
        _serverModels.clear();
        _numInputs.clear();
        _inputNames.clear();
        std::vector<std::string> modelNames;
        int numModels = resp_wrapper.r1().num_models();
        std::cerr << "Server can serve " << std::to_string(numModels) << " models" << std::endl;
        std::cerr << "-----------------------------------------------" << std::endl;
        for (int i = 0; i < numModels; i++) {
          dlserver::Model m;
          m = resp_wrapper.r1().models(i);
          modelNames.push_back(m.label());
          _serverModels.push_back(m);
          _numInputs.push_back(m.inputs_size());
          std::vector<std::string> names;
          for (int j = 0; j < m.inputs_size(); j++) {
            dlserver::ImagePrototype p;
            p = m.inputs(j);
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
  return DLClient::kClassName;
}

const char* DLClient::node_help() const
{ 
  return DLClient::kHelpString;
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
  return _showDynamic && _comms.isConnected();
}