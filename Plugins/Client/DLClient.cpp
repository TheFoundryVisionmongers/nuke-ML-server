// Copyright (c) 2018 The Foundry Visionmongers Ltd.  All Rights Reserved.

#include <cstring>

#include "DLClient.h"

const char* const DLClient::kClassName = "DLClient";
const char* const DLClient::kHelpString = "Connects to a Python server for Deep Learning inference.";

const char* const DLClient::kDefaultHostName = "172.17.0.2";
const int         DLClient::kDefaultPortNumber = 55555;

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
: DD::Image::Iop(node)
, _firstTime(true)
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
      if (_comms.isConnected() && _modelSelected) {
        processImage(_host, _port);
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

bool DLClient::processImage(const std::string& hostStr, int port)
{
  try {
    _comms.connectLoop(hostStr, port);

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
    _comms.readInferenceResponse(_result);
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

        // Read the response from the server
        dlserver::RespondWrapper resp_wrapper;
        _comms.readInfoResponse(resp_wrapper);

        // Parse message and fill in menu items for enumeration knob
        _serverModels.clear();
        _numInputs.clear();
        _inputNames.clear();
        std::vector<std::string> modelNames;
        int numModels = resp_wrapper.r1().nummodels();
        std::cerr << "Server can serve " << std::to_string(numModels) << " models" << std::endl;
        std::cerr << "-----------------------------------------------" << std::endl;
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
