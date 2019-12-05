// Copyright (c) 2018 Foundry.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*************************************************************************

#include <cstring>

#include "MLClient.h"

const char* const MLClient::kClassName = "MLClient";
const char* const MLClient::kHelpString =
  "Connects to a Python server for Machine Learning inference.";

const char* const MLClient::kDefaultHostName = "172.17.0.2";
const int         MLClient::kDefaultPortNumber = 55555;

const DD::Image::ChannelSet MLClient::kDefaultChannels = DD::Image::Mask_RGB;
const int MLClient::kDefaultNumberOfChannels = MLClient::kDefaultChannels.size();

using namespace DD::Image;

/*! This is a function that creates an instance of the operator, and is
   needed for the Iop::Description to work.
 */
static Iop* MLClientCreate(Node* node)
{
  return new MLClient(node);
}

/*! The Iop::Description is how NUKE knows what the name of the operator is,
   how to create one, and the menu item to show the user. The menu item may be
   0 if you do not want the operator to be visible.
 */
const Iop::Description MLClient::description(MLClient::kClassName, 0, MLClientCreate);

//! Constructor. Initialize user controls to their default values.
MLClient::MLClient(Node* node)
: DD::Image::PlanarIop(node)
, _host(MLClient::kDefaultHostName)
, _hostIsValid(true)
, _port(MLClient::kDefaultPortNumber)
, _portIsValid(true)
, _chosenModel(0)
, _modelSelected(false)
, _showDynamic(false)
, _numNewKnobs(0)
, _modelManager(this)
{ }

MLClient::~MLClient() {}

//! The maximum number of input connections the operator can have.
int MLClient::maximum_inputs() const
{
  if (haveValidModelInfo() && _modelSelected) {
    return _numInputs[_chosenModel];
  }
  else {
    return 1;
  }
}

//! The minimum number of input connections the operator can have.
int MLClient::minimum_inputs() const
{ 
  if (haveValidModelInfo() && _modelSelected) {
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
const char* MLClient::input_label(int input, char* buffer) const
{
  if (!haveValidModelInfo() || !_modelSelected) {
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

bool MLClient::useStripes() const
{
  return false;
}

bool MLClient::renderFullPlanes() const
{
  return true;
}

MLClientModelManager& MLClient::getModelManager()
{
  return dynamic_cast<MLClient*>(firstOp())->_modelManager;
}

int MLClient::getNumNewKnobs()
{
  return dynamic_cast<MLClient*>(firstOp())->_numNewKnobs;
}

void MLClient::setNumNewKnobs(int i)
{
  dynamic_cast<MLClient*>(firstOp())->_numNewKnobs = i;
}

void MLClient::_validate(bool forReal)
{
  // Try connect to the server, erroring if it can't connect.
  std::string connectErrorMsg;
  if (!haveValidModelInfo() && !refreshModelsAndKnobsFromServer(connectErrorMsg)) {
    error(connectErrorMsg.c_str());
  }
  // The only other thing needed to do in validate is copy the image info.
  copy_info();
}

void MLClient::getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput &reqData) const
{
  // request all input input as we are going to search the whole input area
  for (int i = 0, endI = getInputs().size(); i < endI; i++) {
    const ChannelSet readChannels = input(i)->info().channels();
    input(i)->request(readChannels, count);
  }
}

void MLClient::renderStripe(ImagePlane& imagePlane)
{
  // Before doing any rendering, check if we've aborted.
  // Note that it's perfectly fine to abort here, it usually
  // means the user has scrubbed quickly around on the timeline
  // or switched between Viewers.
  if (aborted() || cancelled()) {
    // The following print is commented out as it happens too frequently
    // MLClientComms::Vprint("Aborted before processing images.");
    return;
  }

  // Check that it's connected and set up correctly
  if (haveValidModelInfo() && _modelSelected) {
    // Set up our error string
    std::string errorMsg;
    // Set up our incoming response message structure.
    mlserver::RespondWrapper responseWrapper;
    // Wrap up our image data to be sent, send it, and
    // retrieve the response.
    if(!processImage(_host, _port, responseWrapper, errorMsg)) {
      // Test if the failure was due to Nuke aborting
      if (aborted() || cancelled()) {
        // errorMsg should be filled with where / when the
        // processImage() call was aborted.
        MLClientComms::Vprint(errorMsg);
        return;
      }
      // Display the error in Nuke if it was some systematic issue.
      error(errorMsg.c_str());
      return;
    }
    // If control reached here then responseWrapper contains a valid
    // response, so let's try to extract an image from it and
    // place it into our imagePlane.
    if (!renderOutputBuffer(responseWrapper, imagePlane, errorMsg)) {
      MLClientComms::Vprint(errorMsg);
      error(errorMsg.c_str());
      return;
    }
    // If control reached here, it's all good, return.
    return;
  }

  // Check again if we hit abort during processing
  if (aborted() || cancelled()) {
    MLClientComms::Vprint("Aborted without processing image.");
    return;
  }

  // If we reached here by default let's pull an image from input0() so
  // that it's at least passing something through.
  input0().fetchPlane(imagePlane);
}

bool MLClient::refreshModelsAndKnobsFromServer(std::string& errorMsg)
{
  // Before trying to connect, ensure ports and hostname are valid.
  if (!_portIsValid) {
    errorMsg = "Port is invalid.";
    return false;
  }
  if(!_hostIsValid) {
    errorMsg = "Hostname is invalid.";
    return false;
  }

  // Actually try to connect, and pull model info
  mlserver::RespondWrapper responseWrapper;
  {
    // Local scope our comms object so that the connection is torn
    // down after we have our data.
    MLClientComms comms(_host, _port);

    if (!comms.isConnected()) {
      errorMsg = "Could not connect to server. Please check your host / port numbers.";
      return false;
    }

    // Try pull the model info into the responseWrapper
    if(!comms.sendInfoRequestAndReadInfoResponse(responseWrapper, errorMsg)) {
      // If it failed, the error is set, return.
      return false;
    }
  }
  // Parse message and fill in menu items for enumeration knob
  _serverModels.clear();
  _numInputs.clear();
  _inputNames.clear();
  std::vector<std::string> modelNames;
  int numModels = responseWrapper.r1().num_models();
  std::stringstream ss;
  ss << "Server can serve " << std::to_string(numModels) << " models" << std::endl;
  ss << "-----------------------------------------------";
  MLClientComms::Vprint(ss.str());
  for (int i = 0; i < numModels; i++) {
    mlserver::Model m;
    m = responseWrapper.r1().models(i);
    modelNames.push_back(m.label());
    _serverModels.push_back(m);
    _numInputs.push_back(m.inputs_size());
    std::vector<std::string> names;
    for (int j = 0; j < m.inputs_size(); j++) {
      mlserver::ImagePrototype p;
      p = m.inputs(j);
      names.push_back(p.name());
    }
    _inputNames.push_back(names);
  }

  // Sanity check that some models were returned
  if (_serverModels.size() == 0) {
    errorMsg = "Server returned no models.";
    return false;
  }

  // Change enumeration knob choices
  Enumeration_KnobI* pSelectModelEnum = _selectedModelknob->enumerationKnob();
  pSelectModelEnum->menu(modelNames);

  if (_chosenModel >= (int)numModels) {
    _selectedModelknob->set_value(0);
    _chosenModel = 0;
  }

  // Set member variables to indicate our connections and model set-up succeeded.
  _modelSelected = true;
  _showDynamic = true;

  // Update the dynamic knobs
  const mlserver::Model m = _serverModels[_chosenModel];
  if (this == this->firstOp()) {
    getModelManager().parseOptions(m);
  }
  setNumNewKnobs(replace_knobs(knob("models"), getNumNewKnobs(), addDynamicKnobs, this->firstOp()));
  // Return true if control made it here, success.
  return true;
}

//! Return whether we successfully managed to pull model
//! info from the server at some time in the past, and the selected model is
//! valid.
bool MLClient::haveValidModelInfo() const
{
  return _serverModels.size() > 0 && _serverModels.size() > _chosenModel;
}

bool MLClient::processImage(const std::string& hostStr, int port,
  mlserver::RespondWrapper& responseWrapper, std::string& errorMsg)
{
  // Check if Nuke has aborted, making it impossible to pull images.
  if (aborted()) {
    errorMsg = "Process aborted at beginning of processing images.";
    return false;
  }

  try {

    // Sanity check that some models exist and a valid one is selected.
    if (!haveValidModelInfo()) {
      errorMsg = "No models exist to send to server.";
      return false;
    }

    // Checking again after connection is made, just in case.
    if (aborted()) {
      errorMsg = "Process aborted after connection.";
      return false;
    }

    // Validate ourself before proceeding, this ensures if this is being invoked by a button press
    // then it's set up correctly.
    if (!tryValidate(/*for_real*/true)) {
      errorMsg = "Could not set-up node correctly.";
      return false;
    }

    // And again after validate
    if (aborted()) {
      errorMsg = "Process aborted after validating self.";
      return false;
    }

    // Set our format box, this is the dimension of the
    // image that will be passed to the server.
    const Box imageFormat = info().format();

    // Create inference message
    mlserver::RequestInference* requestInference = new mlserver::RequestInference;
    mlserver::Model* m = new mlserver::Model(_serverModels[_chosenModel]);
    getModelManager().updateOptions(*m);
    requestInference->set_allocated_model(m);

    // Parse image. TODO: Check for multiple inputs, different channel size
    for (int i = 0; i < node_inputs(); i++) {
      // Create an ImagePlane, and read each input into it.
      // Get our input & sanity check
      DD::Image::Iop* inputIop = dynamic_cast<DD::Image::Iop*>(input(i));
      if (inputIop == NULL) {
        errorMsg = "Input is empty or not connected.";
        return false;
      }

      // Checking before validating inputs
      if (aborted()) {
        errorMsg = "Process aborted before validating inputs.";
        return false;
      }

      // Try validate & request the input, this should be quick if the data
      // has already been requested.
      if (!inputIop->tryValidate(/*force*/true)) {
        errorMsg = "Unable to validate input.";
        return false;
      }

      // Set our input bounding box, this is what our inputs can give us.
      Box imageBounds = inputIop->info();
      // We're going to clip it to our format.
      imageBounds.intersect(imageFormat);
      const int fx = imageBounds.x();
      const int fy = imageBounds.y();
      const int fr = imageBounds.r();
      const int ft = imageBounds.t();

      // Request our default channels, for our own bounding box
      inputIop->request(fx, fy, fr, ft, kDefaultChannels, 0);
      // Let's assume everything went fine, and fetch our plane
      ImagePlane plane(imageBounds, /*packed*/ true, kDefaultChannels, kDefaultNumberOfChannels);
      inputIop->fetchPlane(plane);

      // Sanity check that that the plane was filled successfully, and nothing
      // was interrupted.
      if (plane.usage() == 0) {
        errorMsg = "No image data fetched from input.";
        return false;
      }

      // Checking after fetching inputs
      if (aborted()) {
        errorMsg = "Process aborted after fetching inputs.";
        return false;
      }

      // Set up our message
      mlserver::Image* image = requestInference->add_images();
      image->set_width(imageFormat.w());
      image->set_height(imageFormat.h());
      image->set_channels(kDefaultNumberOfChannels);

      // Set up our temp contiguous buffer
      size_t byteBufferSize = imageFormat.w() * imageFormat.h() * kDefaultNumberOfChannels * sizeof(float);
      if (byteBufferSize == 0) {
        errorMsg = "Image size is zero.";
        return false;
      }
      // Create and zero our buffer
      byte* byteBuffer = new byte[byteBufferSize];
      std::memset(byteBuffer, 0, byteBufferSize);

      // Copy the data from our image plane to the buffer. Ideally
      // this should be done directly on the plane's data but it
      // can't guarantee that it's contiguous, or packed in the
      // expected way.
      float* floatBuffer = (float*)byteBuffer;
      for (int z = 0; z < kDefaultNumberOfChannels; z++) {
        const int chanStride = z * imageFormat.w() * imageFormat.h();

        for (int ry = fy; ry < ft; ry++) {
          const int rowStride = ry * imageFormat.w();

          ImageTileReadOnlyPtr tile = plane.readableAt(ry, z);
          for (int rx = fx, currentPos = 0; rx < fr; rx++) {
            size_t fullPos = chanStride + rowStride + currentPos++;
            floatBuffer[fullPos] = tile[rx];
          }
        }
      }

      // Set the image data on our message, and release the temp buffer.
      image->set_image(byteBuffer, byteBufferSize);
      delete[] byteBuffer;
    }

    // Send the inference request, await and process the response.
    {
      // Local scope our comms object so that the connection is torn
      // down after we have our data.
      MLClientComms comms(_host, _port);

      if (!comms.isConnected()) {
        errorMsg = "Could not connect to server. Please check your host / port numbers.";
        return false;
      }
      // Try pull the model info into the responseWrapper
      if (!comms.sendInferenceRequestAndReadInferenceResponse(*requestInference, responseWrapper, errorMsg)) {
        // If it failed, the error is set, return.
        return false;
      }
    }
  }
  catch (...) {
    errorMsg = "Error processing messages.";
    MLClientComms::Vprint(errorMsg);
    return false;
  }

  // Return true to indicate success
  return true;
}

bool MLClient::renderOutputBuffer(mlserver::RespondWrapper& responseWrapper, DD::Image::ImagePlane& imagePlane, std::string& errorMsg)
{
  // Sanity check, make sure the response actually contains an image.
  if (!responseWrapper.has_r2() || responseWrapper.r2().num_images() == 0) {
    errorMsg = "No image found in message response.";
    return false;
  }

  // Validate ourself before proceeding, this ensures if this is being invoked by a button press
  // then it's set up correctly. This will return immediately if it's already set up.
  if (!tryValidate(/*for_real*/true)) {
    errorMsg = "Could not set-up node correctly.";
    return false;
  }

  // Get the resulting image data
  const mlserver::Image &imageMessage = responseWrapper.r2().images(0);

  // Verify that the image passed back to us is of the same format as the input
  // format (note, the bounds of the imagePlane may be different, e.g. if there's
  // a Crop on the input.)
  const Box imageFormat = info().format();
  if (imageMessage.width() != imageFormat.w() || imageMessage.height() != imageFormat.h()) {
    errorMsg = "Received Image has dimensions different than expected";
    return false;
  }

  // Set the dimensions of the imagePlane, note this can be different than the format.
  // Clip it to the intersection of the image format.
  Box imageBounds = imagePlane.bounds();
  imageBounds.intersect(imageFormat);
  const int fx = imageBounds.x();
  const int fy = imageBounds.y();
  const int fr = imageBounds.r();
  const int ft = imageBounds.t();

  // This is going to copy back the minimum intersection of channels between
  // what's required to fill in imagePlane, and what's been returned
  // in the response. This allows us to gracefully handle cases where the returned
  // image has too few channels, or when the imagePlane has too many.
  const size_t numChannelsToCopy = (imageMessage.channels() < imagePlane.channels().size()) ? imageMessage.channels() : imagePlane.channels().size();

  // Copy the data
  const char* imageByteDataPtr = imageMessage.image().c_str();

  // Sanity check our image has the correct number of elements
  const size_t numImageElements = imageMessage.image().size() / sizeof(float);
  const size_t numImageElementsToCopy = numChannelsToCopy * (ft - fy) * (fr - fx);

  if (numImageElements < numImageElementsToCopy) {
    errorMsg = "Received Image has insuffient elements.";
    return false;
  }

  // Allow the imagePlane to be writable
  imagePlane.makeWritable();

  float* imageFloatDataPtr = (float*)imageByteDataPtr;
  for (int z = 0; z < numChannelsToCopy; z++) {
    const int chanStride = z * imageFormat.w() * imageFormat.h();

    for (int ry = fy; ry < ft; ry++) {
      const int rowStride = ry * imageFormat.w();

      for (int rx = fx, currentPos = 0; rx < fr; rx++) {
        int fullPos = chanStride + rowStride + currentPos++;
        imagePlane.writableAt(rx, ry, z) = imageFloatDataPtr[fullPos];
      }
    }
  }
  // If this is reached here, return true for success
  return true;
}

void MLClient::addDynamicKnobs(void* p, Knob_Callback f)
{
  if (((MLClient *)p)->getShowDynamic()) {
    for (int i = 0; i < ((MLClient *)p)->getModelManager().getNumOfInts(); i++) {
      std::string name = ((MLClient *)p)->getModelManager().getDynamicIntName(i);
      std::string label = ((MLClient *)p)->getModelManager().getDynamicIntName(i);
      Int_knob(f, ((MLClient *)p)->getModelManager().getDynamicIntValue(i), name.c_str(), label.c_str());
      Newline(f, " ");
    }
    for (int i = 0; i < ((MLClient *)p)->getModelManager().getNumOfFloats(); i++) {
      std::string name = ((MLClient *)p)->getModelManager().getDynamicFloatName(i);
      std::string label = ((MLClient *)p)->getModelManager().getDynamicFloatName(i);
      Float_knob(f, ((MLClient *)p)->getModelManager().getDynamicFloatValue(i), name.c_str(), label.c_str());
      ClearFlags(f, Knob::SLIDER);
      Newline(f, " ");
    }
    for (int i = 0; i < ((MLClient *)p)->getModelManager().getNumOfBools(); i++) {
      std::string name = ((MLClient *)p)->getModelManager().getDynamicBoolName(i);
      std::string label = ((MLClient *)p)->getModelManager().getDynamicBoolName(i);
      Bool_knob(f, ((MLClient *)p)->getModelManager().getDynamicBoolValue(i), name.c_str(), label.c_str());
      Newline(f, " ");
    }
    for (int i = 0; i < ((MLClient *)p)->getModelManager().getNumOfStrings(); i++) {
      std::string name = ((MLClient *)p)->getModelManager().getDynamicStringName(i);
      std::string label = ((MLClient *)p)->getModelManager().getDynamicStringName(i);
      String_knob(f, ((MLClient *)p)->getModelManager().getDynamicStringValue(i), name.c_str(), label.c_str());
      Newline(f, " ");
    }
    for (int i = 0; i < ((MLClient *)p)->getModelManager().getNumOfButtons(); i++) {
      std::string name = ((MLClient *)p)->getModelManager().getDynamicButtonName(i);
      std::string label = ((MLClient *)p)->getModelManager().getDynamicButtonName(i);
      Button(f, name.c_str(), label.c_str());
      Newline(f, " ");
    }
  }
}

void MLClient::knobs(Knob_Callback f)
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
    MLClient::addDynamicKnobs(this->firstOp(), f);
  }
}

int MLClient::knob_changed(Knob* knobChanged)
{
  if (knobChanged->is("host")) {
    if (!MLClientComms::ValidateHostName(_host)) {
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
    std::string connectErrorMsg;
    if (!refreshModelsAndKnobsFromServer(connectErrorMsg)) {
      error(connectErrorMsg.c_str());
    }
    return 1;
  }

  if (knobChanged->is("models")) {
    // Sanity check that some models exist
    if (haveValidModelInfo()) {
      const mlserver::Model m = _serverModels[_chosenModel];
      getModelManager().parseOptions(m);
      setNumNewKnobs(replace_knobs(knob("models"), getNumNewKnobs(), addDynamicKnobs, this->firstOp()));
    }
    return 1;
  }

  // Check if dynamic button is pressed
  for (int i = 0; i < getModelManager().getNumOfButtons(); i++) {
    if (knobChanged->is(getModelManager().getDynamicButtonName(i).c_str())) {
      // Set current button to true (pressed) for model inference
      getModelManager().setDynamicButtonValue(i, true);
      // Set up our error string
      std::string errorMsg;
      // Set up our incoming response message structure.
      mlserver::RespondWrapper responseWrapper;
      // Wrap up our image data to be sent, send it, and
      // retrieve the response.
      if (!processImage(_host, _port, responseWrapper, errorMsg)) {
        error(errorMsg.c_str());
      }

      // Get the resulting general data
      if (responseWrapper.has_r2() && responseWrapper.r2().num_objects() > 0) {
        const mlserver::FieldValuePairAttrib object = responseWrapper.r2().objects(0);
        // Run script in Nuke if object called PythonScript is created
        if (object.name() == "PythonScript") {
          // Check object has string_attributes
          if (object.values_size() != 0
            && object.values(0).string_attributes_size() != 0) {
            mlserver::StringAttrib pythonScript = object.values(0).string_attributes(0);
            // Run Python Script in Nuke
            if (pythonScript.values_size() != 0) {
              std::cout << " cmd=\n" << pythonScript.values(0) << "\n" << std::flush;
              script_command(pythonScript.values(0).c_str(), true, false);
              script_unlock();
            }
          }
        }
      }
      // Set current button to false (unpressed)
      getModelManager().setDynamicButtonValue(i, false);
      return 1;
    }
  }
  return 0;
}

//! Return the name of the class.
const char* MLClient::Class() const
{
  return MLClient::kClassName;
}

const char* MLClient::node_help() const
{
  return MLClient::kHelpString;
}

bool MLClient::getShowDynamic() const
{
  return _showDynamic && haveValidModelInfo();
}
