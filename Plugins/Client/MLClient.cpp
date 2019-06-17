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

#include "MLClient.h"
#include "MLClientMixin-inl.h"

const char* const MLClient::kClassName = "MLClient";
const char* const MLClient::kHelpString = 
  "Connects to a Python server for Machine Learning inference.";

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

//! Constructor. Initialize parent class.
MLClient::MLClient(Node* node)
: MLClientMixin<DD::Image::PlanarIop>(node)
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

void MLClient::_validate(bool forReal)
{
  // Try connect to the server, erroring if it can't connect.
  validateModelKnobs();
  // The only other thing needed to do in validate is copy the image info.
  copy_info();
}

void MLClient::getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput &reqData) const
{
  // request all input input as we are going to search the whole input area
  for (int i = 0, endI = this->getInputs().size(); i < endI; i++) {
    const ChannelSet readChannels = this->input(i)->info().channels();
    this->input(i)->request(readChannels, count);
  }
}

void MLClient::renderStripe(ImagePlane& imagePlane)
{
  // Create a lambda function which captures arguments from this scope and
  // the superclass' scope and call the rendering function particular to
  // specific type of node i.e Iop or GeoOp.
  auto renderFunc = [&](mlserver::RespondWrapper& responseWrapper, std::string& errorMsg)
    { return renderOutputBuffer(responseWrapper, imagePlane, errorMsg); };

  if (!tryInfer(renderFunc)) {
    // If we reached here by default let's pull an image from input0() so
    // that it's at least passing something through.
    input0().fetchPlane(imagePlane);
  }
}

bool MLClient::renderOutputBuffer(mlserver::RespondWrapper& responseWrapper, DD::Image::ImagePlane& imagePlane, std::string& errorMsg)
{
  // Sanity check, make sure the response actually contains an image.
  if(!responseWrapper.has_r2() || responseWrapper.r2().num_images() == 0) {
    errorMsg = "No image found in message response.";
    return false;
  }

  // Validate ourself before proceeding, this ensures if this is being invoked by a button press
  // then it's set up correctly. This will return immediately if it's already set up.
  if( !tryValidate(/*for_real*/true) ) {
    errorMsg = "Could not set-up node correctly.";
    return false;
  }

  // Get the resulting image data
  const mlserver::Image &imageMessage = responseWrapper.r2().images(0);

  // Verify that the image passed back to us is of the same format as the input
  // format (note, the bounds of the imagePlane may be different, e.g. if there's
  // a Crop on the input.)
  const Box imageFormat = info().format();
  if(imageMessage.width() != imageFormat.w() || imageMessage.height() != imageFormat.h()) {
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
  const size_t numChannelsToCopy =  (imageMessage.channels() < imagePlane.channels().size()) ? imageMessage.channels() : imagePlane.channels().size();

  // Allow the imagePlane to be writable
  imagePlane.makeWritable();

  // Copy the data
  const char* imageByteDataPtr = imageMessage.image().c_str();

  float* imageFloatDataPtr = (float*)imageByteDataPtr;
  for (int z = 0; z < numChannelsToCopy; z++) {
    const int chanStride = z * imageFormat.w() * imageFormat.h();

    for(int ry = fy; ry < ft; ry++) {
      const int rowStride = ry * imageFormat.w();

      for(int rx = fx, currentPos = 0; rx < fr; rx++) {
        int fullPos = chanStride + rowStride + currentPos++;
        imagePlane.writableAt(rx, ry, z)  = imageFloatDataPtr[fullPos];
      }
    }
  }

  // If this is reached here, return true for success
  return true;
}

//! Compute specific number of inputs, GeoOp uses one input as a material.
int MLClient::computed_inputs() {
  return node_inputs();
}

//! Return a format of for sending to the server, GeoOp hasn't got output format.
Box MLClient::getFormat() const
{
  return info().format();
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
