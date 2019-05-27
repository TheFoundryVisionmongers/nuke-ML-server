// Copyright (c) 2019 Foundry.
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

#ifndef MLCLIENT_H
#define MLCLIENT_H

// Standard plug-in include files.
#include "DDImage/PlanarIop.h"
#include "DDImage/NukeWrapper.h"
#include "DDImage/Row.h"
#include "DDImage/Tile.h"
#include "DDImage/Knobs.h"
#include "DDImage/Thread.h"
#include <DDImage/Enumeration_KnobI.h>

// Local include files
#include "MLClientComms.h"
#include "MLClientModelManager.h"

//! The Machine Learning (ML) Client plug-in connects Nuke to a Python server to apply ML models to images.
/*! This plug-in can connect to a server (given a host and port), which responds
    with a list of available Machine Learning (ML) models and options.
    On every /a renderStripe() call, the image and model options are sent from Nuke to the server,
    there the server can process the image by doing Machine Learning inference,
    finally the resulting image is sent back to Nuke.
*/
class MLClient : public DD::Image::PlanarIop
{

public:
  // Static consts
  static const char* const kClassName;
  static const char* const kHelpString;

  static const char* const kDefaultHostName;
  static const int         kDefaultPortNumber;

private:
  static const DD::Image::ChannelSet kDefaultChannels;
  static const int kDefaultNumberOfChannels;

public:
  //! Constructor. Initialize user controls to their default values.
  MLClient(Node* node);
  virtual ~MLClient();

public:
  // DDImage::Iop overrides

  //! The maximum number of input connections the operator can have.
  int maximum_inputs() const;
  //! The minimum number of input connections the operator can have.
  int minimum_inputs() const;
  /*! Return the text Nuke should draw on the arrow head for input \a input
      in the DAG window. This should be a very short string, one letter
      ideally. Return null or an empty string to not label the arrow.
  */
  const char* input_label(int input, char* buffer) const;

  bool useStripes() const;
  bool renderFullPlanes() const;

  void _validate(bool);
  void getRequests(const DD::Image::Box& box, const DD::Image::ChannelSet& channels, int count, DD::Image::RequestOutput &reqData) const;

  /*! This function is called by Nuke for processing the current image.
      The image and model options are sent from Nuke to the server,
      there the server can process the image by doing Machine Learning inference,
      finally the resulting image is sent back to Nuke.
      The function tries to reconnect if no connection is set.
  */
  void renderStripe(DD::Image::ImagePlane& imagePlane);

  //! Information to the plug-in manager of DDNewImage/Nuke.
  static const DD::Image::Iop::Description description;

  static void addDynamicKnobs(void* , DD::Image::Knob_Callback);
  void knobs(DD::Image::Knob_Callback f);
  int knob_changed(DD::Image::Knob* );
  
  //! Return the name of the class.
  const char* Class() const;
  const char* node_help() const;

  MLClientModelManager& getModelManager();

private:
  // Private functions for talking to the server
  //! Try connect to the server and set-up the relevant knobs. Return true on
  //! success, false otherwise and setting a descriptive error in errorMsg.
  bool refreshModelsAndKnobsFromServer(std::string& errorMsg);

  //! Return whether we successfully managed to pull model
  //! info from the server at some time in the past, and the selected model is
  //! valid.
  bool haveValidModelInfo() const;

  //! Connect to server, then send inference request and read inference response.
  //! Return true on success, false otherwise filling in the errorMsg.
  bool processImage(const std::string& hostStr, int port, mlserver::RespondWrapper& responseWrapper, std::string& errorMsg);

  //! Parse the response messge from the server, and if it contains
  //! an image, attempt to copy the image to the imagePlane. Return
  //! true on success, false otherwise and fill in the error string.
  bool renderOutputBuffer(mlserver::RespondWrapper& responseWrapper, DD::Image::ImagePlane& imagePlane, std::string& errorMsg);

  //! Return whether the dynamic knobs should be shown or not.
  bool getShowDynamic() const;

private:
  // Private member variables
  
  std::string _host;
  bool _hostIsValid;
  int _port;
  bool _portIsValid;
  int _chosenModel;
  bool _modelSelected;

  DD::Image::Knob* _selectedModelknob;
  std::vector<mlserver::Model> _serverModels;

  std::vector<int> _numInputs;
  std::vector<std::vector<std::string>> _inputNames;

  bool _showDynamic;
  
  MLClientModelManager _modelManager;

  int _numNewKnobs;

};

#endif // MLCLIENT_H
