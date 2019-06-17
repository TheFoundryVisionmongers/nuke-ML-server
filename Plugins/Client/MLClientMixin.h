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

#ifndef MLCLIENTMIXIN_H
#define MLCLIENTMIXIN_H

// Standard plug-in include files.
#include "DDImage/PlanarIop.h"
#include <DDImage/SourceGeo.h>
#include "DDImage/Knobs.h"
#include "DDImage/Tile.h"
#include <DDImage/Enumeration_KnobI.h>

// Local include files
#include "MLClientComms.h"
#include "MLClientModelManager.h"

template<class T>
class MLClientMixin : public T
{

public:
  // Static consts
  static const char* const kDefaultHostName;
  static const int         kDefaultPortNumber;

private:
  static const DD::Image::ChannelSet kDefaultChannels;
  static const int kDefaultNumberOfChannels;

public:
  //! Constructor. Initialize user controls to their default values.
  MLClientMixin(Node* node);
  virtual ~MLClientMixin();

public:
  // DDImage::Iop / DDImage::SourceGeo overrides

  static void addDynamicKnobs(void* , DD::Image::Knob_Callback);
  void knobs(DD::Image::Knob_Callback f);
  int knob_changed(DD::Image::Knob* );

  MLClientModelManager& getModelManager();

protected:
  // Protected functions for talking to the server
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

  /*! This is actual function which is called by Nuke for processing the current image.
      It performs common checks and accpets a lambda from a superclass which captures
      arguments from different scopes and calls the superclass' rendering function
  */
  bool tryInfer(std::function<bool(mlserver::RespondWrapper&, std::string&)> renderFunc);

  //! Validate model knobs inside _validate(), GeoOp doesn't call copy_info().
  void validateModelKnobs();
  //! Compute specific number of inputs, GeoOp uses one input as a material.
  virtual int computed_inputs() = 0;
  //! Return a format of for sending to the server, GeoOp hasn't got output format.
  virtual DD::Image::Box getFormat() const = 0;

private:
  //! Return whether the dynamic knobs should be shown or not.
  bool getShowDynamic() const;

protected:
  // Protected member variables

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

#endif // MLCLIENTMIXIN_H
