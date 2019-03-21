// Copyright (c) 2019 The Foundry Visionmongers Ltd.  All Rights Reserved.

#ifndef DLCLIENT_H
#define DLCLIENT_H

// Standard plug-in include files.
#include "DDImage/Iop.h"
#include "DDImage/NukeWrapper.h"
#include "DDImage/Row.h"
#include "DDImage/Tile.h"
#include "DDImage/Knobs.h"
#include "DDImage/Thread.h"
#include <DDImage/Enumeration_KnobI.h>

// Local include files
#include "DLClientComms.h"


//! The Deep Learning (DL) Client plug-in connects Nuke to a Python server to apply DL models to images.
/*! This plug-in can connect to a server (given a host and port), which responds
    with a list of available Deep Learning (DL) models and options.
    On every /a engine() call, the image and model options are sent from Nuke to the server,
    there the server can process the image by doing Deep Learning inference,
    finally the resulting image is sent back to Nuke.
*/
class DLClient : public DD::Image::Iop
{

public:
  // Static consts
  static const char* const kClassName;
  static const char* const kHelpString;

  static const char* const kDefaultHostName;
  static const int         kDefaultPortNumber;

public:
  //! Constructor. Initialize user controls to their default values.
  DLClient(Node* node);
  virtual ~DLClient();

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

  void _validate(bool);
  void _request(int x, int y, int r, int t, DD::Image::ChannelMask channels, int count);
  void _open();

  // This function does all the work.
  /*! For each line in the area passed to request(), this will be called. It must
      calculate the image data for a region at vertical position \a y, and between
      horizontal positions \a x and \a r, and write it to the passed row
      structure. Usually this works by asking the input for data, and modifying it.
  */
  void engine(int y, int x, int r, DD::Image::ChannelMask channels, DD::Image::Row &out);

  //! Information to the plug-in manager of DDNewImage/Nuke.
  static const DD::Image::Iop::Description description;

  static void addDynamicKnobs(void* , DD::Image::Knob_Callback);
  void knobs(DD::Image::Knob_Callback f);
  int knob_changed(DD::Image::Knob* );
  
  //! Return the name of the class.
  const char* Class() const;
  const char* node_help() const;

private:
  // Private functions for talking to the server

  //! Connect to server, then send inference request and read inference response
  bool processImage(const std::string& hostStr, int port);

  //! Parse the model options from the DL server.
  void parseOptions();
  //! Update any current options from any changes to the DL server.
  void updateOptions(dlserver::Model* model);


private:
  // Private dynamic knob helper functions.

  // Getters of the class
  int getNumOfFloats() const;
  int getNumOfInts() const;
  int getNumOfBools() const;
  int getNumOfStrings() const;

  std::string getDynamicBoolName(int idx);
  std::string getDynamicFloatName(int idx);
  std::string getDynamicIntName(int idx);
  std::string getDynamicStringName(int idx);

  float* getDynamicFloatValue(int idx);
  int* getDynamicIntValue(int idx);
  bool* getDynamicBoolValue(int idx);
  std::string* getDynamicStringValue(int idx);
  bool getShowDynamic() const;

private:
  // Private member variables

  //! Our main client <--> server communications object
  DLClientComms _comms;

  std::vector<std::vector<float>> _inputs;
  std::vector<int> _w;
  std::vector<int> _h;
  std::vector<int> _c;
  std::vector<float> _result;
  
  bool _firstTime;
  std::string _host;
  bool _hostIsValid;
  int _port;
  bool _portIsValid;
  int _chosenModel;
  bool _modelSelected;

  DD::Image::Lock _lock;
  DD::Image::Knob* _selectedModelknob;
  std::vector<dlserver::Model> _serverModels;

  std::vector<int> _numInputs;
  std::vector<std::vector<std::string>> _inputNames;

  bool _showDynamic;
  std::vector<int> _dynamicBoolValues;
  std::vector<int> _dynamicIntValues;
  std::vector<float> _dynamicFloatValues;
  std::vector<std::string> _dynamicStringValues;
  std::vector<std::string> _dynamicBoolNames;
  std::vector<std::string> _dynamicIntNames;
  std::vector<std::string> _dynamicFloatNames;
  std::vector<std::string> _dynamicStringNames;
  int _numNewKnobs;

};

#endif // DLCLIENT_H
