// Copyright (c) 2019 Alexander Mishurov.
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

#ifndef MLCLIENTGEO_H
#define MLCLIENTGEO_H

// Standard plug-in include files.
#include "DDImage/NukeWrapper.h"
#include "DDImage/Row.h"
#include "DDImage/Thread.h"

// Local include files
#include "MLClientMixin.h"

class MLClientGeo : public MLClientMixin<DD::Image::SourceGeo>
{

public:
  // Static consts
  static const char* const kClassName;
  static const char* const kHelpString;

public:
  //! Constructor. Initialize parent class.
  MLClientGeo(Node* node);
  virtual ~MLClientGeo();

public:
  // DDImage::SourceGeo overrides

  //! The op needs both the GeoOp knobs and the MLClientMixin knobs.
  void knobs(DD::Image::Knob_Callback f);
  int knob_changed(DD::Image::Knob* );
  //! The maximum number of input connections the operator can have.
  int maximum_inputs() const;
  //! The minimum number of input connections the operator can have.
  int minimum_inputs() const;
  /*! Return the text Nuke should draw on the arrow head for input \a input
      in the DAG window. This should be a very short string, one letter
      ideally. Return null or an empty string to not label the arrow.
  */
  const char* input_label(int input, char* buffer) const;

  //! Unlike SourceGeo, this op uses the first input for sending data to the server.
  DD::Image::Iop* default_material_iop() const;

  void _validate(bool);
  void get_geometry_hash();
  /*! This function is called by Nuke for processing the current image.
      The image and model options are sent from Nuke to the server,
      there the server can process the image by doing Machine Learning inference,
      finally the resulting data is sent back to Nuke.
      The function tries to reconnect if no connection is set.
  */
  void create_geometry(DD::Image::Scene& scene, DD::Image::GeometryList& out);

  //! Information to the plug-in manager of DDNewImage/Nuke.
  static const DD::Image::Op::Description description;

  //! Return the name of the class.
  const char* Class() const;
  const char* node_help() const;

private:
  // Private functions

  //! Computes specific number of inputs, GeoOp uses one input as a material.
  int computed_inputs();
  //! Returns a format of for sending to the server, GeoOp hasn't got output format.
  DD::Image::Box getFormat() const;
  //! Parse the response messge from the server, and if it contains
  //! a geometry object, attempt to create geometry from the recieved data. Return
  //! true on success, false otherwise and fill in the error string.
  bool drawGeometry(mlserver::RespondWrapper& responseWrapper, DD::Image::Scene& scene,
                    DD::Image::GeometryList& out, std::string& errorMsg);
private:
  // Private member variables

  //! Struct and map for mapping data from the response
  //! to Nuke's GeoInfo vector attributes.
  struct FloatAttrib {
    std::string name;
    int dim;
    DD::Image::GroupType group;
    DD::Image::AttribType type;
    const google::protobuf::RepeatedField<float>* data;
  };
  std::map<std::string, FloatAttrib> floatAttribs;

};

#endif // MLCLIENTGEO_H
