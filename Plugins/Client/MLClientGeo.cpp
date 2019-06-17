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

#include <DDImage/Point.h>
#include <DDImage/PolyMesh.h>

#include "MLClientGeo.h"
#include "MLClientMixin-inl.h"

const char* const MLClientGeo::kClassName = "MLClientGeo";
const char* const MLClientGeo::kHelpString = 
  "Connects to a Python server for Machine Learning inference.";

using namespace DD::Image;

/*! This is a function that creates an instance of the operator, and is
   needed for the Op::Description to work.
 */
static Op* MLClientGeoCreate(Node* node)
{
  return new MLClientGeo(node);
}

/*! The Iop::Description is how NUKE knows what the name of the operator is,
   how to create one, and the menu item to show the user. The menu item may be
   0 if you do not want the operator to be visible.
 */
const Op::Description MLClientGeo::description(MLClientGeo::kClassName, MLClientGeoCreate);

//! Constructor. Initialize user controls to their default values.
MLClientGeo::MLClientGeo(Node* node)
: MLClientMixin<DD::Image::SourceGeo>(node)
{
  /*! The map maps names of repeated FloatAttribs to the actual
      GeoInfo vector attributes. It would be enough for
      Nuke's standard attributes. For now it covers the attributes listed here
  */
  floatAttribs["uv Group_Vertices"] = {"uv", 4, Group_Vertices, VECTOR4_ATTRIB, nullptr };
  floatAttribs["N Group_Vertices"] = {"N", 3, Group_Vertices, NORMAL_ATTRIB, nullptr };
  floatAttribs["Cf Group_Points"] = {"Cf", 3, Group_Points, VECTOR3_ATTRIB, nullptr };
}

MLClientGeo::~MLClientGeo() {}

//! It can have one input as a source image for the server, for now.
int MLClientGeo::maximum_inputs() const
{
  return 1;
}

//! The second optional input is a material
int MLClientGeo::minimum_inputs() const
{
  return 2;
}

/*! Return the text Nuke should draw on the arrow head for input \a input
    in the DAG window. This should be a very short string, one letter
    ideally. Return null or an empty string to not label the arrow.
*/
const char* MLClientGeo::input_label(int input, char* buffer) const
{
  if (input == 1)
    return "tex";
  return MLClientMixin<SourceGeo>::input_label(input, buffer);
}

/*! SourceGeo's test_input() accepts only Iops as it is. Thus there's
    no need in overriding test_input() and default_input().
    Yet I iop_input() and input0() should be used for sending image to
    server. If second input isn't connected, return default Iop from SourceGeo
    so Nuke weren't crashing due to accesses to a non-allocated memory.
*/
Iop* MLClientGeo::default_material_iop() const
{
  if (input1())
    return input1()->iop();
  return MLClientMixin<SourceGeo>::default_input(0)->iop();
}

void MLClientGeo::_validate(bool forReal)
{
  // Try connect to the server, erroring if it can't connect.
  validateModelKnobs();
  MLClientMixin<SourceGeo>::_validate(forReal);
}

void MLClientGeo::create_geometry(Scene& scene, GeometryList& out)
{
  // Create a lambda function which captures arguments from this scope and
  // the superclass' scope and call the rendering function particular to
  // specific type of node i.e Iop or GeoOp
  auto renderFunc = [&](mlserver::RespondWrapper& responseWrapper, std::string& errorMsg)
    { return drawGeometry(responseWrapper, scene, out, errorMsg); };

  if (rebuild(Mask_Primitives) || rebuild(Mask_Points))
    tryInfer(renderFunc);
}

bool MLClientGeo::drawGeometry(mlserver::RespondWrapper& responseWrapper, Scene& scene, GeometryList& out, std::string& errorMsg)
{
  // Sanity check, make sure the response actually contains an object.
  if(!responseWrapper.has_r2() || responseWrapper.r2().num_objects() == 0) {
    errorMsg = "No object found in message response.";
    return false;
  }

  // Validate ourself before proceeding, this ensures if this is being invoked by a button press
  // then it's set up correctly. This will return immediately if it's already set up.
  if( !tryValidate(/*for_real*/true) ) {
    errorMsg = "Could not set-up node correctly.";
    return false;
  }

  const mlserver::FieldValuePair* geo = nullptr;
  for (auto & field : responseWrapper.r2().objects()) {
    // Points should go first, and there're should be at least coordinates
    // of 1 point otherewise there would be nothing to draw
    if (field.name() == "Geo" && field.values().size() && field.values(0).float_attributes().size() &&
        field.values(0).float_attributes(0).name() == "points" &&
        field.values(0).float_attributes(0).values().size() > 3) {
      geo = &field.values(0);
      break;
    }
  }

  if (!geo) {
    errorMsg = "No geometry data found in message response.";
    return false;
  }

  /*! The current code assumes that the received data contains at least
      position coordinates of one points, optionally it can contain
      indices of triangulated mesh and attributes such as uv and normals.
  */

  // Collect point positions, face indices and attributes.
  const google::protobuf::RepeatedField<int>* facesIn = nullptr;
  for (auto & attr : geo->int_attributes()) {
    if (attr.name() == "indices")
      facesIn = &attr.values();
  }

  const google::protobuf::RepeatedField<float>* pointsIn = nullptr;
  for (auto& attr : geo->float_attributes()) {
    if (attr.name() == "points") {
      pointsIn = &attr.values();
      continue;
    }
    auto it = floatAttribs.find(attr.name());
    if (it != floatAttribs.end())
      it->second.data = &attr.values();
  }

  // Check the collected data for consistency.
  if ((floatAttribs["uv Group_Vertices"].data && facesIn &&
       floatAttribs["uv Group_Vertices"].data->size() != facesIn->size() * 4) ||
      (floatAttribs["N Group_Vertices"].data && facesIn &&
       floatAttribs["N Group_Vertices"].data->size() != facesIn->size() * 3)) {
    errorMsg = "Geometry data is inconsistent";
    return false;
  }

  int obj = 0;
  int numPoints = pointsIn->size() / 3;

  if (rebuild(Mask_Primitives)) {
    out.delete_objects();
    out.add_object(obj);

    if (facesIn) {
      // If face indices exist in response data, create PolyMesh.
      int numCorners = facesIn->size();
      auto mesh = new PolyMesh(numCorners, numCorners / 3);
      for(int i = 2; i < numCorners; i += 3) {
        int corners[3] = { (*facesIn)[i - 2], (*facesIn)[i - 1], (*facesIn)[i] };
        mesh->add_face(3, corners);
      }
      out.add_primitive(obj, mesh);
    } else {
      // Otherwise just draw points.
      for (int i = 0; i < numPoints; i++)
        out.add_primitive(obj, new Point(Point::RenderMode::DISC, 5.0, i));
    }
  }

  if (rebuild(Mask_Primitives) || rebuild(Mask_Points)) {
    PointList* pointsOut = out.writable_points(obj);
    pointsOut->resize(numPoints);

    // Reinterpret a raw flat array into a raw 2d array
    // for copying into DDImage PointList in one step.
    const float (&points2d)[numPoints][3] = *reinterpret_cast<const float (*)[numPoints][3]>(pointsIn->data());
    std::copy(points2d, points2d + numPoints, pointsOut->begin());
  }

  if (rebuild(Mask_Primitives) || rebuild(Mask_Attributes)) {
    // Use floatAttribs map to asign data from the response to the GeoInfo attributes.
    for (auto& pair : floatAttribs) {
      FloatAttrib& attr = pair.second;
      // Ignore attribute if there's no data.
      if (!attr.data)
        continue;

      Attribute* attrOut = out.writable_attribute(obj, attr.group, attr.name.c_str(), attr.type);
      assert(attrOut);
      int size = attrOut->size();
      int dim = attr.dim;

      // Assign data to an attribute accorfing to its type.
      for(int i = 0; i < size; i++) {
        float item[dim];
        for (int j = 0; j < dim; j++)
          item[j] = (*attr.data)[i * dim + j];
        if (attr.type == NORMAL_ATTRIB) {
          attrOut->normal(i).set(item[0], item[1], item[2]);
        } else if (attr.type == VECTOR4_ATTRIB) {
          attrOut->vector4(i).set(item[0], item[1], item[2], item[3]);
        } else if (attr.type == VECTOR3_ATTRIB) {
          attrOut->vector3(i).set(item[0], item[1], item[2]);
        }
      }
    }
  }

  // If this is reached here, return true for success
  return true;
}

//! The op needs both the GeoOp knobs and the MLClientMixin knobs.
void MLClientGeo::knobs(Knob_Callback f)
{
  SourceGeo::knobs(f);
  MLClientMixin<SourceGeo>::knobs(f);
}

//! Return 1 if knobs from one or both superclasses have changed.
int MLClientGeo::knob_changed(Knob* k)
{
  return SourceGeo::knob_changed(k) || MLClientMixin<SourceGeo>::knob_changed(k);
}

void MLClientGeo::get_geometry_hash()
{
  SourceGeo::get_geometry_hash();
  geo_hash[Group_Primitives].append(input_iop()->hash());
  geo_hash[Group_Attributes].append(default_material_iop()->hash());
}

//! Compute specific number of inputs, GeoOp uses one input as a material.
int MLClientGeo::computed_inputs() {
  return 1;
}

//! Return a format of for sending to the server, GeoOp hasn't got output format.
Box MLClientGeo::getFormat() const
{
  return input_iop()->format();
}

//! Return the name of the class.
const char* MLClientGeo::Class() const
{
  return MLClientGeo::kClassName;
}

const char* MLClientGeo::node_help() const
{ 
  return MLClientGeo::kHelpString;
}
