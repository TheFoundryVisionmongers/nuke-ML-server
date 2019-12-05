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

#include "MLClientModelManager.h"
#include "DDImage/Knob.h"

MLClientModelManager::MLClientModelManager(DD::Image::Op* parent)
: _parent(parent)
{ }

MLClientModelManager::~MLClientModelManager()
{ }

//! Parse options from the server model /m to the MLClientModelManager
void MLClientModelManager::parseOptions(const mlserver::Model& m)
{
  clear();

  for (int i = 0, endI = m.bool_options_size(); i < endI; i++) {
    mlserver::BoolAttrib option;
    option = m.bool_options(i);
    if (option.values(0)) {
      _dynamicBoolValues.push_back(1);
    }
    else {
      _dynamicBoolValues.push_back(0);
    }
    _dynamicBoolNames.push_back(option.name());
  }
  for (int i = 0, endI = m.int_options_size(); i < endI; i++) {
    mlserver::IntAttrib option;
    option = m.int_options(i);
    _dynamicIntValues.push_back(option.values(0));
    _dynamicIntNames.push_back(option.name());
  }
  for (int i = 0, endI = m.float_options_size(); i < endI; i++) {
    mlserver::FloatAttrib option;
    option = m.float_options(i);
    _dynamicFloatValues.push_back(option.values(0));
    _dynamicFloatNames.push_back(option.name());
  }
  for (int i = 0, endI = m.string_options_size(); i < endI; i++) {
    mlserver::StringAttrib option;
    option = m.string_options(i);
    _dynamicStringValues.push_back(option.values(0));
    _dynamicStringNames.push_back(option.name());
  }
  for (int i = 0, endI = m.button_options_size(); i < endI; i++) {
    mlserver::BoolAttrib option;
    option = m.button_options(i);
    if (option.values(0)) {
      _dynamicButtonValues.push_back(1);
    }
    else {
      _dynamicButtonValues.push_back(0);
    }
    _dynamicButtonNames.push_back(option.name());
  }
}

//! Use current knob values to update options on the server model /m
//! in order to later request inference on this model
void MLClientModelManager::updateOptions(mlserver::Model& m)
{
  m.clear_bool_options();
  for (int i = 0; i < _dynamicBoolValues.size(); i++) {
    mlserver::BoolAttrib* option = m.add_bool_options();
    option->set_name(_dynamicBoolNames[i]);
    DD::Image::Knob* k = _parent->knob(_dynamicBoolNames[i].c_str());
    bool val = false;
    if (k != nullptr) {
      val = k->get_value();
    }
    option->add_values(val);
  }

  m.clear_int_options();
  for (int i = 0; i < _dynamicIntValues.size(); i++) {
    mlserver::IntAttrib* option = m.add_int_options();
    option->set_name(_dynamicIntNames[i]);
    DD::Image::Knob* k = _parent->knob(_dynamicIntNames[i].c_str());
    int val = 0;
    if (k != nullptr) {
      val = k->get_value();
    }
    option->add_values(val);
  }

  m.clear_float_options();
  for (int i = 0; i < _dynamicFloatValues.size(); i++) {
    mlserver::FloatAttrib* option = m.add_float_options();
    option->set_name(_dynamicFloatNames[i]);
    DD::Image::Knob* k = _parent->knob(_dynamicFloatNames[i].c_str());
    float val = 0.0f;
    if (k != nullptr) {
      val = k->get_value();
    }
    option->add_values(val);
  }

  m.clear_string_options();
  for (int i = 0; i < _dynamicStringValues.size(); i++) {
    mlserver::StringAttrib* option = m.add_string_options();
    option->set_name(_dynamicStringNames[i]);
    DD::Image::Knob* k = _parent->knob(_dynamicStringNames[i].c_str());
    const char* val = "";
    if(k != nullptr) {
      val = k->get_text();
      if (val==nullptr) {
        val = "";
      }
    }
    option->add_values(val);
  }

  m.clear_button_options();
  for (int i = 0; i < _dynamicButtonValues.size(); i++) {
    mlserver::BoolAttrib* option = m.add_button_options();
    option->set_name(_dynamicButtonNames[i]);
    // Get member value instead of knob value to catch button push
    option->add_values(_dynamicButtonValues[i]);
  }
}

int MLClientModelManager::getNumOfFloats() const
{
  return _dynamicFloatValues.size();
}

int MLClientModelManager::getNumOfInts() const
{
  return _dynamicIntValues.size();
}

int MLClientModelManager::getNumOfBools() const
{
  return _dynamicBoolValues.size();
}

int MLClientModelManager::getNumOfStrings() const
{
  return _dynamicStringValues.size();
}

int MLClientModelManager::getNumOfButtons() const
{
  return _dynamicButtonValues.size();
}

std::string MLClientModelManager::getDynamicBoolName(int idx)
{
  return _dynamicBoolNames[idx];
}

std::string MLClientModelManager::getDynamicFloatName(int idx)
{
  return _dynamicFloatNames[idx];
}

std::string MLClientModelManager::getDynamicIntName(int idx)
{
  return _dynamicIntNames[idx];
}

std::string MLClientModelManager::getDynamicStringName(int idx)
{
  return _dynamicStringNames[idx];
}

std::string MLClientModelManager::getDynamicButtonName(int idx)
{
  return _dynamicButtonNames[idx];
}

float* MLClientModelManager::getDynamicFloatValue(int idx)
{
  return &_dynamicFloatValues[idx];
}

int* MLClientModelManager::getDynamicIntValue(int idx)
{
  return &_dynamicIntValues[idx];
}

bool* MLClientModelManager::getDynamicBoolValue(int idx)
{
  return (bool*)&_dynamicBoolValues[idx];
}

std::string* MLClientModelManager::getDynamicStringValue(int idx)
{
  return &_dynamicStringValues[idx];
}

bool* MLClientModelManager::getDynamicButtonValue(int idx)
{
  return (bool*)&_dynamicButtonValues[idx];
}

void MLClientModelManager::setDynamicButtonValue(int idx, int value)
{
  _dynamicButtonValues[idx] = value;
}

void MLClientModelManager::clear()
{
  _dynamicBoolValues.clear();
  _dynamicIntValues.clear();
  _dynamicFloatValues.clear();
  _dynamicStringValues.clear();
  _dynamicButtonValues.clear();

  _dynamicBoolNames.clear();
  _dynamicIntNames.clear();
  _dynamicFloatNames.clear();
  _dynamicStringNames.clear();
  _dynamicButtonNames.clear();
}
