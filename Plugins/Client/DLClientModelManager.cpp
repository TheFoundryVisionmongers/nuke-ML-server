// Copyright (c) 2019 The Foundry Visionmongers Ltd.  All Rights Reserved.
// This is strictly non-commercial.

#include "DLClientModelManager.h"

DLClientModelManager::DLClientModelManager()
{ }

DLClientModelManager::~DLClientModelManager()
{ }


void DLClientModelManager::parseOptions(const dlserver::Model& m)
{
  clear();

  for (int i = 0, endI = m.bool_options_size(); i < endI; i++) {
    dlserver::BoolAttrib option;
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
    dlserver::IntAttrib option;
    option = m.int_options(i);
    _dynamicIntValues.push_back(option.values(0));
    _dynamicIntNames.push_back(option.name());
  }
  for (int i = 0, endI = m.float_options_size(); i < endI; i++) {
    dlserver::FloatAttrib option;
    option = m.float_options(i);
    _dynamicFloatValues.push_back(option.values(0));
    _dynamicFloatNames.push_back(option.name());
  }
  for (int i = 0, endI = m.string_options_size(); i < endI; i++) {
    dlserver::StringAttrib option;
    option = m.string_options(i);
    _dynamicStringValues.push_back(option.values(0));
    _dynamicStringNames.push_back(option.name());
  }
  for (int i = 0, endI = m.button_options_size(); i < endI; i++) {
    dlserver::BoolAttrib option;
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

void DLClientModelManager::updateOptions(dlserver::Model& m)
{
  m.clear_bool_options();
  for (int i = 0; i < _dynamicBoolValues.size(); i++) {
    dlserver::BoolAttrib* option = m.add_bool_options();
    option->set_name(_dynamicBoolNames[i]);
    option->add_values(_dynamicBoolValues[i]);
  }

  m.clear_int_options();
  for (int i = 0; i < _dynamicIntValues.size(); i++) {
    dlserver::IntAttrib* option = m.add_int_options();
    option->set_name(_dynamicIntNames[i]);
    option->add_values(_dynamicIntValues[i]);
  }

  m.clear_float_options();
  for (int i = 0; i < _dynamicFloatValues.size(); i++) {
    dlserver::FloatAttrib* option = m.add_float_options();
    option->set_name(_dynamicFloatNames[i]);
    option->add_values(_dynamicFloatValues[i]);
  }

  m.clear_string_options();
  for (int i = 0; i < _dynamicStringValues.size(); i++) {
    dlserver::StringAttrib* option = m.add_string_options();
    option->set_name(_dynamicStringNames[i]);
    option->add_values(_dynamicStringValues[i]);
  }

  m.clear_button_options();
  for (int i = 0; i < _dynamicButtonValues.size(); i++) {
    dlserver::BoolAttrib* option = m.add_button_options();
    option->set_name(_dynamicButtonNames[i]);
    option->add_values(_dynamicButtonValues[i]);
  }
}

int DLClientModelManager::getNumOfFloats() const
{ 
  return _dynamicFloatValues.size();
}

int DLClientModelManager::getNumOfInts() const
{ 
  return _dynamicIntValues.size();
}

int DLClientModelManager::getNumOfBools() const
{ 
  return _dynamicBoolValues.size();
}

int DLClientModelManager::getNumOfStrings() const
{ 
  return _dynamicStringValues.size();
}

int DLClientModelManager::getNumOfButtons() const
{
  return _dynamicButtonValues.size();
}

std::string DLClientModelManager::getDynamicBoolName(int idx)
{ 
  return _dynamicBoolNames[idx];
}

std::string DLClientModelManager::getDynamicFloatName(int idx)
{ 
  return _dynamicFloatNames[idx];
}

std::string DLClientModelManager::getDynamicIntName(int idx)
{ 
  return _dynamicIntNames[idx];
}

std::string DLClientModelManager::getDynamicStringName(int idx)
{ 
  return _dynamicStringNames[idx];
}

std::string DLClientModelManager::getDynamicButtonName(int idx)
{ 
  return _dynamicButtonNames[idx];
}

float* DLClientModelManager::getDynamicFloatValue(int idx)
{ 
  return &_dynamicFloatValues[idx];
}

int* DLClientModelManager::getDynamicIntValue(int idx)
{ 
  return &_dynamicIntValues[idx];
}

bool* DLClientModelManager::getDynamicBoolValue(int idx)
{ 
  return (bool* )&_dynamicBoolValues[idx];
}

std::string* DLClientModelManager::getDynamicStringValue(int idx)
{ 
  return &_dynamicStringValues[idx];
}

bool* DLClientModelManager::getDynamicButtonValue(int idx)
{ 
  return (bool* )&_dynamicButtonValues[idx];
}

void DLClientModelManager::setDynamicButtonValue(int idx, int value)
{
  _dynamicButtonValues[idx] = value;
}

void DLClientModelManager::clear()
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