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

#ifndef MLClientModelManager_H
#define MLClientModelManager_H

#include <string>
#include <vector>
#include "message.pb.h"

#include "DDImage/Op.h"
#include "DDImage/Knobs.h"

class MLClientModelManager;

//! The role of this custom knob is to serialise and to store the selected model and its parameters.
//! As these exist as dynamic knobs, this is to workaround the fact that we would need to know about these knobs 
//! in advance to save them the usual way. 
class MLClientModelKnob : public DD::Image::Knob
{
  public:
    MLClientModelKnob(DD::Image::Knob_Closure* kc, DD::Image::Op* op, const char* name);

    // Knob overrides.
    const char* Class() const override;
    bool not_default () const override;
    //! Serialises the currently selected model and its parameters as follows:
    //! {model:modelName;param1:value1;param2:value2}
    void to_script (std::ostream &out, const DD::Image::OutputContext *, bool quote) const override;
    //! Deserialises the saved model and its parameters.
    //! The model can then be retreived with getModel() 
    //! and the dictionary of parameters with getParameters().
    bool from_script(const char * src) override;

    std::string getModel() const;
    const std::map<std::string, std::string>& getParameters() const;

  private:
    //! Serialises the dynamic knobs to the given output stream.
    //! This function is generic for the Ints, Floats and Bools knobs
    //! provided that the corresponding getNumOfT and getDynamicTName
    //! functions are given. 
    void toScriptT(MLClientModelManager& mlManager, std::ostream &out, 
        int (MLClientModelManager::*getNum)() const, 
        std::string (MLClientModelManager::*getDynamicName)(int)) const;
    //! Serialises the dynamic knobs containing strings to the given output stream.
    void toScriptStrings(MLClientModelManager& mlManager, std::ostream &out) const;

  private:
    DD::Image::Op* _op;
    std::string _model;
    std::map<std::string, std::string> _parameters;
};

//! Class to parse and store knobs for a given model.
class MLClientModelManager
{
  public:
    explicit MLClientModelManager(DD::Image::Op* parent);
    ~MLClientModelManager();

    // Getters of the class
    int getNumOfFloats() const;
    int getNumOfInts() const;
    int getNumOfBools() const;
    int getNumOfStrings() const;
    int getNumOfButtons() const;

    std::string getDynamicBoolName(int idx);
    std::string getDynamicFloatName(int idx);
    std::string getDynamicIntName(int idx);
    std::string getDynamicStringName(int idx);
    std::string getDynamicButtonName(int idx);

    float* getDynamicFloatValue(int idx);
    int* getDynamicIntValue(int idx);
    bool* getDynamicBoolValue(int idx);
    std::string* getDynamicStringValue(int idx);
    bool* getDynamicButtonValue(int idx);
    void setDynamicButtonValue(int idx, int value);

    void clear();
    //! Parse the model options from the ML server.
    void parseOptions(const mlserver::Model& m);
    //! Update any current options from any changes to the ML server.
    void updateOptions(mlserver::Model& m);

  private:
    DD::Image::Op* _parent;
    std::vector<int> _dynamicBoolValues;
    std::vector<int> _dynamicIntValues;
    std::vector<float> _dynamicFloatValues;
    std::vector<std::string> _dynamicStringValues;
    std::vector<int> _dynamicButtonValues;
    std::vector<std::string> _dynamicBoolNames;
    std::vector<std::string> _dynamicIntNames;
    std::vector<std::string> _dynamicFloatNames;
    std::vector<std::string> _dynamicStringNames;
    std::vector<std::string> _dynamicButtonNames;
};

#endif
