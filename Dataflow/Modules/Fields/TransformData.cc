/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  TransformData: Unary field data operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Fields/TransformData.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Containers/HashTable.h>
#include <iostream>

namespace SCIRun {

class TransformData : public Module
{
private:
  GuiString function_;
  GuiString outputdatatype_;

public:
  TransformData(GuiContext* ctx);
  virtual ~TransformData();
  virtual void execute();
};


DECLARE_MAKER(TransformData)


TransformData::TransformData(GuiContext* ctx)
  : Module("TransformData", ctx, Filter,"FieldsData", "SCIRun"),
    function_(ctx->subVar("function")),
    outputdatatype_(ctx->subVar("outputdatatype"))
{
}


TransformData::~TransformData()
{
}


void
TransformData::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("Input field is empty.");
    return;
  }

  string outputdatatype = outputdatatype_.get();
  if (outputdatatype == "input")
  {
    outputdatatype = ifieldhandle->get_type_description(1)->get_name();
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->order_type_description();
  const string oftn = ifieldhandle->get_type_description(0)->get_name() +
    "<" + outputdatatype + "> ";
  int hoffset = 0;
  Handle<TransformDataAlgo> algo;

  // remove trailing white-space from the function string
  string func = function_.get();
  while (func.size() && isspace(func[func.size()-1]))
  {
    func.resize(func.size()-1);
  }

  while (1)
  {
    CompileInfoHandle ci =
      TransformDataAlgo::get_compile_info(ftd, oftn, ltd, func, hoffset);
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      error("Your function would not compile.");
      gui->eval(id + " compile_error "+ci->filename_);
      DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
      return;
    }
    if (algo->identify() == func)
    {
      break;
    }
    hoffset++;
  }

  FieldHandle ofieldhandle = algo->execute(ifieldhandle);

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }
  ofield_port->send(ofieldhandle);
}


CompileInfoHandle
TransformDataAlgo::get_compile_info(const TypeDescription *field_td,
				    string ofieldtypename,
				    const TypeDescription *loc_td,
				    string function,
				    int hashoffset)

{
  unsigned int hashval = Hash(function, 0x7fffffff) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("TransformDataInstance" + to_string(hashval));
  static const string base_class_name("TransformDataAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       field_td->get_filename() + "." +
		       to_filename(ofieldtypename) + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       field_td->get_name() + ", " +
		       ofieldtypename + ", " +
		       loc_td->get_name());

  // Code for the function.
  string class_declaration =
    string("template <class IFIELD, class OFIELD, class LOC>\n") +
    "class " + template_name + " : public TransformDataAlgoT<IFIELD, OFIELD, LOC>\n" +
    "{\n" +
    "  virtual void function(typename OFIELD::value_type &result,\n" +
    "                        double x, double y, double z,\n" +
    "                        const typename IFIELD::value_type &v)\n" +
    "  {\n" +
    "    " + function + "\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + string_Cify(function) + "\"); }\n" +
    "};\n";

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_post_include(class_declaration);
  field_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
