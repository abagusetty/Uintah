//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : Colormap2.cc
//    Author : Milan Ikits
//    Date   : Mon Jul  5 18:33:29 2004

#include <Core/Util/NotFinished.h>
#include <Core/Persistent/Persistent.h>
#include <Packages/Volume/Core/Datatypes/Colormap2.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;

namespace Volume {

static Persistent* maker()
{
  return scinew Colormap2;
}

PersistentTypeID Colormap2::type_id("Colormap2", "Datatype", maker);

#define Colormap2_VERSION 1
void Colormap2::io(Piostream&)
{
  NOT_FINISHED("Colormap2::io(Piostream&)");
}

Colormap2::Colormap2()
  : dirty_(false), lock_("Colormap2 lock")
{}

Colormap2::~Colormap2()
{
  // TODO:  Delete widgets here.
}

Array3<float>&
Colormap2::array()
{
  return array_;
}

bool
Colormap2::dirty()
{
  return dirty_;
}

void
Colormap2::set_dirty(bool b)
{
  dirty_ = b;
}

void
Colormap2::lock_array()
{
  lock_.lock();
}

void
Colormap2::unlock_array()
{
  lock_.unlock();
}

} // End namespace Volume
