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
//    File   : EditTransferFunc2.cc
//    Author : Milan Ikits
//    Author : Michael Callahan
//    Date   : Thu Jul  8 01:50:58 2004

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Array3.h>

#include <sci_gl.h>
#include <Packages/Volume/Core/Util/Pbuffer.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>
#include <Packages/Volume/Dataflow/Ports/Colormap2Port.h>
#include <Packages/Teem/Core/Datatypes/NrrdData.h>
#include <Packages/Teem/Dataflow/Ports/NrrdPort.h>

#include <Core/Geom/GeomOpenGL.h>
#include <tcl.h>
#include <tk.h>

#include <iostream>

// tcl interpreter corresponding to this module
extern Tcl_Interp* the_interp;

// the OpenGL context structure
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

using namespace SCIRun;
using namespace SCITeem;

namespace Volume {


class EditTransferFunc2 : public Module {

  GLXContext ctx_;
  Display* dpy_;
  Window win_;
  int width_, height_;
  bool button_;
  vector<CM2Widget*> widget_;
  Pbuffer* pbuffer_;
  bool use_pbuffer_;
  
  Nrrd* histo_;
  bool histo_dirty_;
  GLuint histo_tex_;

  Colormap2Handle cmap_;
  bool cmap_dirty_;
  bool cmap_size_dirty_;
  bool cmap_out_dirty_;
  GLuint cmap_tex_;
  
  int pick_widget_; // Which widget is selected.
  int pick_object_; // The part of the widget that is selected.

public:
  EditTransferFunc2(GuiContext* ctx);
  virtual ~EditTransferFunc2();

  virtual void execute();

  void tcl_command(GuiArgs&, void*);

  bool create_histo();
  
  void update();
  void redraw();

  void push(int x, int y, int button);
  void motion(int x, int y);
  void release(int x, int y, int button);
};



DECLARE_MAKER(EditTransferFunc2)

EditTransferFunc2::EditTransferFunc2(GuiContext* ctx)
  : Module("EditTransferFunc2", ctx, Filter, "Visualization", "Volume"),
    ctx_(0), dpy_(0), win_(0), button_(0), pbuffer_(0), use_pbuffer_(true),
    histo_(0), histo_dirty_(false), histo_tex_(0),
    cmap_(new Colormap2),
    cmap_dirty_(true), cmap_size_dirty_(true), cmap_out_dirty_(true), cmap_tex_(0),
    pick_widget_(-1), pick_object_(0)
{
  widget_.push_back(scinew TriangleCM2Widget());
  widget_.push_back(scinew RectangleCM2Widget());
  //widget_.push_back(scinew RectangleCM2Widget());
}


EditTransferFunc2::~EditTransferFunc2()
{
  // Clean up currently unmemorymanaged widgets.
  //for (unsigned int i = 0; i < widget_.size(); i++)
  //{
  //delete widget_[i];
  //}
  //widget_.clear();
}


void
EditTransferFunc2::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("No command for EditTransferFunc");
    return;
  }

  // mouse commands are motion, down, release
  if (args[1] == "mouse") {
    //cerr << "EVENT: mouse" << endl;
    int x, y, b;
    string_to_int(args[3], x);
    string_to_int(args[4], y);
    if (args[2] == "motion") {
      if (button_ == 0) // not buttons down!
	return;
      motion(x, y);
    } else {
      string_to_int(args[5], b); // which button it was
      if (args[2] == "push") {
	push(x, y, b);
      } else {
	release(x, y, b);
      }
    }
  } else if (args[1] == "resize") {
    //cerr << "EVENT: resize" << endl;
    string_to_int(args[2], width_);
    string_to_int(args[3], height_);
    update();
    redraw();
  } else if (args[1] == "expose") {
    //cerr << "EVENT: expose" << endl;
    update();
    redraw();
  } else if (args[1] == "closewindow") {
    //cerr << "EVENT: close" << endl;
    ctx_ = 0;
  } else {
    Module::tcl_command(args, userdata);
  }
}



void
EditTransferFunc2::push(int x, int y, int button)
{
  //cerr << "push: " << x << " " << y << " " << button << endl;

  unsigned int i;

  button_ = button;

  for (i = 0; i < widget_.size(); i++)
  {
    widget_[i]->unselect_all();
  }

  pick_widget_ = -1;
  pick_object_ = 0;
  for (unsigned int i = 0; i < widget_.size(); i++)
  {
    const int tmp = widget_[i]->pick(x, 255-y, 512, 256);
    if (tmp)
    {
      pick_widget_ = i;
      pick_object_ = tmp;
      widget_[i]->select(tmp);
      break;
    }
  }
  update();
  redraw();
}



void
EditTransferFunc2::motion(int x, int y)
{
  //cerr << "motion: " << x << " " << y << endl;

  if (pick_widget_ != -1)
  {
    widget_[pick_widget_]->move(pick_object_, x, 255-y, 512, 256);
    cmap_dirty_ = true;
  }
  update();
  redraw();
}



void
EditTransferFunc2::release(int x, int y, int button)
{
  //cerr << "release: " << x << " " << y << " " << button << endl;

  button_ = 0;
  if (pick_widget_ != -1)
  {
    widget_[pick_widget_]->release(pick_object_, x, 255-y, 512, 256);
    cmap_dirty_ = true;
  }

  update();
  redraw();
}



void
EditTransferFunc2::execute()
{
  //cerr << "execute" << endl;
  
  NrrdIPort* histo_port = (NrrdIPort*)get_iport("Histogram");
  if(histo_port) {
    NrrdDataHandle h;
    histo_port->get(h);
    if(h.get_rep()) {
      if(h->nrrd->dim != 2 && h->nrrd->dim != 3) {
        error("Invalid input dimension.");
      }
      if(histo_ != h->nrrd) {
        histo_ = h->nrrd;
        histo_dirty_ = true;
      }
    } else {
      if(histo_ != 0)
        histo_dirty_ = true;
      histo_ = 0;
    }
  } else {
    if(histo_ != 0)
      histo_dirty_ = true;
    histo_ = 0;
  }

  update();
  redraw();

  Colormap2OPort* cmap_port = (Colormap2OPort*)get_oport("Colormap");
  if(cmap_port) {
    cmap_port->send(cmap_);
  }
}



void
EditTransferFunc2::update()
{
  //cerr << "update" << endl;

  bool do_execute = false;
  
  gui->lock();

  //----------------------------------------------------------------
  // obtain rendering ctx 
  if (!ctx_) {
    const string myname(".ui" + id + ".f.gl.gl");
    Tk_Window tkwin = Tk_NameToWindow(the_interp, ccast_unsafe(myname),
                                      Tk_MainWindow(the_interp));
    if (!tkwin) {
      warning("Unable to locate window!");
      gui->unlock();
      return;
    }
    dpy_ = Tk_Display(tkwin);
    win_ = Tk_WindowId(tkwin);
    ctx_ = OpenGLGetContext(the_interp, ccast_unsafe(myname));
    width_ = Tk_Width(tkwin);
    height_ = Tk_Height(tkwin);
    // check if it was created
    if(!ctx_) {
      error("Unable to obtain OpenGL context!");
      gui->unlock();
      return;
    }
  }
  glXMakeCurrent(dpy_, win_, ctx_);

  if(use_pbuffer_) {
    if (TriangleCM2Widget::Init()
        || RectangleCM2Widget::Init()) {
      use_pbuffer_ = false;
      cerr << "Shaders not supported; switching to software rasterization" << endl;
    }
  }
  
  // create pbuffer
  if(use_pbuffer_ && (!pbuffer_
      || pbuffer_->width() != width_
      || pbuffer_->height() != height_)) {
    if(pbuffer_) {
      pbuffer_->destroy();
      delete pbuffer_;
    }
    pbuffer_ = new Pbuffer(width_, height_, GL_INT, 8, true, GL_FALSE);
    if(pbuffer_->create()) {
      use_pbuffer_ = false;
      pbuffer_->destroy();
      delete pbuffer_;
      pbuffer_ = 0;
      cerr << "Pbuffers not supported; switching to software rasterization" << endl;
    }
  }
  
  //----------------------------------------------------------------
  // update colormap array
  if(use_pbuffer_) {
    pbuffer_->makeCurrent();
    glDrawBuffer(GL_FRONT);
    glViewport(0, 0, width_, height_);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-1.0, -1.0, 0.0);
    glScalef(2.0, 2.0, 2.0);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    
    // Rasterize widgets
    for (unsigned int i = 0; i < widget_.size(); i++)
    {
      widget_[i]->rasterize();
    }

    glDisable(GL_BLEND);
    
    pbuffer_->swapBuffers();
    
    glXMakeCurrent(dpy_, win_, ctx_);

    if (cmap_dirty_)
    {
      cmap_->widgets() = widget_;
      cmap_dirty_ = false;
    }
  } else {
    Array3<float>& cmap = cmap_->array();
    if(width_ != cmap.dim2() || height_ != cmap.dim1()) {
      cmap_size_dirty_ = true;
      cmap_dirty_ = true;
    }

    if (cmap_dirty_) {
      cmap_->lock_array();
      // realloc cmap
      if(cmap_size_dirty_)
        cmap.resize(height_, width_, 4);
      // clear cmap
      for(int i=0; i<cmap.dim1(); i++) {
        for(int j=0; j<cmap.dim2(); j++) {
          cmap(i,j,0) = 0.0;
          cmap(i,j,1) = 0.0;
          cmap(i,j,2) = 0.0;
          cmap(i,j,3) = 0.0;
        }
      }
      // Rasterize widgets
      for (unsigned int i = 0; i < widget_.size(); i++)
      {
	widget_[i]->rasterize(cmap);
      }

      // Update textures
      if(cmap_size_dirty_) {
        if(glIsTexture(cmap_tex_)) {
          glDeleteTextures(1, &cmap_tex_);
          cmap_tex_ = 0;
        }
        glGenTextures(1, &cmap_tex_);
        glBindTexture(GL_TEXTURE_2D, cmap_tex_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cmap.dim2(), cmap.dim1(),
                     0, GL_RGBA, GL_FLOAT, &cmap(0,0,0));
        glBindTexture(GL_TEXTURE_2D, 0);
        cmap_size_dirty_ = false;
      } else {
        glBindTexture(GL_TEXTURE_2D, cmap_tex_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cmap.dim2(), cmap.dim1(),
                     0, GL_RGBA, GL_FLOAT, &cmap(0,0,0));
        glBindTexture(GL_TEXTURE_2D, 0);
      }
      cmap_dirty_ = false;
      cmap_->unlock_array();
      do_execute = true;
    }
  }
  
  //----------------------------------------------------------------
  // update histo tex
  if (histo_dirty_) {
    if(glIsTexture(histo_tex_)) {
      glDeleteTextures(1, &histo_tex_);
      histo_tex_ = 0;
    }
    glGenTextures(1, &histo_tex_);
    glBindTexture(GL_TEXTURE_2D, histo_tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    int axis_size[3];
    nrrdAxisInfoGet_nva(histo_, nrrdAxisInfoSize, axis_size);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, axis_size[histo_->dim-2],
                 axis_size[histo_->dim-1], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                 histo_->data);
    glBindTexture(GL_TEXTURE_2D, 0);
    histo_dirty_ = false;
  }

  gui->unlock();

  if (do_execute) {
    want_to_execute();
  }
}



void
EditTransferFunc2::redraw()
{
  //cerr << "redraw" << endl;

  gui->lock();
  
  //----------------------------------------------------------------
  // draw
  glDrawBuffer(GL_BACK);
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);
  
  glViewport(0, 0, width_, height_);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(-1.0, -1.0, 0.0);
  glScalef(2.0, 2.0, 2.0);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glDisable(GL_CULL_FACE);

  // draw histo
  if(histo_) {
    glActiveTexture(GL_TEXTURE0);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, histo_tex_);
    glColor4f(0.75, 0.75, 0.75, 1.0);
    glBegin(GL_QUADS);
    {
      glTexCoord2f( 0.0,  0.0);
      glVertex2f( 0.0,  0.0);
      glTexCoord2f( 1.0,  0.0);
      glVertex2f( 1.0,  0.0);
      glTexCoord2f( 1.0,  1.0);
      glVertex2f( 1.0,  1.0);
      glTexCoord2f( 0.0,  1.0);
      glVertex2f( 0.0,  1.0);
    }
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
  }

  // draw cmap
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  if(use_pbuffer_) {
    pbuffer_->bind(GL_FRONT);
  } else {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, cmap_tex_);
  }
  glBegin(GL_QUADS);
  {
    glTexCoord2f( 0.0,  0.0);
    glVertex2f( 0.0,  0.0);
    glTexCoord2f( 1.0,  0.0);
    glVertex2f( 1.0,  0.0);
    glTexCoord2f( 1.0,  1.0);
    glVertex2f( 1.0,  1.0);
    glTexCoord2f( 0.0,  1.0);
    glVertex2f( 0.0,  1.0);
  }
  glEnd();
  if(use_pbuffer_) {
    pbuffer_->release(GL_FRONT);
  } else {
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
  }
  glDisable(GL_BLEND);
  
  // draw widgets
  for (unsigned int i = 0; i < widget_.size(); i++)
  {
    widget_[i]->draw();
  }
  
  glXSwapBuffers(dpy_, win_);
  glXMakeCurrent(dpy_, 0, 0);

  gui->unlock();
}


} // end namespace Volume
