//static char *id="@(#) $Id$";

/*
 *  TransformField.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
#include <tcl.h>
#include <tk.h>
#include <iostream>
using std::cerr;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;
using namespace SCICore::Containers;

class TransformField : public Module {
    ScalarFieldIPort *iport;
    ScalarFieldOPort *oport;
    ScalarFieldHandle sfOH;		// output bitFld
    ScalarFieldHandle last_sfIH;	// last input fld

    TCLstring xmap;
    TCLstring ymap;
    TCLstring zmap;

    int xxmap;
    int yymap;
    int zzmap;

     int tcl_execute;
public:
    TransformField(const clString& id);
    virtual ~TransformField();
    virtual void execute();
    void set_str_vars();
    void tcl_command( TCLArgs&, void * );
    clString makeAbsMapStr();
};

Module* make_TransformField(const clString& id) {
  return new TransformField(id);
}

TransformField::TransformField(const clString& id)
: Module("TransformField", id, Source), tcl_execute(0),
  xmap("xmap", id, this), ymap("ymap", id, this), zmap("zmap", id, this),
  xxmap(1), yymap(2), zzmap(3)
{
    // Create the input port
    iport = scinew ScalarFieldIPort(this, "SFRG", ScalarFieldIPort::Atomic);
    add_iport(iport);
    oport = scinew ScalarFieldOPort(this, "SFRG",ScalarFieldIPort::Atomic);
    add_oport(oport);
}

TransformField::~TransformField()
{
}

void TransformField::execute()
{
    ScalarFieldHandle sfIH;
    iport->get(sfIH);
    if (!sfIH.get_rep()) return;
    if (!tcl_execute && (sfIH.get_rep() == last_sfIH.get_rep())) return;
    ScalarFieldRGBase *sfrgb;
    if ((sfrgb=sfIH->getRGBase()) == 0) return;

    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGshort *ifs, *ofs;
    ScalarFieldRGuchar *ifu, *ofu;
    ScalarFieldRGchar *ifc, *ofc;
    
    ScalarFieldRGBase *ofb;

    ifd=sfrgb->getRGDouble();
    iff=sfrgb->getRGFloat();
    ifi=sfrgb->getRGInt();
    ifs=sfrgb->getRGShort();
    ifu=sfrgb->getRGUchar();
    ifc=sfrgb->getRGChar();
    
    ofd=0;
    off=0;
    ofs=0;
    ofi=0;
    ofc=0;

    if (sfIH.get_rep() != last_sfIH.get_rep()) {	// new field came in
	int nx=sfrgb->nx;
	int ny=sfrgb->ny;
	int nz=sfrgb->nz;
	Point min;
	Point max;
	sfrgb->get_bounds(min, max);
	clString map=makeAbsMapStr();

	int basex, basey, basez, incrx, incry, incrz;
	if (xxmap>0) {basex=0; incrx=1;} else {basex=nx-1; incrx=-1;}
	if (yymap>0) {basey=0; incry=1;} else {basey=ny-1; incry=-1;}
	if (zzmap>0) {basez=0; incrz=1;} else {basez=nz-1; incrz=-1;}

 	if (map=="123") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(nx,ny,nz);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(nx,ny,nz);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(nx,ny,nz);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(nx,ny,nz);
		ofb=ofs;
	    } else if (ifu) {
		ofu=scinew ScalarFieldRGuchar(); 
		ofu->resize(nx,ny,nz);
		ofb=ofu;
	    } else if (ifc) {
		ofc=scinew ScalarFieldRGchar(); 
		ofc->resize(nx,ny,nz);
		ofb=ofc;
	    }
	    ofb->set_bounds(Point(min.x(), min.y(), min.z()), 
			    Point(max.x(), max.y(), max.z()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			if (ofd) ofd->grid(i,j,k)=ifd->grid(ii,jj,kk);
			else if (off) off->grid(i,j,k)=iff->grid(ii,jj,kk);
			else if (ofi) ofi->grid(i,j,k)=ifi->grid(ii,jj,kk);
			else if (ofs) ofs->grid(i,j,k)=ifs->grid(ii,jj,kk);
			else if (ofu) ofu->grid(i,j,k)=ifu->grid(ii,jj,kk);
	                else if (ofc) ofc->grid(i,j,k)=ifc->grid(ii,jj,kk);
	    
	} else if (map=="132") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(nx,nz,ny);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(nx,nz,ny);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(nx,nz,ny);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(nx,nz,ny);
		ofb=ofs;
	    } else if (ifu) {
		ofu=scinew ScalarFieldRGuchar(); 
		ofu->resize(nx,nz,ny);
		ofb=ofu;
	    } else if (ifc) { 
                ofc=scinew ScalarFieldRGchar();
                ofc->resize(nx,nz,ny);         
                ofb=ofc;                       
            }
	    ofb->set_bounds(Point(min.x(), min.z(), min.y()), 
                            Point(max.x(), max.z(), max.y()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			if (ofd) ofd->grid(i,k,j)=ifd->grid(ii,jj,kk);
                        else if (off) off->grid(i,k,j)=iff->grid(ii,jj,kk);
                        else if (ofi) ofi->grid(i,k,j)=ifi->grid(ii,jj,kk);
                        else if (ofs) ofs->grid(i,k,j)=ifs->grid(ii,jj,kk);
                        else if (ofc) ofc->grid(i,k,j)=ifc->grid(ii,jj,kk);
                        else if (ofu) ofu->grid(i,j,k)=ifu->grid(ii,jj,kk);
	} else if (map=="213") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(ny,nx,nz);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(ny,nx,nz);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(ny,nx,nz);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(ny,nx,nz);
		ofb=ofs;
	    } else if (ifu) {
		ofu=scinew ScalarFieldRGuchar(); 
		ofu->resize(ny,nx,nz);
		ofb=ofu;
	    } else if (ifc) {
                ofc=scinew ScalarFieldRGchar(); 
                ofc->resize(ny,nx,nz);
                ofb=ofc;
            }
            ofb->set_bounds(Point(min.y(), min.x(), min.z()), 
                            Point(max.y(), max.x(), max.z()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
                        if (ofd) ofd->grid(i,j,k)=ifd->grid(jj,ii,kk);
                        else if (off) off->grid(j,i,k)=iff->grid(ii,jj,kk);
                        else if (ofi) ofi->grid(j,i,k)=ifi->grid(ii,jj,kk);
                        else if (ofs) ofs->grid(j,i,k)=ifs->grid(ii,jj,kk);
                        else if (ofc) ofc->grid(j,i,k)=ifc->grid(ii,jj,kk);
                        else if (ofu) ofu->grid(j,i,k)=ifu->grid(ii,jj,kk);
	} else if (map=="231") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(ny,nz,nx);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(ny,nz,nx);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(ny,nz,nx);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(ny,nz,nx);
		ofb=ofs;
	    } else if (ifu) {
		ofu=scinew ScalarFieldRGuchar(); 
		ofu->resize(ny,nz,nx);
		ofb=ofu;
	    } else if (ifc) {
                ofc=scinew ScalarFieldRGchar(); 
                ofc->resize(ny,nz,nx);
                ofb=ofc;
            }
            ofb->set_bounds(Point(min.y(), min.z(), min.x()), 
                            Point(max.y(), max.z(), max.x()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			if (ofd) ofd->grid(i,j,k)=ifd->grid(ii,jj,kk);
			else if (off) off->grid(j,k,i)=iff->grid(jj,kk,ii);
			else if (ofi) ofi->grid(j,k,i)=ifi->grid(jj,kk,ii);
			else if (ofs) ofs->grid(j,k,i)=ifs->grid(jj,kk,ii);
                        else if (ofc) ofc->grid(j,k,i)=ifc->grid(ii,jj,kk);
			else if (ofu) ofu->grid(j,k,i)=ifu->grid(jj,kk,ii);
	} else if (map=="312") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(nz,nx,ny);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(nz,nx,ny);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(nz,nx,ny);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(nz,nx,ny);
		ofb=ofs;
	    } else if (ifu) {
		ofu=scinew ScalarFieldRGuchar(); 
		ofu->resize(nz,nx,ny);
		ofb=ofu;
	    } else if (ifc) {
                ofc=scinew ScalarFieldRGchar(); 
                ofc->resize(nz,nx,ny);
                ofb=ofc;
            }
            ofb->set_bounds(Point(min.z(), min.x(), min.y()), 
                            Point(max.z(), max.x(), max.y()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
                        if (ofd) ofd->grid(k,i,j)=ifd->grid(ii,jj,kk);
                        else if (off) off->grid(k,i,j)=iff->grid(ii,jj,kk);
                        else if (ofi) ofi->grid(k,i,j)=ifi->grid(ii,jj,kk);
                        else if (ofs) ofs->grid(k,i,j)=ifs->grid(ii,jj,kk);
                        else if (ofc) ofc->grid(k,i,j)=ifc->grid(ii,jj,kk);
	} else if (map=="321") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(nz,ny,nx);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(nz,ny,nx);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(nz,ny,nx);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(nz,ny,nx);
		ofb=ofs;
	    } else if (ifu) {
		ofu=scinew ScalarFieldRGuchar(); 
		ofu->resize(nz,ny,nx);
		ofb=ofu;
	    } else if (ifc) {
                ofc=scinew ScalarFieldRGchar(); 
                ofc->resize(nz,ny,nx);
                ofb=ofc;
            }
            ofb->set_bounds(Point(min.z(), min.y(), min.x()), 
                            Point(max.z(), max.y(), max.x()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
                        if (ofd) ofd->grid(k,j,i)=ifd->grid(ii,jj,kk);
                        else if (off) off->grid(k,j,i)=iff->grid(ii,jj,kk);
                        else if (ofi) ofi->grid(k,j,i)=ifi->grid(ii,jj,kk);
                        else if (ofs) ofs->grid(k,j,i)=ifs->grid(ii,jj,kk);
                        else if (ofc) ofc->grid(k,j,i)=ifc->grid(ii,jj,kk);
                        else if (ofu) ofu->grid(k,j,i)=ifu->grid(ii,jj,kk);
	    

//	    outFld->resize(nz,ny,nx);
//	    outFld->grid(i,j,k)=sfrg->grid(kk,jj,ii);

	} else {
	    cerr << "ERROR: TransformField::execute() doesn't recognize map code: "<<map<<"\n";
	}
	sfOH=ofb;
    }
    oport->send(sfOH);
    tcl_execute=0;
}
	    
void TransformField::set_str_vars() {
    if (xxmap==1) xmap.set("x <- x+");
    if (xxmap==-1) xmap.set("x <- x-");
    if (xxmap==2) xmap.set("x <- y+");
    if (xxmap==-2) xmap.set("x <- y-");
    if (xxmap==3) xmap.set("x <- z+");
    if (xxmap==-3) xmap.set("x <- z-");
    if (yymap==1) ymap.set("y <- x+");
    if (yymap==-1) ymap.set("y <- x-");
    if (yymap==2) ymap.set("y <- y+");
    if (yymap==-2) ymap.set("y <- y-");
    if (yymap==3) ymap.set("y <- z+");
    if (yymap==-3) ymap.set("y <- z-");
    if (zzmap==1) zmap.set("z <- x+");
    if (zzmap==-1) zmap.set("z <- x-");
    if (zzmap==2) zmap.set("z <- y+");
    if (zzmap==-2) zmap.set("z <- y-");
    if (zzmap==3) zmap.set("z <- z+");
    if (zzmap==-3) zmap.set("z <- z-");
}

clString TransformField::makeAbsMapStr() {
    return to_string(Abs(xxmap))+to_string(Abs(yymap))+to_string(Abs(zzmap));
}

void TransformField::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "send") {
	tcl_execute=1;
	want_to_execute();
    } else if (args[1] == "flipx") {
	reset_vars();
	xxmap*=-1;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "flipy") {
	reset_vars();
	yymap*=-1;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "flipz") {
	reset_vars();
	zzmap*=-1;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "cyclePos") {
	reset_vars();
	int tmp=xxmap;
	xxmap=yymap;
	yymap=zzmap;
	zzmap=tmp;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "cycleNeg") {
	reset_vars();
	int tmp=zzmap;
	zzmap=yymap;
	yymap=xxmap;
	xxmap=tmp;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "reset") {
	reset_vars();
	xxmap=1;
	yymap=2;
	zzmap=3;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "swapXY") {
	reset_vars();
	int tmp=xxmap;
	xxmap=yymap;
	yymap=tmp;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "swapYZ") {
	reset_vars();
	int tmp=yymap;
	yymap=zzmap;
	zzmap=tmp;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "swapXZ") {
	reset_vars();
	int tmp=xxmap;
	xxmap=zzmap;
	zzmap=tmp;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else {
        Module::tcl_command(args, userdata);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.7  2000/02/08 21:45:28  kuzimmer
// stuff for transforming and type changes of scalarfieldRGs
//
// Revision 1.6  1999/10/07 02:06:49  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:49  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:47  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:44  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:30  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:44  mcq
// Initial commit
//
// Revision 1.2  1999/04/28 20:51:13  dav
// deleted some files that are dependent on DaveW files
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
