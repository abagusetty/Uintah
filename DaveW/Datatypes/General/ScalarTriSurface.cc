//static char *id="@(#) $Id$";

/*
 *  ScalarTriSurface.cc: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <DaveW/Datatypes/General/ScalarTriSurface.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Containers/TrivialAllocator.h>
#include <SCICore/Containers/Queue.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Grid.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Malloc/Allocator.h>

#include <iostream.h>

namespace DaveW {
namespace Datatypes {

static Persistent* make_ScalarTriSurface()
{
    return scinew ScalarTriSurface;
}

PersistentTypeID ScalarTriSurface::type_id("ScalarTriSurface", "Surface", make_ScalarTriSurface);

ScalarTriSurface::ScalarTriSurface()
: TriSurface(ScalarTriSurf)
{
}

ScalarTriSurface::ScalarTriSurface(const TriSurface& ts, const Array1<double>& d)
: TriSurface(ts, ScalarTriSurf), data(d)
{
}

ScalarTriSurface::ScalarTriSurface(const TriSurface& ts)
: TriSurface(ts, ScalarTriSurf)
{
}

ScalarTriSurface::ScalarTriSurface(const ScalarTriSurface& copy)
: TriSurface(copy, ScalarTriSurf), data(copy.data)
{
}

ScalarTriSurface::~ScalarTriSurface() {
}
#define ScalarTriSurface_VERSION 1

void ScalarTriSurface::io(Piostream& stream) {
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    /*int version=*/
    stream.begin_class("ScalarTriSurface", ScalarTriSurface_VERSION);
    TriSurface::io(stream);
    Pio(stream, data);
    stream.end_class();
}

Surface* ScalarTriSurface::clone()
{
    return scinew ScalarTriSurface(*this);
}

GeomObj* ScalarTriSurface::get_obj(const ColorMapHandle&)
{
    return 0;
}

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/08/23 02:53:00  dmw
// Dave's Datatypes
//
// Revision 1.1  1999/05/03 04:52:03  dmw
// Added and updated DaveW Datatypes/Modules
//
//
