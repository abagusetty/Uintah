
/*
 *  HeadLight.h:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_HeadLight_h
#define SCI_Geom_HeadLight_h 1

#include <Geom/Light.h>
#include <Geom/Color.h>
#include <Geometry/Point.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::PersistentSpace::Persistent;

class HeadLight : public Light {
    Color c;
public:
    HeadLight(const clString& name, const Color&);
    virtual ~HeadLight();
    virtual void compute_lighting(const View& view, const Point& at,
				  Color&, Vector&);
    virtual GeomObj* geom();
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx);
#endif
    virtual void lintens(const OcclusionData& od, const Point& hit_position,
			 Color& light, Vector& light_dir);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:48  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:11  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:08  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

#endif /* SCI_Geom_HeadLight_h */
