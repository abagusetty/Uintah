#ifndef SCI_project_CCVectorField_h
#define SCI_project_CCVectorField_h 1

//#include "UintahVectorField.h"
#include <SCICore/Datatypes/VectorField.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <SCICore/Geometry/Point.h>
#include <vector>

namespace SCICore {
  namespace Datatypes {

using namespace SCICore::Geometry;
using namespace Uintah;
using std::vector;


class SCICORESHARE CCVectorField : public VectorField {
public:
  CCVectorField();
  CCVectorField(const CCVectorField&);
  CCVectorField(GridP grid, LevelP level, string var, int mat,
		const vector< CCVariable<Vector> >& vars);
  virtual ~CCVectorField() {}
  virtual VectorField* clone();

  virtual void compute_bounds();
  virtual void get_boundary_lines(Array1<Point>& lines);
  virtual int interpolate(const Point&, Vector&);
  virtual int interpolate(const Point&, Vector&,
			  int& ix, int exhaustive=0);

  void SetGrid( GridP g ){ grid = g; }
  void SetLevel( LevelP l){ level = l; }
  void SetName( string vname ) { _varname = vname; }
  void SetMaterial( int index) { _matIndex = index; }
  void AddVar( const CCVariable<Vector>& var);
//   int nPatches(){ return _vars.size();}
//   string& name(){ return _varname; }
//   int material(){ return _matIndex; }
//   const CCVariable<Vector>& var(int i){ return _vars[i];}
private:
  vector< CCVariable<Vector> > _vars;
  GridP grid;
  LevelP level;
  string _varname;
  int _matIndex;
};

} // End namespace Datatypes
} // End namespace SCICore

#endif
