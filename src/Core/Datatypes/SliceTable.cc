#include <Core/Geometry/Ray.h>
#include <Core/Util/NotFinished.h>
#include <Core/Datatypes/VolumeUtils.h>
#include <iostream>
#include <string>
#include <Core/Datatypes/SliceTable.h>
using std::cerr;
using std::endl;

using std::string;

namespace SCIRun {



SliceTable::SliceTable( const Point& min,  const Point& max,  const Ray& view,
	      int slices, int treedepth):
  view(view), slices(slices), treedepth(treedepth)
{
  Point corner[8];
  corner[0] = min;
  corner[1] = Point(min.x(), min.y(), max.z());
  corner[2] = Point(min.x(), max.y(), min.z());
  corner[3] = Point(min.x(), max.y(), max.z());
  corner[4] = Point(max.x(), min.y(), min.z());
  corner[5] = Point(max.x(), min.y(), max.z());
  corner[6] = Point(max.x(), max.y(), min.z());
  corner[7] = max;

  Vector v = max - min;
  v.normalize();
  DT = intersectParam(-v, corner[7], Ray(corner[0], v) );
  DT /= slices;


  int i,j,k;
  double t[8];
  for( i = 0; i < 8; i++){
    order[i] = i;
    t[i] = intersectParam(-view.direction(),
			   corner[i], view);
  }
  double tmp;
  // Sort the slices in order back to front
  for(j = 0; j < 8; j++){
    for(i = j+1; i < 8; i++){
      if( t[j] < t[i] ){
	tmp = t[i];  k = order[i];
	t[i] = t[j]; order[i] = order[j];
	t[j] = tmp;  order[j] = k;
      }
    }
  }
  minT = t[7];
  maxT = t[0];
  
}

SliceTable::~SliceTable()
{}

void SliceTable::getParameters(const Brick& brick, double& tmin,
			       double& tmax, double& dt) const
{
  const double min = intersectParam(-view.direction(),
				    brick[ order[7] ],
				    view);
  const double max = intersectParam(-view.direction(),
				    brick[ order[0] ],
				    view);

  dt = DT * pow(2.0, treedepth  - brick.level());

  const double max_steps = floor((max - minT)/dt);
  tmax = minT + max_steps * dt;
  const double min_steps = floor((min - minT)/dt);
  tmin = minT + (min_steps + 1.0) * dt;
}

#define SLICETABLE_VERSION 1
  
} // End namespace SCIRun
