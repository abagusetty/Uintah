/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 * Manual template instantiations
 */


/*
 * These aren't used by Datatypes directly, but since they are used in
 * a lot of different modules, we instantiate them here to avoid bloat
 *
 * Find the bloaters with:
find . -name "*.ii" -print | xargs cat | sort | uniq -c | sort -nr | more
 */

#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>



using namespace SCIRun;
#ifdef __sgi
#pragma set woff 1468
#endif

#include <Core/Geometry/Tensor.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/MaskedTetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/MaskedLatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Datatypes/PropertyManager.h>



template class LockingHandle<ColumnMatrix>;
template class LockingHandle<Matrix>;

template class MaskedTetVol<Tensor>;
template class MaskedTetVol<Vector>;
template class MaskedTetVol<double>;
template class MaskedTetVol<int>;
template class MaskedTetVol<short>;
template class MaskedTetVol<char>;
template class MaskedTetVol<bool>;

template class TetVol<Tensor>;
template class TetVol<Vector>;
template class TetVol<double>;
template class TetVol<int>;
template class TetVol<short>;
template class TetVol<char>;
template class TetVol<bool>;
template class GenericField<TetVolMesh, vector<Tensor> >;
template class GenericField<TetVolMesh, vector<Vector> >;
template class GenericField<TetVolMesh, vector<double> >;
template class GenericField<TetVolMesh, vector<int> >;
template class GenericField<TetVolMesh, vector<short> >;
template class GenericField<TetVolMesh, vector<char> >;
template class GenericField<TetVolMesh, vector<bool> >;

template class MaskedLatticeVol<Tensor>;
template class MaskedLatticeVol<Vector>;
template class MaskedLatticeVol<double>;
template class MaskedLatticeVol<int>;
template class MaskedLatticeVol<short>;
template class MaskedLatticeVol<char>;
template class MaskedLatticeVol<bool>;

template class LatticeVol<Tensor>;
template class LatticeVol<Vector>;
template class LatticeVol<double>;
template class LatticeVol<int>;
template class LatticeVol<short>;
template class LatticeVol<char>;
template class LatticeVol<bool>;
template class GenericField<LatVolMesh, FData3d<Tensor> >;
template class GenericField<LatVolMesh, FData3d<Vector> >;
template class GenericField<LatVolMesh, FData3d<double> >;
template class GenericField<LatVolMesh, FData3d<int> >;
template class GenericField<LatVolMesh, FData3d<short> >;
template class GenericField<LatVolMesh, FData3d<char> >;
template class GenericField<LatVolMesh, FData3d<bool> >;



template class TriSurf<Tensor>;
template class TriSurf<Vector>;
template class TriSurf<double>;
template class TriSurf<int>;
template class TriSurf<short>;
template class TriSurf<char>;
template class TriSurf<bool>;
template class GenericField<TriSurfMesh, vector<Tensor> >;
template class GenericField<TriSurfMesh, vector<Vector> >;
template class GenericField<TriSurfMesh, vector<double> >;
template class GenericField<TriSurfMesh, vector<int> >;
template class GenericField<TriSurfMesh, vector<short> >;
template class GenericField<TriSurfMesh, vector<char> >;
template class GenericField<TriSurfMesh, vector<bool> >;

template class ContourField<double>;
template class GenericField<ContourMesh, vector<double> >;

template class PointCloud<Tensor>;
template class PointCloud<Vector>;
template class PointCloud<double>;
template class PointCloud<int>;
template class PointCloud<short>;
template class PointCloud<char>;
template class PointCloud<bool>;
template class GenericField<PointCloudMesh, vector<Tensor> >;
template class GenericField<PointCloudMesh, vector<Vector> >;
template class GenericField<PointCloudMesh, vector<double> >;
template class GenericField<PointCloudMesh, vector<int> >;
template class GenericField<PointCloudMesh, vector<short> >;
template class GenericField<PointCloudMesh, vector<char> >;
template class GenericField<PointCloudMesh, vector<bool> >;

template class Property<string>;
template class Property<Array1<double> >;
template class Property<Array1<Tensor> >;
template class Property<pair<int,double> >;
template class Property<pair<double,double> >;
template class Property<pair<float,float> >;
template class Property<pair<unsigned int,unsigned int> >;
template class Property<pair<int,int> >;
template class Property<pair<unsigned short,unsigned short> >;
template class Property<pair<short,short> >;
template class Property<pair<unsigned char,unsigned char> >;
template class Property<pair<char,char> >;

//! Compute the gradient g in cell ci.
template <>
Vector TetVol<Vector>::cell_gradient(TetVolMesh::cell_index /*ci*/)
{
  ASSERT(type_name(1) != "Vector");  // redundant, useful error message
  return Vector(0, 0, 0);
}


template <>
Vector TetVol<Tensor>::cell_gradient(TetVolMesh::cell_index /*ci*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return Vector(0, 0, 0);
}

template <> bool LatticeVol<Tensor>::get_gradient(Vector &, Point &/*p*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return false;
}

template <> bool LatticeVol<Vector>::get_gradient(Vector &, Point &/*p*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return false;
}


#ifdef __sgi
#pragma reset woff 1468
#endif










