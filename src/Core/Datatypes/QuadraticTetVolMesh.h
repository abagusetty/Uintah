//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : QuadraticTetVolMesh.h
//    Author : Martin Cole
//    Date   : Sun Feb 24 14:25:39 2002

#ifndef Datatypes_QuadraticTetVolMesh_h
#define Datatypes_QuadraticTetVolMesh_h

#include <Core/Datatypes/TetVolMesh.h>

namespace SCIRun {

class SCICORESHARE QuadraticTetVolMesh : public TetVolMesh
{
public:
  QuadraticTetVolMesh();
  QuadraticTetVolMesh(const TetVolMesh &tv);
  QuadraticTetVolMesh(const QuadraticTetVolMesh &copy);

  virtual QuadraticTetVolMesh *clone() 
  { return new QuadraticTetVolMesh(*this); }
  virtual ~QuadraticTetVolMesh();

  void begin(Node::iterator &) const;
  void begin(Edge::iterator &) const;
  void begin(Face::iterator &) const;
  void begin(Cell::iterator &) const;

  void end(Node::iterator &) const;
  void end(Edge::iterator &) const;
  void end(Face::iterator &) const;
  void end(Cell::iterator &) const;

  void size(Node::size_type &) const;
  void size(Edge::size_type &) const;
  void size(Face::size_type &) const;
  void size(Cell::size_type &) const;

  void get_nodes(Node::array_type &array, Cell::index_type idx) const;
  void get_point(Point &result, Node::index_type index) const;

  void get_weights(const Point& p, Node::array_type &l, 
		   vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &) 
  { ASSERTFAIL("QuadraticTetVolMesh::get_weights for edges isn't supported"); }
  void get_weights(const Point &, Face::array_type &, vector<double> &) 
  { ASSERTFAIL("QuadraticTetVolMesh::get_weights for faces isn't supported"); }
  void get_weights(const Point &p, Cell::array_type &l, vector<double> &w)
  { TetVolMesh::get_weights(p, l, w); }
  //! get gradient relative to point p
  void get_gradient_basis(Cell::index_type ci, const Point& p,
			    Vector& g0, Vector& g1, Vector& g2, Vector& g3, 
			    Vector& g4, Vector& g5, Vector& g6, Vector& g7, 
			    Vector& g8, Vector& g9) const;

  //! gradient for gauss pts 
  double get_gradient_basis(Cell::index_type ci, int gaussPt, const Point&, 
			    Vector& g0, Vector& g1, Vector& g2, Vector& g3, 
			    Vector& g4, Vector& g5, Vector& g6, Vector& g7, 
			    Vector& g8, Vector& g9) const;
    private:
  const Point& ave_point(const Point &p0, const Point &p1) const;
  double calc_jac_derivs(Vector &dxi, Vector &dnu, Vector &dgam, 
			 double xi, double nu, double gam, 
			 Cell::index_type ci) const;
  double calc_dphi_dgam(int ptNum, double xi, double nu, 
			double gam) const;
  double calc_dphi_dnu(int ptNum, double xi, double nu, 
		       double gam) const;
  double calc_dphi_dxi(int ptNum, double xi, double nu, 
		       double gam) const;
};

// Handle type for TetVolMesh mesh.
typedef LockingHandle<QuadraticTetVolMesh> QuadraticTetVolMeshHandle;

} // namespace SCIRun


#endif // Datatypes_QuadraticTetVolMesh_h
