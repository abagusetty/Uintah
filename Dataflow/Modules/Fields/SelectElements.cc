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

//    File   : SelectElements.cc
//    Author : David Weinstein
//    Date   : August 2001

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE SelectElements : public Module {
public:
  GuiString value_;
  SelectElements(const string& id);
  virtual ~SelectElements();
  virtual void execute();
};

extern "C" PSECORESHARE Module* make_SelectElements(const string& id) {
  return scinew SelectElements(id);
}

SelectElements::SelectElements(const string& id)
  : Module("SelectElements", id, Source, "Fields", "SCIRun"),
    value_("value", id, this)
{
}

SelectElements::~SelectElements(){
}

void SelectElements::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *ifieldPort = (FieldIPort*)get_iport("TetVol");

  if (!ifieldPort) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  FieldHandle field;
  if (!ifieldPort->get(field) || !field.get_rep()) return;

  ScalarFieldInterface *sfi = field->query_scalar_interface();
  double min, max;
  sfi->compute_min_max(min, max);
  string value = value_.get();
  char **v;
  char *value_str = new char[30];
  v = &value_str;
  strcpy(value_str, value.c_str());
  char *matl;
  Array1<int> values;
  while (matl = strtok(value_str, " ,")) {
    value_str=0;
    values.add(atoi(matl));
  }
  delete[] (*v);

  int ii;
  for (ii=0; ii<values.size(); ii++) {
    if (values[ii] < min) {
      cerr << "Error - min="<<min<<" value="<<values[ii]<<"\n";
      values[ii]=(int)min;
    } else if (values[ii] > max) {
      cerr << "Error - max="<<max<<" value="<<values[ii]<<"\n";
      values[ii]=(int)max;
    }
  }

  TetVolField<int> *tvI = dynamic_cast<TetVolField<int> *>(field.get_rep());
  if (!tvI) {
    cerr << "Error -- input field wasn't a TetVolField<int>\n";
    return;
  }

  FieldOPort *ofieldPort = (FieldOPort*)get_oport("TetVol");
  if (!ofieldPort) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  MatrixOPort *omat1Port = (MatrixOPort*)get_oport("LeadFieldRestrictColumns");
  if (!omat1Port) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  MatrixOPort *omat2Port = (MatrixOPort*)get_oport("LeadFieldInflateRows");
  if (!omat2Port) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  TetVolMeshHandle tvm = tvI->get_typed_mesh();
  vector<pair<string, Tensor> > conds;
  tvI->get_property("conductivity_table", conds);
  vector<int> fdata;
  TetVolMesh::Cell::size_type ntets;
  tvm->size(ntets);
  Array1<int> tet_valid(ntets);
  tet_valid.initialize(0);
  TetVolMesh::Cell::iterator citer; tvm->begin(citer);
  TetVolMesh::Cell::iterator citere; tvm->end(citere);
  int count=0;
  Array1<int> indices;
  while (citer != citere) {
    TetVolMesh::Cell::index_type ci = *citer;
    ++citer;
    for (ii=0; ii<values.size(); ii++) {
      if (tvI->fdata()[ci] == values[ii]) {
	tet_valid[count]=1;
	fdata.push_back(values[ii]);
	indices.add(count);
      }
    }
    count++;
  }
  cerr << "Found "<<fdata.size()<<" elements (out of "<<count<<") with specified conductivity indices.\n";

  ColumnMatrix *cm = scinew ColumnMatrix(indices.size()*3);
  for (ii=0; ii<indices.size(); ii++) {
    (*cm)[ii*3]=indices[ii]*3;
    (*cm)[ii*3+1]=indices[ii]*3+1;
    (*cm)[ii*3+2]=indices[ii]*3+2;
  }
  MatrixHandle cmH(cm);
  omat1Port->send(cmH);

  int k=0;
  ColumnMatrix *cm2 = scinew ColumnMatrix(count*3);
  for (ii=0; ii<count; ii++) {
    if (tet_valid[ii]) {
      (*cm2)[ii*3]=k;
      (*cm2)[ii*3+1]=k+1;
      (*cm2)[ii*3+2]=k+2;
      k+=3;
    } else {
      (*cm2)[ii*3]=-1;
      (*cm2)[ii*3+1]=-1;
      (*cm2)[ii*3+2]=-1;
    }
  }
  MatrixHandle cmH2(cm2);
  omat2Port->send(cmH2);

  TetVolMesh *mesh = scinew TetVolMesh;

  TetVolMesh::Node::iterator niter; tvm->begin(niter);
  TetVolMesh::Node::iterator niter_end; tvm->end(niter_end);
  while (niter != niter_end) {
    TetVolMesh::Node::index_type ni = *niter;
    ++niter;
    Point p;
    tvm->get_center(p, ni);
    mesh->add_point(p);
  }
  tvm->begin(citer); tvm->end(citere);
  count=0;
  while(citer != citere) {
    TetVolMesh::Cell::index_type ci = *citer;
    if (tet_valid[count]) {
      TetVolMesh::Node::array_type arr(4);
      tvm->get_nodes(arr, ci);
      mesh->add_elem(arr);
    }
    ++citer; 
    ++count;
  }

  TetVolMesh *mesh_no_unattached_nodes = scinew TetVolMesh;
  TetVolMesh::Node::size_type nnodes;
  mesh->size(nnodes);
  Array1<TetVolMesh::Node::index_type> node_map(nnodes);
  node_map.initialize(-1);
  mesh->flush_changes();
  mesh->begin(niter);
  mesh->end(niter_end);
  TetVolMesh::Node::array_type narr;
  int total=0;
  int added=0;
//  FILE *fout = fopen("/tmp/map-entire-volume-nodes-to-heart-volume-nodes.txt", "wt");
  while(niter != niter_end) {
    mesh->get_neighbors(narr, *niter);
    if (narr.size()) {
      Point p;
      mesh->get_center(p, *niter);
      node_map[total]=added;
      added++;
      mesh_no_unattached_nodes->add_point(p);
//      fprintf(fout, "%d\n", total);
    }
    ++niter;
    total++;
  }
//  fclose(fout);
  mesh->begin(citer);
  mesh->end(citere);
  while(citer != citere) {
    TetVolMesh::Cell::index_type ci = *citer;
    TetVolMesh::Node::array_type arr(4);
    mesh->get_nodes(arr, ci);
    if (arr[0] == -1 ||
	arr[1] == -1 ||
	arr[2] == -1 ||
	arr[3] == -1) {
      cerr << "ERROR - TET CONTAINS UNMAPPED NODE!\n";
      exit(0);
    }
    arr[0] = node_map[(int)arr[0]];
    arr[1] = node_map[(int)arr[1]];
    arr[2] = node_map[(int)arr[2]];
    arr[3] = node_map[(int)arr[3]];
    mesh_no_unattached_nodes->add_elem(arr);
    ++citer;
  }    
  

//  TetVolField<int> *tv = scinew TetVolField<int>(mesh_no_unattached_nodes, Field::CELL);
//  tv->fdata() = fdata;
//  tv->set_property("conductivity_table", conds, false);
  TetVolField<double> *tv = scinew TetVolField<double>(mesh_no_unattached_nodes, Field::NODE);

  FieldHandle tvH(tv);
  
  ofieldPort->send(tvH);
}    
} // end SCIRun
