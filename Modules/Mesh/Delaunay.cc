/*
 *  Delaunay.cc:  Delaunay Triangulation in 3D
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/MeshPort.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>

class Delaunay : public Module {
    MeshIPort* iport;
    MeshOPort* oport;
public:
    Delaunay(const clString& id);
    Delaunay(const Delaunay&, int deep);
    virtual ~Delaunay();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_Delaunay(const clString& id)
{
    return new Delaunay(id);
}

static RegisterModule db1("Mesh", "Delaunay", make_Delaunay);

Delaunay::Delaunay(const clString& id)
: Module("Delaunay", id, Filter)
{
    iport=new MeshIPort(this, "Input Mesh", MeshIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=new MeshOPort(this, "Delaunay Mesh", MeshIPort::Atomic);
    add_oport(oport);
}

Delaunay::Delaunay(const Delaunay& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("Delaunay::Delaunay");
}

Delaunay::~Delaunay()
{
}

Module* Delaunay::clone(int deep)
{
    return new Delaunay(*this, deep);
}

void Delaunay::execute()
{
    MeshHandle mesh_handle;
    if(!iport->get(mesh_handle))
	return;

    // Get our own copy of the mesh...
    mesh_handle.detach();
    Mesh* mesh=mesh_handle.get_rep();
    mesh->elems.remove_all();

    int nnodes=mesh->nodes.size();
    BBox bbox;
    for(int i=0;i<nnodes;i++)
	bbox.extend(mesh->nodes[i]->p);

    double epsilon=1.e-4;

    // Extend by max-(eps, eps, eps) and min+(eps, eps, eps) to
    // avoid thin/degenerate bounds
    bbox.extend(bbox.max()-Vector(epsilon, epsilon, epsilon));
    bbox.extend(bbox.min()+Vector(epsilon, epsilon, epsilon));

    // Make the bbox square...
    Point center(bbox.center());
    double le=bbox.longest_edge();
    Vector diag(le, le, le);
    Point bmin(center-diag/2.);
    Point bmax(center+diag/2.);

    // Make the initial mesh with a tetra which encloses the bounding
    // box.  The first point is at the minimum point.  The other 3
    // have one of the coordinates at bmin+diagonal*3.
    mesh->nodes.add(new Node(bmin));
    mesh->nodes.add(new Node(bmin+Vector(le*3, 0, 0)));
    mesh->nodes.add(new Node(bmin+Vector(0, le*3, 0)));
    mesh->nodes.add(new Node(bmin+Vector(0, 0, le*3)));

    Element* el=new Element(mesh, nnodes+0, nnodes+1, nnodes+2, nnodes+3);
    el->orient();
    el->faces[0]=el->faces[1]=el->faces[2]=el->faces[3]=-1;
    mesh->elems.add(el);

    for(int node=0;node<nnodes;node++){
	// Add this node...
	cerr << "Adding node: " << node << " of " << nnodes << endl;
	Point p(mesh->nodes[node]->p);

	// Find which element this node is in
	int in_element;
	if(!mesh->locate(p, in_element)){
	    error("Mesher upset - point outside of domain...");
	    return;
	}

	Array1<int> to_remove;
	to_remove.add(in_element);
	Array1<int> done;
	done.add(in_element);
	HashTable<Face, int> face_table;

	// Find it's neighbors...
	// We might be able to fix this loop to make it
	// O(N) instead of O(n^2) - use a Queue
	int i=0;
	while(i<to_remove.size()){
	    // See if the neighbor should also be removed...
	    Element* e=mesh->elems[to_remove[i]];
	    // Add these faces to the list of exposed faces...
	    Face f1(e->n[0], e->n[1], e->n[2]);
	    Face f2(e->n[0], e->n[1], e->n[3]);
	    Face f3(e->n[0], e->n[2], e->n[3]);
	    Face f4(e->n[1], e->n[2], e->n[3]);

	    // If the face is in the list, remove it.
	    // Otherwise, add it.
	    int dummy;
	    if(face_table.lookup(f1, dummy))
		face_table.remove(f1);
	    else
		face_table.insert(f1, dummy);

	    if(face_table.lookup(f2, dummy))
		face_table.remove(f2);
	    else
		face_table.insert(f2, dummy);

	    if(face_table.lookup(f3, dummy))
		face_table.remove(f3);
	    else
		face_table.insert(f3, dummy);

	    if(face_table.lookup(f4, dummy))
		face_table.remove(f4);
	    else
		face_table.insert(f4, dummy);

	    for(int j=0;j<4;j++){
		int skip=0;
		int neighbor=e->faces[j];
		for(int ii=0;ii<done.size();ii++){
		    if(neighbor==done[ii]){
			skip=1;
			cerr << "Breaking..." << endl;
			break;
		    }
		}
		if(neighbor==-1 || neighbor==-2)
		    skip=1;
		if(!skip){
		    // Process this neighbor
		    if(!skip){
			// See if this simplex is deleted by this point
			Point cen;
			double rad2;
			e->get_sphere2(cen, rad2);
			double ndist2=(p-cen).length2();
			if(ndist2 < rad2){
			    // This one must go...
			    to_remove.add(neighbor);
			}
		    }
		    done.add(neighbor);
		}
	    }
	    i++;
	}
	// Remove the to_remove elements...
	for(i=0;i<to_remove.size();i++){
	    int idx=to_remove[i];
	    delete mesh->elems[idx];
	    mesh->elems[idx]=0;
	}

	// Add the new elements from the faces...
	HashTableIter<Face, int> fiter(&face_table);
	for(fiter.first();fiter.ok();++fiter){
	    Face f(fiter.get_key());
	    Element* ne=new Element(mesh, node, f.n[0], f.n[1], f.n[2]);
	    ne->orient();
	    mesh->elems.add(ne);
	}
	mesh->compute_neighbors();
    }
}
