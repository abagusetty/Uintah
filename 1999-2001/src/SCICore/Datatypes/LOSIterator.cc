#include <SCICore/Datatypes/LOSIterator.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Datatypes/Brick.h>


namespace SCICore {
namespace Datatypes {

using SCICore::Geometry::BBox;
using SCICore::Geometry::Vector;



LOSIterator::LOSIterator(const GLTexture3D* tex, Ray view,
			 Point control):
    GLTextureIterator( tex, view, control )
{
  Vector v = control - view.origin();
  Point p;
  const Octree< Brick* >* node = (tex->bontree);
  if ( tex->depth() == 0 ){
    next = (*node)();
  } else {
    int child;
    BBox box((*node)()->bbox());
    box.PrepareIntersect( view.origin() );
    if( !box.Intersect( view.origin(), v, p )){
      next = (*node)();
      return;
    }
    do {
      order.push_back( traversal( node ));
      path.push_back( node );
      child = order.back()->front();
      order.back()->pop_front();
      node = (*node)[child];
      BBox b((*node)()->bbox());
      if( !b.Intersect( view.origin(), v, p )){
	break;
      }
    } while (node->type() == Octree<Brick* >::PARENT);
    
    next = (*node)();
  }

}


Brick*
LOSIterator::Start()
{
  return next;
}

Brick*
LOSIterator::Next()
{
  // Get the last iterator
    if( !done )
      SetNext();
    return next;
}

bool
LOSIterator::isDone()
{
  return done;
}

void 
LOSIterator::SetNext()
{ 
  Vector v = control - view.origin();
  Point p;

  while( path.size() != 0 && 
	 !order.back()->size() ){
    path.pop_back();
    order.pop_back();
  }

  if( path.size() == 0 ){
    done = true;
    return;
  }

  int child;
  const Octree< Brick* >* node = path.back();
  child = order.back()->front();
  order.back()->pop_front();
  node = (*node)[child];
  
  while( node->type() == Octree<Brick* >::PARENT ){
   BBox b( (*node)()->bbox() );
   b.PrepareIntersect( view.origin() );

    if(!b.Intersect( view.origin(), v, p )){
      break;
    }
    order.push_back( traversal( node ));
    path.push_back( node );
    child = order.back()->front();
    order.back()->pop_front();
    node = (*node)[child];
  }
  next = (*node)();
}

} //Datatypes
} //SCICore

