/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

//
// SlaveDatabase.cc
//

#include "SlaveDatabase.h"

#include <stdio.h>

namespace dav {

SlaveInfo::SlaveInfo( string machineId, Mc2Sc mc2Sc, int maxProcs ) : 
  d_machineId( machineId ),
  d_mc2Sc( mc2Sc ),
  d_maxProcs( maxProcs )
{
}

SlaveDatabase::SlaveDatabase() : d_numSlaves( 0 )
{
}

SlaveDatabase::~SlaveDatabase()
{
}

int
SlaveDatabase::numActive()
{
  return d_numSlaves;
}

SlaveInfo *
SlaveDatabase::find( string machineId )
{
  int size = d_slaves.size();
  for( int pos = 0; pos < size; pos++ )
    {
      if( d_slaves[ pos ]->d_machineId == machineId )
	{
	  return d_slaves[ pos ];
	}
    }
  return NULL;
}

SlaveInfo *
SlaveDatabase::leastLoaded()
{
  // Eventually determine least loaded SC and return that one.

  if( d_slaves.size() != 0 )
    return d_slaves[ 0 ];
  else
    return NULL;
}

array1<SlaveInfo*>::iterator
SlaveDatabase::findPosition( string machineId )
{
  array1<SlaveInfo*>::iterator iter = d_slaves.begin();
 
  for( ; iter != d_slaves.end(); iter++ )
    {
      if( (*iter)->d_machineId == machineId )
	{
	  return iter;
	}
    }
  throw InternalError( "This version of find should only be called when "
		       "you know that the item is in the list.  Thus "
		       "this exception should never be raised.",
                       __FILE__, __LINE__ );
}

void
SlaveDatabase::add( SlaveInfo * si ) throw (InternalError)
{
  if( find( si->d_machineId ) )
    throw InternalError( const_cast<char*>
        ((si->d_machineId + " already in SlaveDatabase ").c_str()),
        __FILE__, __LINE__ );
  d_slaves.push_back( si );
  d_numSlaves++;
}

void
SlaveDatabase::remove( SlaveInfo * si )
{
  if( !find( si->d_machineId ) )
    throw InternalError( const_cast<char*> ((si->d_machineId 
        + " not in SlaveDatabase ").c_str()), __FILE__, __LINE__ );
  d_slaves.erase( findPosition( si->d_machineId ) );
  d_numSlaves--;
}

void
SlaveDatabase::getMachineIds( array1<string> & ids )
{
  array1<SlaveInfo*>::iterator iter = d_slaves.begin();
  for( ; iter != d_slaves.end(); iter++ )
    {
      ids.push_back( (*iter)->d_machineId );
    }
}

void
SlaveDatabase::updateGuis()
{
  int size = d_slaves.size();

  if( size == 0 )
    return;

  printf( "SlaveDatabase: Running through all SCs"
	  " and telling them to send gui info:\n" );

  for( int pos = size-1; pos >= 0; pos-- )
    {
      printf("SlaveDatabase: Talking to SC %s\n",
	                       d_slaves[pos]->d_machineId.c_str());
      d_slaves[ pos ]->d_mc2Sc->updateNewGui();
    }
}

void
SlaveDatabase::shutDown()
{
  int size = d_slaves.size();

  if( size == 0 )
    return;

  printf( "SlaveDatabase: Running through all SCs"
	  " and shutting them down:\n" );

  while( d_slaves.size() != 0 )
    {
      SlaveInfo * si = d_slaves.back();
      d_slaves.pop_back();
      printf("   Shutting down %s\n", si->d_machineId.c_str());
      si->d_mc2Sc->shutDown();
    }
}

} // end namespace dav


