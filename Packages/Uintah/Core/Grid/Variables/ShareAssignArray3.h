#ifndef UINTAH_HOMEBREW_SHAREASSIGNARRAY3_H
#define UINTAH_HOMEBREW_SHAREASSIGNARRAY3_H

#include <Packages/Uintah/Core/Grid/Variables/Array3.h>

#ifndef _WIN32
#include <unistd.h>
#endif
#include <cerrno>

namespace Uintah {

class TypeDescription;

/**************************************

CLASS
ShareAssignArray3
  
Short description...

GENERAL INFORMATION

ShareAssignArray3.h

Version of the Array3 class, but with allowed assignment that shares
the data rather than copies it

Wayne Witzel
Department of Computer Science
University of Utah

Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

Copyright (C) 2001 SCI Group

KEYWORDS
Array3, Assignment

DESCRIPTION
Long description...
  
WARNING
  
****************************************/

template<class T>
class ShareAssignArray3 : public Array3<T> {
public:
  ShareAssignArray3()
    : Array3<T>() {}
  ShareAssignArray3(const Array3<T>& pv)
    : Array3<T>(pv) {}

  virtual ~ShareAssignArray3() {}
  
  ShareAssignArray3<T>& operator=(const Array3<T>& pv)
  { copyPointer(pv); return *this; }

  ShareAssignArray3<T>& operator=(const ShareAssignArray3<T>& pv)
  { copyPointer(pv); return *this; }
};

} // End namespace Uintah

#endif
