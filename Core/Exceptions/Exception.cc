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



/*
 *  Exception.h: Base exception class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Core/Exceptions/Exception.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#  include <unistd.h>
#else
#  define strcasecmp stricmp
#  include <io.h>
#  include <process.h>
#  include "StackWalker.h"
#endif

#include <sgi_stl_warnings_off.h>
#include   <iostream>
#include   <iomanip>
#include   <sstream>
#include <sgi_stl_warnings_on.h>

#ifdef HAVE_EXC
#  include <libexc.h>
#elif (defined(__GNUC__) && defined(__linux))
#  include <execinfo.h>
#  include <cxxabi.h>
#  include <dlfcn.h>
#elif defined(REDSTORM)
#  include <execinfo.h>
#endif

#if defined(_AIX)
// Needed for strcasecmp on aix 4.3 (on 5.1 we don't need this.)
// currently blue is 4.3.
#  include <strings.h>
#endif

namespace SCIRun {

using namespace std;

Exception::Exception()
{
  stacktrace_ = strdup(getStackTrace().c_str());
}

Exception::~Exception()
{
  if(stacktrace_) {
    free((char*)stacktrace_);
    stacktrace_ = 0;
  }
}

// This is just to fool the compiler so that it will not complain about
// "loop expressions are constant"  (SGI mipspro)  See Exception.h for
// use - Steve.
bool Exception::alwaysFalse()
{
  return false;
}


void Exception::sci_throw(const Exception& exc)
{
  // This is a function invoked by the SCI_THROW macro.  It
  // can be useful for tracking down fatal errors, since the
  // normal exception mechanism will unwind the stack before
  // you can get anything useful out of it.

  // Set this environment variable if you want to have a default
  // response to the question below.  Value values are:
  // ask, dbx, cvd, throw, abort
  char* emode = getenv("SCI_EXCEPTIONMODE");
  if(!emode)
    emode = "dbx"; // Default exceptionmode

  // If the mode is not "throw", we print out a message
  if(strcasecmp(emode, "throw") != 0){
    cerr << "\n\nAn exception was thrown.  Msg: " << exc.message() << "\n";
    if(exc.stacktrace_){
      cerr << exc.stacktrace_;
    }
    // Print out the exception type (clasname) and the message
    cerr << "\nException type: " << exc.type() << '\n';
    cerr << "Exception message: " << exc.message() << '\n';
  }
  // See what we should do
  for(;;){
    if(strcasecmp(emode, "ask") == 0){
      // Ask the user
      cerr << "\nthrow(t)/dbx(d)/cvd(c)/abort(a)? ";
      emode=0;
      while(!emode){
        char action;
        char buf[100];
        while(read(fileno(stdin), buf, 100) <= 0){
          if(errno != EINTR){
            cerr <<  "\nCould not read response, throwing exception\n";
            emode = "throw";
            break;
          }
        }
        action=buf[0];
        switch(action){
        case 't': case 'T':
          emode="throw";
          break;
        case 'd': case 'D':
          emode="dbx";
          break;
        case 'c': case 'C':
          emode="cvd";
          break;
        case 'a': case 'A':
          emode="abort";
          break;
        default:
          break;
        }
      }
    }

    if(strcasecmp(emode, "throw") == 0) {
      // We cannot throw from here, so we just return and the
      // exception will be thrown by the SCI_THROW macro
      return;
    } else if(strcasecmp(emode, "dbx") == 0){
#if defined( REDSTORM )
      cout << "Error: running debugger at exception is not supported on RedStorm\n";
#else
      // Fire up the debugger
      char command[100];
      if(getenv("SCI_DBXCOMMAND")){
        sprintf(command, getenv("SCI_DBXCOMMAND"), getpid());
      } else {
      cout << "Error: running debugger at exception is not supported on RedStorm\n";
#ifdef HAVE_EXC
        sprintf(command, "winterm -c dbx -p %d &", getpid());
#else
        sprintf(command, "xterm -e gdb %d&", getpid());
#endif
      }
      cerr << "Starting: " << command << '\n';
      system(command);
      emode="ask";
#endif
    } else if(strcasecmp(emode, "cvd") == 0){
#if defined( REDSTORM )
      cout << "Error: running debugger at exception is not supported on RedStorm\n";
#else
      // Fire up the slow, fancy debugger
      char command[100];
      sprintf(command, "cvd -pid %d &", getpid());
      cerr << "Starting: " << command << '\n';
      system(command);
      emode="ask";
#endif
    } else if(strcasecmp(emode, "abort") == 0){
      // This will trigger the thread library, but we cannot
      // directly call the thread library here or it would create
      // a circular dependency
      abort();
    } else {
      cerr << "Unknown exception mode: " << emode << ", aborting\n";
      abort();
    }
  }
}

string getStackTrace(void* context /*=0*/)
{
  ostringstream stacktrace;
#if defined(HAVE_EXC) || (defined(__GNUC__) && defined(__linux)) || defined(REDSTORM)
  static const int MAXSTACK = 100;
#endif

#ifdef HAVE_EXC
  // Use -lexc to print out a stack trace
  static const int MAXNAMELEN = 1000;
  __uint64_t addrs[MAXSTACK];
  char* cnames_str = new char[MAXSTACK*MAXNAMELEN];
  char* names[MAXSTACK];
  for(int i=0;i<MAXSTACK;i++)
    names[i]=cnames_str+i*MAXNAMELEN;
  int nframes = trace_back_stack(0, addrs, names, MAXSTACK, MAXNAMELEN);
  if(nframes == 0){
    stacktrace << "Backtrace not available!\n";
  } else {
    stacktrace << "Backtrace:\n";
    stacktrace.flags(ios::hex);
    // Skip the first procedure (us)
    for(int i=1;i<nframes;i++)
      stacktrace << "0x" << (void*)addrs[i] << ": " << names[i] << '\n';
  }
#elif defined(REDSTORM)

  // FYI, RedStorm doesn't seem to provide the function names as might be expected
  // when using backtrace_symbols.  So in the Uintah/tools/StackTrace/ directory
  // is code to get the function names.
  void * callstack[ MAXSTACK ];
  int    nframes = backtrace( callstack, MAXSTACK );

  if( nframes == 0 ){
    stacktrace << "Backtrace not available!\n";
  } else {
    char ** strs = backtrace_symbols( callstack, nframes );
    for( int pos = 0; pos < nframes; ++pos ) {
      stacktrace << callstack[ pos ] << "\n";
    }
    free(strs);
  }

#elif defined(__GNUC__) && defined(__linux)
  static void *addresses[MAXSTACK];
  int n = backtrace( addresses, MAXSTACK );
  if (n < 2){
    stacktrace << "Backtrace not available!\n";
  } else {
    stacktrace << "Backtrace:\n";
    stacktrace.flags(ios::hex);
    char **names = backtrace_symbols( addresses, n );
    for ( int i = 2; i < n; i++ ) {
     Dl_info info;
     char *demangled = NULL;

     //Attempt to demangle this if possible
     //Get the nearest symbol to feed to demangler
     if(dladdr(addresses[i], &info) != 0) {
      int stat;
      // __cxa_demangle is a naughty obscure backend and no
      // self-respecting person would ever call it directly. ;-)
      // However it is a convenient glibc way to demangle syms.
      demangled = abi::__cxa_demangle(info.dli_sname,0,0,&stat);
     }
     if (demangled != NULL) {
      //Chop off the garbage from the raw symbol
      char *loc = strchr(names[i], '(');
      if (loc != NULL) *loc = '\0';
      
      stacktrace << i - 1 << ". " << names[i] << '\n';
      stacktrace << "  in " << demangled << '\n';
      delete demangled;
     } else { // Just output the raw symbol
      stacktrace << i - 1 << ". " << names[i] << '\n';
     }
    }
    free(names);
  }
#elif defined(_WIN32)
  StackWalker sw;
  stacktrace << sw.GetCallstack(context);
#endif
  return stacktrace.str();
}

} // End namespace SCIRun
