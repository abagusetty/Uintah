
/*
 *  Thread: The thread class
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCICore_Thread_Thread_h
#define SCICore_Thread_Thread_h

#include <SCICore/share/share.h>

namespace SCICore {
    namespace Thread {
	struct Thread_private;
	class ParallelBase;
	class Runnable;
	class ThreadGroup;
	
/**************************************
 
CLASS
   Thread
   
KEYWORDS
   Thread
   
DESCRIPTION

   The Thread class provides a new context in which to run.  A single
   Runnable class is attached to a single Thread class, which are
   executed in another thread.
   
****************************************/
	class SCICORESHARE Thread {
	public:
	    //////////
	    // Possible thread start states
	    enum ActiveState {
		Activated,
		Stopped,
		NotActivated
	    };
	    
	    //////////
	    // Create a thread, which will execute the <b>run()</b>
	    // method in the <b>runner</b> object. The thread <b>name</b>
	    // is used for identification purposes, and does not need to
	    // be unique with respect to other threads.  <b>Group</b>
	    // specifies the ThreadGroup that to which this thread
	    // should belong.  If no group is specified (group==0),
	    // the default group is used.
	    Thread(Runnable* runner, const char* name,
		   ThreadGroup* group=0, ActiveState state=Activated);
	    
	    //////////
	    // Return the <b>ThreadGroup</b> associated with this thread.
	    ThreadGroup* getThreadGroup();
	    
	    //////////
	    // Return the <b>Runnable</b> associated with this thread.
	    Runnable* getRunnable();
	    
	    //////////
	    // Flag the thread as a daemon thread.  When all non-deamon
	    // threads exit, the program will exit.
	    void setDaemon(bool to=true);
	    
	    //////////
	    // Returns true if the thread is tagged as a daemon thread.
	    bool isDaemon() const;
	    
	    //////////
	    // If the thread is started in the the NotActivated
	    // state, use this to activate the thread.
	    void activate(bool stopped);
	    
	    //////////
	    // Arrange to have the thread deleted automatically at exit.
	    // The pointer to the thread should not be used by any other
	    // threads once this has been called.
	    void detach();
	    
	    //////////
	    // Returns true if the thread is detached
	    bool isDetached() const;
	    
	    //////////
	    // Set the stack size for a particular thread.  In order
	    // to use this thread, you must create the thread in the
	    // stopped state, set the stack size, and then start the
	    // thread.  Setting the stack size for a thread that is
	    // running or has ever been run, will throw an exception.
	    void setStackSize(unsigned long stackSize);
	    
	    //////////
	    // Returns the stack size for the thread
	    unsigned long getStackSize() const;
	    
	    //////////
	    // Kill all threads and exit with <b>code</b>.
	    static void exitAll(int code);
	    
	    //////////
	    // Exit the currently running thread
	    static void exit();
	    
	    //////////
	    // Returns a pointer to the currently running thread.
	    static Thread* self();
	    
	    //////////
	    // Stop the thread.
	    void stop();
	    
	    //////////
	    // Resume the thread
	    void resume();
	    
	    //////////
	    // Blocks the calling thread until this thead has finished
	    // executing. You cannot join detached threads or daemon threads.
	    void join();
	    
	    //////////
	    // Returns the name of the thread
	    const char* getThreadName() const;
	    
	    //////////
	    // Returns the number of processors on the system
	    static int numProcessors();
	    
	    //////////
	    // Request that the thread migrate to processor <i>proc</i>.
	    // If <i>proc</i> is -1, then the thread is free to run
	    // anywhere.
	    void migrate(int proc);
	    
	    //////////
	    // Start up several threads that will run in parallel.  A new
	    // <b>ThreadGroup</b> is created as a child of the optional parent.
	    // If <i>block</i> is true, theen the caller will block until all
	    // of the threads return.  Otherwise, the call will return
	    // immediately.
	    static ThreadGroup* parallel(const ParallelBase& helper,
					 int nthreads, bool block,
					 ThreadGroup* threadGroup=0);
	    
	    //////////
	    // Abort the current thread, or the process.  Prints a message on
	    // stderr, and the user may choose one of:
	    // <pre>continue(c)/dbx(d)/cvd(v)/kill thread(k)/exit(e)</pre>
	    static void niceAbort();
	    
	    //////////
	    // Mark a section as one that could block for debugging purposes.
	    // The <b>int</b> that is returned should be passed into
	    // <i>couldBlockDone(int)</i> when the section has completed.  This
	    // will typically not be used outside of the thread implementation
	    static int couldBlock(const char* why);
	    
	    //////////
	    // Mark the end of a selection that could block.
	    // <i>restore</i> was returned from a previous invocation
	    // of the above <b>couldBlock</b>.
	    static void couldBlockDone(int restore);
	    
	    //////////
	    // The calling process voluntarily gives up time to another process
	    static void yield();

	private:
	    friend class Barrier;
	    friend class Mutex;
	    friend class Runnable;	    
	    friend class Semaphore;
	    friend class Thread_private;
	    
	    Runnable* d_runner;
	    const char* d_threadname;
	    ThreadGroup* d_group;
	    unsigned long d_stacksize;
	    bool d_daemon;
	    bool d_detached;
	    bool d_activated;
	    void os_start(bool stopped);
	    static void initialize();
	    Thread(ThreadGroup* g, const char* name);

	public:
	    static void checkExit();
	    int d_cpu;
	    ~Thread();
	    Thread_private* d_priv;
	    void run_body();	    	    
	    enum ThreadState {
		STARTUP,
		RUNNING,
		IDLE,
		SHUTDOWN,
		DIED,
		PROGRAM_EXIT,
		JOINING,
		BLOCK_ANY,
		BLOCK_BARRIER,
		BLOCK_MUTEX,
		BLOCK_SEMAPHORE,
		BLOCK_CONDITIONVARIABLE
	    };
	private:
	    static const char* getStateString(ThreadState);
	    static int push_bstack(Thread_private*, ThreadState s,
				   const char* why);
	    static void pop_bstack(Thread_private*, int oldstate);
	    int get_thread_id();
	    static void print_threads();

	    // Cannot copy them
	    Thread(const Thread&);
	    Thread& operator=(const Thread&);
	};	
    }
}

#endif

//
// $Log$
// Revision 1.17  1999/11/10 19:41:16  moulding
// modified the comment for yield
//
// Revision 1.16  1999/11/03 01:23:03  moulding
// fixed a typo in the description (comment) for Thread::isDetached()
//
// Revision 1.15  1999/11/01 22:50:01  moulding
// fixed a typo in the (comment) description for Thread::couldBlock
//
// Revision 1.14  1999/10/15 20:56:51  ikits
// Fixed conflict w/ get_tid in /usr/include/task.h. Replaced by get_thread_id.
//
// Revision 1.13  1999/09/24 18:55:08  moulding
// added SCICORESHARE, for win32, to class declarations
//
// Revision 1.12  1999/09/22 19:10:28  sparker
// Implemented timedWait method for ConditionVariable.  A default
// implementation of CV is no longer possible, so the code is moved
// to Thread_irix.cc.  The timedWait method for irix uses uspollsema
// and select.
//
// Revision 1.11  1999/09/03 19:51:16  sparker
// Fixed bug where if Thread::parallel was called with block=false, the
//   helper object could get destroyed before it was used.
// Removed include of SCICore/Thread/ParallelBase and
//  SCICore/Thread/Runnable from Thread.h to minimize dependencies
// Fixed naming of parallel helper threads.
//
// Revision 1.10  1999/09/02 16:52:44  sparker
// Updates to cocoon documentation
//
// Revision 1.9  1999/08/31 08:59:05  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.8  1999/08/29 00:47:02  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.7  1999/08/28 03:46:51  sparker
// Final updates before integration with PSE
//
// Revision 1.6  1999/08/25 22:36:01  sparker
// More thread library updates - now compiles
//
// Revision 1.5  1999/08/25 19:00:51  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.4  1999/08/25 02:38:00  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

