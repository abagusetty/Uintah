
#ifndef SCI_THREAD_POOLMUTEX_H
#define SCI_THREAD_POOLMUTEX_H 1

/**************************************
 
CLASS
   PoolMutex
   
KEYWORDS
   PoolMutex
   
DESCRIPTION
   Provides a simple mutual exclusion primitive.  This
   differs from the <b>Mutex</b> class in that the mutexes are allocated
   from a pool of mutexes.  A single mutex may get assigned to
   more than one object.  This will still provide the atomicity
   guaranteed by a mutex, but requires significantly less memory.  Since
   the mutex may be associated with many other objects, <b>PoolMutex</b>
   should only be used in low-use, short duration scenarios.  As with
   <b>Mutex</b>, <b>lock()</b> and <b>unlock()</b> will lock and unlock
   the mutex, and <b>PoolMutex</b> is not a recursive Mutex (See
   <b>RecursiveMutex</b>), and calling lock() in a nested call will
   result in an error or deadlock.
PATTERNS


WARNING
   
****************************************/

class Mutex;

class PoolMutex {
    unsigned short d_mutexIndex;

public:
    //////////
    // Create the mutex.  The mutex is allocated in the unlocked state.
    PoolMutex();
    
    //////////
    // Destroy the mutex.  Destroying the mutex in the locked state has
    // undefined results.
    ~PoolMutex();

    //////////
    // Acquire the Mutex.  This method will block until the mutex is
    // acquired.
    void lock();

    //////////
    // Release the Mutex, unblocking any other threads that are blocked
    // waiting for the Mutex.
    void unlock();

    //////////
    // Attempt to acquire the Mutex without blocking.  Returns true if the
    // mutex was available and actually acquired.
    bool tryLock();
};

#endif


