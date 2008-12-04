// 
// File:          pde_FEM.hxx
// Symbol:        pde.FEM-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Client-side glue code for pde.FEM
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_pde_FEM_hxx
#define included_pde_FEM_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace pde { 

  class FEM;
} // end namespace pde

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::pde::FEM >;
}
// 
// Forward declarations for method dependencies.
// 
namespace gov { 
  namespace cca { 

    class CCAException;
  } // end namespace cca
} // end namespace gov

namespace gov { 
  namespace cca { 

    class Services;
  } // end namespace cca
} // end namespace gov

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_pde_FEM_IOR_h
#include "pde_FEM_IOR.h"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_pdeports_FEMmatrixPort_hxx
#include "pdeports_FEMmatrixPort.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
namespace sidl {
  namespace rmi {
    class Call;
    class Return;
    class Ticket;
  }
  namespace rmi {
    class InstanceHandle;
  }
}
namespace pde { 

  /**
   * Symbol "pde.FEM" (version 0.1)
   */
  class FEM: public virtual ::gov::cca::Component, public virtual 
    ::pdeports::FEMmatrixPort, public virtual ::sidl::BaseClass {

    //////////////////////////////////////////////////
    // 
    // Special methods for throwing exceptions
    // 

  private:
    static 
    void
    throwException1(
      const char* methodName,
      struct sidl_BaseInterface__object *_exception
    )
      // throws:
      //    ::gov::cca::CCAException
      //    ::sidl::RuntimeException
    ;
    static 
    void
    throwException0(
      const char* methodName,
      struct sidl_BaseInterface__object *_exception
    )
      // throws:
    ;

    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:
    /**
     * user defined non-static method
     */
    int32_t
    makeFEMmatrices (
      /* in array<int> */const ::sidl::array<int32_t>& mesh,
      /* in array<double> */const ::sidl::array<double>& nodes,
      /* in array<int> */const ::sidl::array<int32_t>& dirichletNodes,
      /* in array<double> */const ::sidl::array<double>& dirichletValues,
      /* out array<double,2> */::sidl::array<double>& Ag,
      /* out array<double> */::sidl::array<double>& fg,
      /* out */int32_t& size
    )
    ;



    /**
     *  Starts up a component presence in the calling framework.
     * @param services the component instance's handle on the framework world.
     * Contracts concerning Svc and setServices:
     * 
     * The component interaction with the CCA framework
     * and Ports begins on the call to setServices by the framework.
     * 
     * This function is called exactly once for each instance created
     * by the framework.
     * 
     * The argument Svc will never be nil/null.
     * 
     * Those uses ports which are automatically connected by the framework
     * (so-called service-ports) may be obtained via getPort during
     * setServices.
     */
    void
    setServices (
      /* in */const ::gov::cca::Services& services
    )
    // throws:
    //    ::gov::cca::CCAException
    //    ::sidl::RuntimeException
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct pde_FEM__object ior_t;
    typedef struct pde_FEM__external ext_t;
    typedef struct pde_FEM__sepv sepv_t;

    // default constructor
    FEM() { }
    // static constructor
    static ::pde::FEM _create();


#ifdef WITH_RMI

    // RMI constructor
    static ::pde::FEM _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::pde::FEM _connect( /*in*/ const std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::pde::FEM _connect( /*in*/ const std::string& url, /*in*/ const 
      bool ar  );


#endif /*WITH_RMI*/

    // default destructor
    virtual ~FEM () { }

    // copy constructor
    FEM ( const FEM& original );

    // assignment operator
    FEM& operator= ( const FEM& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    FEM ( FEM::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    FEM ( FEM::ior_t* ior, bool isWeak );

    inline ior_t* _get_ior() const throw() {
      return reinterpret_cast< ior_t*>(d_self);
    }

    inline void _set_ior( ior_t* ptr ) throw () { 
      if(d_self == ptr) {return;}
      d_self = reinterpret_cast< void*>(ptr);

      if( ptr != NULL ) {
        pdeports_FEMmatrixPort_IORCache = &((*ptr).d_pdeports_femmatrixport);
        gov_cca_Port_IORCache = &((*ptr).d_gov_cca_port);
        gov_cca_Component_IORCache = &((*ptr).d_gov_cca_component);
      } else {
        pdeports_FEMmatrixPort_IORCache = NULL;
        gov_cca_Port_IORCache = NULL;
        gov_cca_Component_IORCache = NULL;
      }
    }

    virtual int _set_ior_typesafe( struct sidl_BaseInterface__object *obj,
                                   const ::std::type_info &argtype );

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return "pde.FEM";}

    static struct pde_FEM__object* _cast(const void* src);

    // execute member function by name
    void _exec(const std::string& methodName,
               ::sidl::rmi::Call& inArgs,
               ::sidl::rmi::Return& outArgs);

    /**
     * Get the URL of the Implementation of this object (for RMI)
     */
    ::std::string
    _getURL() // throws:
    //    ::sidl::RuntimeException
    ;


    /**
     * Method to enable/disable method hooks invocation.
     */
    void
    _set_hooks (
      /* in */bool enable
    )
    // throws:
    //    ::sidl::RuntimeException
    ;


    /**
     * Method to enable/disable interface contract enforcement.
     */
    void
    _set_contracts (
      /* in */bool enable,
      /* in */const ::std::string& enfFilename,
      /* in */bool resetCounters
    )
    // throws:
    //    ::sidl::RuntimeException
    ;


    /**
     * Method to dump contract enforcement statistics.
     */
    void
    _dump_stats (
      /* in */const ::std::string& filename,
      /* in */const ::std::string& prefix
    )
    // throws:
    //    ::sidl::RuntimeException
    ;

    // return true iff object is remote
    bool _isRemote() const { 
      ior_t* self = const_cast<ior_t*>(_get_ior() );
      struct sidl_BaseInterface__object *throwaway_exception;
      return (*self->d_epv->f__isRemote)(self, &throwaway_exception) == TRUE;
    }

    // return true iff object is local
    bool _isLocal() const {
      return !_isRemote();
    }

  protected:
    // Pointer to external (DLL loadable) symbols (shared among instances)
    static const ext_t * s_ext;

  public:
    static const ext_t * _get_ext() throw ( ::sidl::NullIORException );

  }; // end class FEM
} // end namespace pde

extern "C" {


#pragma weak pde_FEM__connectI

  /**
   * RMI connector function for the class. (no addref)
   */
  struct pde_FEM__object*
  pde_FEM__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::pde::FEM > {
    typedef array< ::pde::FEM > cxx_array_t;
    typedef ::pde::FEM cxx_item_t;
    typedef struct pde_FEM__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct pde_FEM__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::pde::FEM > > iterator;
    typedef const_array_iter< array_traits< ::pde::FEM > > const_iterator;
  };

  // array specialization
  template<>
  class array< ::pde::FEM >: public interface_array< array_traits< ::pde::FEM > 
    > {
  public:
    typedef interface_array< array_traits< ::pde::FEM > > Base;
    typedef array_traits< ::pde::FEM >::cxx_array_t          cxx_array_t;
    typedef array_traits< ::pde::FEM >::cxx_item_t           cxx_item_t;
    typedef array_traits< ::pde::FEM >::ior_array_t          ior_array_t;
    typedef array_traits< ::pde::FEM >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::pde::FEM >::ior_item_t           ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct pde_FEM__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::pde::FEM >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::pde::FEM >&
    operator =( const array< ::pde::FEM >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#endif
