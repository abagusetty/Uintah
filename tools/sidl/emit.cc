
#include "Spec.h"
#include "SymbolTable.h"
extern "C" {
#include <sys/uuid.h>
}
#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>

using std::cerr;
using std::for_each;
using std::map;
using std::string;
using std::vector;

static string handle_class = "\
class @ {\n\
    @_interface* ptr;\n\
public:\n\
    static const ::Component::PIDL::TypeInfo* _getTypeInfo();\n\
    typedef @_interface interfacetype;\n\
    inline @()\n\
    {\n\
        ptr=0;\n\
    }\n\
    inline @(@_interface* ptr)\n\
        : ptr(ptr)\n\
    {\n\
    }\n\
    inline ~@()\n\
    {\n\
    }\n\
    inline @(const @& copy)\n\
        : ptr(copy.ptr)\n\
    {\n\
    }\n\
    inline @& operator=(const @& copy)\n\
    {\n\
        ptr=copy.ptr;\n\
        return *this;\n\
    }\n\
    inline @_interface* operator->() const\n\
    {\n\
        return ptr;\n\
    }\n\
    inline operator bool() const\n\
    {\n\
        return ptr != 0;\n\
    }\n\
";

struct Leader {
};
static Leader leader;

struct SState : public std::ostringstream {
    std::string leader;
    SymbolTable* currentPackage;
    EmitState*e;
    SState(EmitState* e) : e(e) {
	currentPackage=0;
	leader="";
    }
    void begin_namespace(SymbolTable*);
    void close_namespace();
    void recursive_begin_namespace(SymbolTable*);
    std::ostream& operator<<(const Leader&) {
	*this << leader;
	return *this;
    }
    std::string push_leader() {
	string oldleader=leader;
	leader+="    ";
	return oldleader;
    }
    void pop_leader(const std::string& oldleader) {
	leader=oldleader;
    }
};

struct EmitState {
    int instanceNum;
    int handlerNum;
    EmitState();

    SState fwd;
    SState decl;
    SState out;
    SState proxy;
};

EmitState::EmitState()
    : fwd(this), decl(this), out(this), proxy(this)
{
    instanceNum=0;
    handlerNum=0;
}

void Specification::emit(std::ostream& out, std::ostream& hdr) const
{
    EmitState e;
    // Emit code for each definition
    globals->emit(e);
    e.fwd.close_namespace();
    e.decl.close_namespace();
    e.out.close_namespace();
    e.proxy.close_namespace();

    hdr << "/*\n";
    hdr << " * This code was automatically generated by sidl,\n";
    hdr << " * do not edit directly!\n";
    hdr << " */\n";
    hdr << "\n";
    hdr << "#ifndef xxx_replace_this_later\n";
    hdr << "#define xxx_replace_this_later\n";
    hdr << "\n";
    hdr << "#include <Component/PIDL/Object.h>\n";
    hdr << "#include <Component/PIDL/pidl_cast.h>\n";
    hdr << "#include <Component/PIDL/array.h>\n";
    hdr << "#include <Component/PIDL/string.h>\n";
    hdr << "\n";
    hdr << e.fwd.str();
    hdr << e.decl.str();
    hdr << "\n#endif\n\n";

    // Emit #includes
    out << "/*\n";
    out << " * This code was automatically generated by sidl,\n";
    out << " * do not edit directly!\n";
    out << " */\n";
    out << "\n";
    out << "#include \"PingPong_sidl.h\"\n";
    out << "#include <SCICore/Exceptions/InternalError.h>\n";
    out << "#include <Component/PIDL/GlobusError.h>\n";
    out << "#include <Component/PIDL/Object_proxy.h>\n";
    out << "#include <Component/PIDL/ProxyBase.h>\n";
    out << "#include <Component/PIDL/ReplyEP.h>\n";
    out << "#include <Component/PIDL/Reference.h>\n";
    out << "#include <Component/PIDL/TypeInfo.h>\n";
    out << "#include <Component/PIDL/TypeInfo_internal.h>\n";
    out << "#include <SCICore/Util/NotFinished.h>\n";
    out << "#include <SCICore/Thread/Thread.h>\n";
    out << "#include <iostream>\n";
    out << "#include <globus_nexus.h>\n";
    out << "\n";
    out << e.proxy.str();
    out << e.out.str();
}

void SymbolTable::emit(EmitState& e) const
{
    for(map<string, Symbol*>::const_iterator iter=symbols.begin();
	iter != symbols.end();iter++){
	iter->second->emit(e);
    }
}

void Symbol::emit(EmitState& e)
{
    switch(type){
    case PackageType:
	definition->getSymbolTable()->emit(e);
	break;
    case InterfaceType:
    case ClassType:
	definition->emit(e);
	break;
    case MethodType:
	cerr << "Symbol::emit called for a method!\n";
	exit(1);
    }
    emitted_forward=true;
}

void Symbol::emit_forward(EmitState& e)
{
    if(emitted_forward)
	return;
    switch(type){
    case PackageType:
	cerr << "Why is emit forward being called for a package?\n";
	exit(1);
    case InterfaceType:
    case ClassType:
	if(definition->emitted_declaration){
	    emitted_forward=true;
	    return;
	}
	e.fwd.begin_namespace(symtab);
	e.fwd << leader << "class " << name << ";\n";
	break;
    case MethodType:
	cerr << "Symbol::emit_forward called for a method!\n";
	exit(1);
    }
    emitted_forward=true;
}	

void Package::emit(EmitState&)
{
    cerr << "Package::emit should not be called...\n";
    exit(1);
}

void SState::begin_namespace(SymbolTable* stab)
{
    if(currentPackage == stab)
	return;
    // Close off previous namespace...
    close_namespace();

    // Open new namespace...
    recursive_begin_namespace(stab);
    currentPackage=stab;
}

void SState::close_namespace()
{
    if(currentPackage){
	while(currentPackage->getParent()){
	    for(SymbolTable* p=currentPackage->getParent();p->getParent()!=0;p=p->getParent())
		*this << "    ";
	    *this << "} // End namespace " << currentPackage->getName() << '\n';
	    currentPackage=currentPackage->getParent();
	}
	*this << "\n";
    }
    leader="";
}

void SState::recursive_begin_namespace(SymbolTable* stab)
{
    SymbolTable* parent=stab->getParent();
    if(parent){
	recursive_begin_namespace(parent);
	*this << leader << "namespace " << stab->getName() << " {\n";
	push_leader();
    }
}

bool CI::iam_class()
{
    bool iam=false;
    if(dynamic_cast<Class*>(this))
	iam=true;
    return iam;
}

void CI::emit(EmitState& e)
{
    if(emitted_declaration)
	return;
    // Emit parent classes...
    if(parentclass)
	parentclass->emit(e);
    for(std::vector<Interface*>::iterator iter=parent_ifaces.begin();
	iter != parent_ifaces.end(); iter++){
	(*iter)->emit(e);
    }

    emit_header(e);
    emit_proxyclass(e);

    e.instanceNum++;

    // Emit handler functions
    emit_handlers(e);

    // Emit handler table
    emit_handler_table(e);

    emit_typeinfo(e);

    emit_interface(e);

    emit_proxy(e);
    emitted_declaration=true;
}

void CI::emit_typeinfo(EmitState& e)
{
    std::string fn=cppfullname(0);
    uuid_t uuid;
    uint_t status;
    uuid_create(&uuid, &status);
    if(status != uuid_s_ok){
	cerr << "Error creating uuid!\n";
	exit(1);
    }
    char* uuid_str;
    uuid_to_string(&uuid, &uuid_str, &status);
    if(status != uuid_s_ok){
	cerr << "Error creating uuid string!\n";
	exit(1);
    }
    e.out << "const ::Component::PIDL::TypeInfo* " << fn << "::_getTypeInfo()\n";
    e.out << "{\n";
    e.out << "    static ::Component::PIDL::TypeInfo* ti=0;\n";
    e.out << "    if(!ti){\n";
    e.out << "        ::Component::PIDL::TypeInfo_internal* tii=\n";
    e.out << "            new ::Component::PIDL::TypeInfo_internal(\"" << cppfullname(0) << "\", \"" << uuid_str << "\",\n";
    e.out << "                                                     _handler_table" << e.instanceNum << ",\n";
    e.out << "                                                     sizeof(_handler_table" << e.instanceNum << ")/sizeof(globus_nexus_handler_t),\n";
    e.out << "                                                     &" << fn << "_proxy::create_proxy);\n\n";
    SymbolTable* localScope=symbols->getParent();
    if(parentclass)
	e.out << "        tii->add_parentclass(" << parentclass->cppfullname(localScope) << "::_getTypeInfo(), " << parentclass->vtable_base << ");\n";
    for(vector<Interface*>::iterator iter=parent_ifaces.begin();
	iter != parent_ifaces.end(); iter++){
	e.out << "        tii->add_parentiface(" << (*iter)->cppfullname(localScope) << "::_getTypeInfo(), " << (*iter)->vtable_base << ");\n";
    }
    e.out << "        ti=new ::Component::PIDL::TypeInfo(tii);\n";
    e.out << "    }\n";
    e.out << "    return ti;\n";
    e.out << "}\n\n";

    free(uuid_str);
}

void CI::emit_handlers(EmitState& e)
{
    // Emit isa handler...
    e.out << "// methods from " << name << " " << curfile << ":" << lineno << "\n\n";
    e.out << "// isa handler\n";
    isaHandler=++e.handlerNum;
    e.out << "static void _handler" << isaHandler << "(globus_nexus_endpoint_t*,\n";
    e.out << "                      globus_nexus_buffer_t* _recvbuff, globus_bool_t)\n";
    e.out << "{\n";
    e.out << "    int classname_size;\n";
    e.out << "    globus_nexus_get_int(_recvbuff, &classname_size, 1);\n";
    e.out << "    char* classname=new char[classname_size+1];\n";
    e.out << "    globus_nexus_get_char(_recvbuff, classname, classname_size);\n";
    e.out << "    classname[classname_size]=0;\n";
    e.out << "    int uuid_size;\n";
    e.out << "    globus_nexus_get_int(_recvbuff, &uuid_size, 1);\n";
    e.out << "    char* uuid=new char[uuid_size+1];\n";
    e.out << "    globus_nexus_get_char(_recvbuff, uuid, uuid_size);\n";
    e.out << "    uuid[uuid_size]=0;\n";
    e.out << "    globus_nexus_startpoint_t _sp;\n";
    e.out << "    if(int _gerr=globus_nexus_get_startpoint(_recvbuff, &_sp, 1))\n";
    e.out << "        throw ::Component::PIDL::GlobusError(\"get_startpoint\", _gerr);\n";
    e.out << "    if(int _gerr=globus_nexus_buffer_destroy(_recvbuff))\n";
    e.out << "        throw ::Component::PIDL::GlobusError(\"buffer_destroy\", _gerr);\n";
    e.out << "    const ::Component::PIDL::TypeInfo* ti=" << cppfullname(0) << "::_getTypeInfo();\n";
    e.out << "    int result=ti->isa(classname, uuid);\n";
    e.out << "    delete[] classname;\n";
    e.out << "    delete[] uuid;\n";
    e.out << "    int flag;\n";
    e.out << "    if(result == ::Component::PIDL::TypeInfo::vtable_invalid)\n";
    e.out << "        flag=0;\n";
    e.out << "    else\n";
    e.out << "        flag=1;\n";
    e.out << "    globus_nexus_buffer_t sendbuff;\n";
    e.out << "    int rsize=globus_nexus_sizeof_int(2);\n";
    e.out << "    if(int gerr=globus_nexus_buffer_init(&sendbuff, rsize, 0))\n";
    e.out << "        throw ::Component::PIDL::GlobusError(\"buffer_init\", gerr);\n";
    e.out << "    globus_nexus_put_int(&sendbuff, &flag, 1);\n";
    e.out << "    globus_nexus_put_int(&sendbuff, &result, 1);\n";
    e.out << "    if(int gerr=globus_nexus_send_rsr(&sendbuff, &_sp, 0, GLOBUS_TRUE, GLOBUS_FALSE))\n";
    e.out << "        throw ::Component::PIDL::GlobusError(\"send_rsr\", gerr);\n";
    e.out << "    if(int gerr=globus_nexus_startpoint_destroy(&_sp))\n";
    e.out << "        throw ::Component::PIDL::GlobusError(\"startpoint_destroy\", gerr);\n";
    e.out << "}\n\n";

    // Emit method handlers...
    std::vector<Method*> vtab;
    gatherVtable(vtab);
    int handlerOff=0;
    for(vector<Method*>::const_iterator iter=vtab.begin();
	iter != vtab.end();iter++){
	Method* m=*iter;
	e.handlerNum++;
	m->handlerNum=e.handlerNum;
	m->handlerOff=handlerOff++;
	m->emit_handler(e);
    }
}

void CI::emit_recursive_vtable_comment(EmitState& e, bool top)
{
    e.out << "  // " << (top?"":"and ") << (iam_class()?"class ":"interface ") << name << "\n";
    if(parentclass)
	parentclass->emit_recursive_vtable_comment(e, false);

    for(vector<Interface*>::const_iterator iter=parent_ifaces.begin();
	iter != parent_ifaces.end(); iter++){
	(*iter)->emit_recursive_vtable_comment(e, false);
    }
}

bool CI::singly_inherited() const
{
    // A class is singly inherited if it has one parent class,
    // or one parent interface, and it's parent is singly_inherited
    if((parentclass && parent_ifaces.size() > 0)
       || parent_ifaces.size()>1)
	return false;
    if(parentclass){
	if(!parentclass->singly_inherited())
	    return false;
    } else if(parent_ifaces.size()>0){
	// First element...
	if(!(*parent_ifaces.begin())->singly_inherited())
	    return false;
    }
    return true;
}

void CI::emit_handler_table_body(EmitState& e, int& vtable_base, bool top)
{
    bool single=singly_inherited();
    if(single)
	emit_recursive_vtable_comment(e, true);
    else
	e.out << "  // " << (iam_class()?"class ":"interface ") << name << "\n";
    e.out << "  // vtable_base = " << vtable_base << '\n';
    std::vector<Method*> vtab;
    gatherVtable(vtab);
    for(vector<Method*>::const_iterator iter=vtab.begin();
	iter != vtab.end();iter++){
	if(iter != vtab.begin())
	    e.out << ",\n";
	Method* m=*iter;
	m->emit_comment(e, "    ", false);
	e.out << "    {GLOBUS_NEXUS_HANDLER_TYPE_THREADED, _handler" << m->handlerNum << "}";
    } 
    e.out << ",\n    // Red zone\n";    
    e.out << "    {GLOBUS_NEXUS_HANDLER_TYPE_THREADED, 0},\n";

    if(single){
	if(top){
	    if(parentclass)
		parentclass->vtable_base=vtable_base;
	    for(vector<Interface*>::iterator iter=parent_ifaces.begin();
		iter != parent_ifaces.end(); iter++){
		(*iter)->vtable_base=vtable_base;
	    }
	}
	return;
    }
    // For each parent, emit the handler table...
    vtable_base+=vtab.size()+1;
    if(parentclass){
	if(top)
	    parentclass->vtable_base=vtable_base;
	parentclass->emit_handler_table_body(e, vtable_base, false);
    }
    for(vector<Interface*>::iterator iter=parent_ifaces.begin();
	iter != parent_ifaces.end(); iter++){
	if(top)
	    (*iter)->vtable_base=vtable_base;
	(*iter)->emit_handler_table_body(e, vtable_base, false);
    }
}

void CI::emit_handler_table(EmitState& e)
{
    e.out << "// handler table for " << (iam_class()?"class ":"interface ") << name << "\n";
    e.out << "//" << curfile << ":" << lineno << "\n\n";
    e.out << "static globus_nexus_handler_t _handler_table" << e.instanceNum << "[] =\n";
    e.out << "{\n";
    e.out << "    {GLOBUS_NEXUS_HANDLER_TYPE_THREADED, _handler" << isaHandler << "},\n";
    e.out << "    {GLOBUS_NEXUS_HANDLER_TYPE_THREADED, 0},\n";
    int vtable_base=2;

    emit_handler_table_body(e, vtable_base, true);

    e.out << "\n}; // vtable_size=" << vtable_base << "\n\n";
}

bool Method::reply_required() const
{
    return true; // For now
}

string Method::get_classname() const
{
    string n;
    if(myclass)
	n=myclass->cppfullname(0);
    else if(myinterface)
	n=myinterface->cppfullname(0);
    else {
	cerr << "ERROR: Method does not have an associated class or interface\n";
	exit(1);
    }
    return n+"_interface";
}

void Method::emit_comment(EmitState& e, const std::string& leader,
			  bool print_filename) const
{
    if(print_filename)
	e.out << leader << "// from " << curfile << ":" << lineno << '\n';
    e.out << leader << "// " << fullsignature() << '\n';
}

void Method::emit_prototype(SState& out, Context ctx,
			    SymbolTable* localScope) const
{
    out << leader << "// " << fullsignature() << '\n';
    out << leader << "virtual ";
    return_type->emit_prototype(out, Type::ReturnType, localScope);
    out << " " << name << "(";
    std::vector<Argument*>& list=args->getList();
    int c=0;
    for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	if(c++>0)
	    out << ", ";
	Argument* arg=*iter;
	arg->emit_prototype(out, localScope);
    }
    out << ")";
    if(ctx == PureVirtual)
	out << "=0";
    out << ";\n";
}

void Method::emit_prototype_defin(EmitState& e, const std::string& prefix,
				  SymbolTable* localScope) const
{
    e.out << "// " << fullsignature() << '\n';
    return_type->emit_prototype(e.out, Type::ReturnType, 0);
    e.out << " " << prefix << name << "(";
    std::vector<Argument*>& list=args->getList();
    int c=0;
    for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	if(c++>0)
	    e.out << ", ";
	Argument* arg=*iter;
	std::ostringstream argname;
	argname << "_arg" << c;
	arg->emit_prototype_defin(e.out, argname.str(), localScope);
    }
    e.out << ")";
}

void Method::emit_handler(EmitState& e) const
{
    // Server-side handlers
    emit_comment(e, "", true);
    e.out << "static void _handler" << e.handlerNum << "(globus_nexus_endpoint_t* _ep,\n";
    e.out << "                      globus_nexus_buffer_t* _recvbuff, globus_bool_t)\n";
    e.out << "{\n";

    std::vector<Argument*>& list=args->getList();
    int argNum=0;
    for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	argNum++;
	Argument* arg=*iter;
	if(arg->getMode() != Argument::Out) {
	    std::ostringstream argname;
	    argname << "_arg" << argNum;
	    arg->emit_unmarshall(e, argname.str(), "_recvbuff");
	}
    }

    // If we are sending a reply, unmarshall the startpoint
    e.out << "\n";
    if(reply_required()){
	e.out << "    globus_nexus_startpoint_t _sp;\n";
	e.out << "    if(int _gerr=globus_nexus_get_startpoint(_recvbuff, &_sp, 1))\n";
	e.out << "        throw ::Component::PIDL::GlobusError(\"get_startpoint\", _gerr);\n";
    }
    // Destroy the buffer...
    e.out << "    if(int _gerr=globus_nexus_buffer_destroy(_recvbuff))\n";
    e.out << "        throw ::Component::PIDL::GlobusError(\"buffer_destroy\", _gerr);\n";
    e.out << "    void* _v=globus_nexus_endpoint_get_user_pointer(_ep);\n";
    string myclass=get_classname();
    e.out << "    " << myclass << "* _obj=static_cast<" << myclass << "*>(_v);\n";
    e.out << "\n";
    e.out << "    // Call the method\n";
    // Call the method...
    e.out << "    ";
    if(return_type){
	if(!return_type->isvoid()){
	    return_type->emit_rettype(e, "_ret");
	    e.out << " = ";
	}
    }
    e.out << "_obj->" << name << "(";
    argNum=0;
    for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	if(argNum > 0)
	    e.out << ", ";
	argNum++;
	e.out << "_arg" << argNum;
    }
    e.out << ");\n";

    // Clean up in arguments
    if(reply_required()){
	// Set up startpoints for any objects...
	if(return_type){
	    if(!return_type->isvoid()){
		return_type->emit_startpoints(e, "_ret");
	    }
	}
	argNum=0;
	for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	    argNum++;
	    Argument* arg=*iter;
	    if(arg->getMode() != Argument::In) {
		std::ostringstream argname;
		argname << "_arg" << argNum;
		arg->emit_startpoints(e, argname.str());
	    }
	}
	// Size the reply...
	e.out << "    unsigned long _rsize=globus_nexus_sizeof_int(1)";
	if(return_type){
	    if(!return_type->isvoid()){
		return_type->emit_marshallsize(e, "_ret");
	    }
	}
	argNum=0;
	for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	    argNum++;
	    Argument* arg=*iter;
	    if(arg->getMode() != Argument::In) {
		std::ostringstream argname;
		argname << "_arg" << argNum;
		arg->emit_marshallsize(e, argname.str());
	    }
	}
	e.out << ";\n";
	e.out << "    globus_nexus_buffer_t _sendbuff;\n";
	e.out << "    if(int _gerr=globus_nexus_buffer_init(&_sendbuff, _rsize, 0))\n";
	e.out << "        throw ::Component::PIDL::GlobusError(\"buffer_init\", _gerr);\n";
	e.out << "    int _flag=0;\n";
	e.out << "    globus_nexus_put_int(&_sendbuff, &_flag, 1);\n";
	if(return_type){
	    if(!return_type->isvoid()){
		e.out << "    // Marshall return value\n";
		return_type->emit_marshall(e, "_ret", "&_sendbuff");
	    }
	}
	e.out << "    // Marshall arguments\n";
	argNum=0;
	for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	    argNum++;
	    Argument* arg=*iter;
	    if(arg->getMode() != Argument::In) {
		std::ostringstream argname;
		argname << "_arg" << argNum;
		arg->emit_marshall(e, argname.str(), "&_sendbuff");
	    }
	}

	e.out << "    // Send the reply...\n";
	int reply_handler_id=0; // Always 0
	e.out << "    if(int _gerr=globus_nexus_send_rsr(&_sendbuff, &_sp, " << reply_handler_id << ", GLOBUS_TRUE, GLOBUS_FALSE))\n";
	e.out << "        throw ::Component::PIDL::GlobusError(\"send_rsr\", _gerr);\n";
	e.out << "    if(int _gerr=globus_nexus_startpoint_destroy(&_sp))\n";
	e.out << "        throw ::Component::PIDL::GlobusError(\"startpoint_destroy\", _gerr);\n";
    }
    // Clean up inout and out arguments
    
    e.out << "}\n\n";
}

class output_sub {
    SState& out;
    std::string classname;
public:
    output_sub(SState& out, const std::string& classname)
	: out(out), classname(classname) {}
    void operator()(char x) {
	if(x=='@')
	    out << classname;
	else if(x=='\n')
	    out << '\n' << out.leader;
	else
	    out << x;
    }
};

void CI::emit_proxyclass(EmitState& e)
{
    e.proxy.begin_namespace(symbols->getParent());

    // Proxy
    std::string pname=name+"_proxy";
    std::string iname=name+"_interface";

    e.proxy << leader << "class " << pname << " : public ::Component::PIDL::ProxyBase, public " << iname << " {\n";
    e.proxy << leader << "public:\n";
    e.proxy << leader << "    " << pname << "(const ::Component::PIDL::Reference&);\n";
    std::string oldleader=e.proxy.push_leader();
    std::vector<Method*> vtab;
    gatherVtable(vtab);

    for(vector<Method*>::const_iterator iter=vtab.begin();
	iter != vtab.end();iter++){
	e.proxy << '\n';
	Method* m=*iter;
	m->emit_prototype(e.proxy, Method::Normal, symbols->getParent());
    }
    e.proxy.pop_leader(oldleader);
    e.proxy << leader << "protected:\n";
    e.proxy << leader << "    virtual ~" << pname << "();\n";
    e.proxy << leader << "private:\n";
    e.proxy << leader << "    virtual void _getReference(::Component::PIDL::Reference&, bool copy) const;\n";
    e.proxy << leader << "    friend const ::Component::PIDL::TypeInfo* " << name << "::_getTypeInfo();\n";
    e.proxy << leader << "    static ::Component::PIDL::Object_interface* create_proxy(const ::Component::PIDL::Reference&);\n";
    e.proxy << leader << "    " << pname << "(const " << pname << "&);\n";
    e.proxy << leader << "    " << pname << "& operator=(const " << pname << "&);\n";
    e.proxy << leader << "};\n\n";
    e.proxy.close_namespace();
}

void CI::emit_header(EmitState& e)
{
    e.decl.begin_namespace(symbols->getParent());

    std::vector<Method*>& mymethods=myMethods();

    // interface
    std::string iname=name+"_interface";
    e.decl << leader << "class " << iname << " : ";

    // Parents
    bool haveone=false;
    if(parentclass){
	e.decl << "public " << parentclass->cppfullname(e.decl.currentPackage) << "_interface";
	haveone=true;
    }
    for(vector<Interface*>::iterator iter=parent_ifaces.begin();
	iter != parent_ifaces.end(); iter++){
	if(!haveone){
	    haveone=true;
	} else {
	    e.decl << ", ";
	}
	if(!iam_class())
	    e.decl << "virtual ";
	e.decl << "public " << (*iter)->cppfullname(e.decl.currentPackage) << "_interface";
    }
    if(!haveone)
	e.decl << "virtual public ::Component::PIDL::Object_interface";
    e.decl << " {\n";

    // The interace class body
    e.decl << leader << "public:\n";
    e.decl << leader << "    virtual ~" << iname << "();\n";
    std::string oldleader=e.decl.push_leader();
    for(vector<Method*>::const_iterator iter=mymethods.begin();
	iter != mymethods.end();iter++){
	e.decl << '\n';
	Method* m=*iter;
	m->emit_prototype(e.decl, Method::PureVirtual, symbols->getParent());
    }
    e.decl.pop_leader(oldleader);
    // The type signature method...
    e.decl << leader << "    virtual const ::Component::PIDL::TypeInfo* _getTypeInfo() const;\n";
    e.decl << leader << "protected:\n";
    e.decl << leader << "    " << iname << "(bool initServer=true);\n";
    e.decl << leader << "private:\n";
    e.decl << leader << "    " << iname << "(const " << iname << "&);\n";
    e.decl << leader << "    " << iname << "& operator=(const " << iname << "&);\n";
    e.decl << leader << "};\n\n";

    // The Handle
    e.decl << leader;
    for_each(handle_class.begin(), handle_class.end(), output_sub(e.decl, name));
    e.decl << "    // Conversion operations\n";
    // Emit an operator() for each parent class and interface...
    vector<CI*> parents;
    gatherParents(parents);
    for(std::vector<CI*>::iterator iter=parents.begin();
	iter != parents.end(); iter++){
	if(*iter != this){
	    e.decl << leader << "    inline operator " << (*iter)->cppfullname(e.decl.currentPackage) << "() const\n";
	    e.decl << leader << "    {\n";
	    e.decl << leader << "        return ptr;\n";
	    e.decl << leader << "    }\n";
	    e.decl << "\n";
	}
    }
    e.decl << leader << "};\n\n";
}

void CI::emit_interface(EmitState& e)
{
    std::string fn=cppfullname(0)+"_interface";
    std::string cn=cppclassname()+"_interface";
    e.out << fn << "::" << cn << "(bool initServer)\n";
    e.out << "{\n";
    e.out << "    if(initServer)\n";
    e.out << "        initializeServer(" << cppfullname(0) << "::_getTypeInfo(), this);\n";
    e.out << "}\n\n";

    e.out << fn << "::~" << cn << "()\n";
    e.out << "{\n";
    e.out << "}\n\n";

    e.out << "const ::Component::PIDL::TypeInfo* " << fn << "::_getTypeInfo() const\n";
    e.out << "{\n";
    e.out << "    return " << cppfullname(0) << "::_getTypeInfo();\n";
    e.out << "}\n\n";
}

void CI::emit_proxy(EmitState& e)
{
    std::string fn=cppfullname(0)+"_proxy";
    if(fn[0] == ':' && fn[1] == ':')
	fn=fn.substr(2);
    std::string cn=cppclassname()+"_proxy";
    e.out << fn << "::" << cn << "(const ::Component::PIDL::Reference& ref)\n";
    e.out << " : ::Component::PIDL::ProxyBase(ref)";
    e.out << ", " << cppclassname() << "_interface(false)\n";
    e.out << "\n";
    e.out << "{\n";
    e.out << "}\n\n";
    e.out << fn << "::~" << cn << "()\n";
    e.out << "{\n";
    e.out << "}\n\n";
    e.out << "void " << fn << "::_getReference(::Component::PIDL::Reference& ref, bool copy) const\n";
    e.out << "{\n";
    e.out << "    _proxyGetReference(ref, copy);\n";
    e.out << "}\n\n";
    e.out << "::Component::PIDL::Object_interface* " << fn << "::create_proxy(const ::Component::PIDL::Reference& ref)\n";
    e.out << "{\n";
    e.out << "    return new " << cn << "(ref);\n";
    e.out << "}\n\n";

    std::vector<Method*> vtab;
    gatherVtable(vtab);
    SymbolTable* localScope=symbols->getParent();
    for(vector<Method*>::const_iterator iter=vtab.begin();
	iter != vtab.end();iter++){
	e.out << '\n';
	Method* m=*iter;
	m->emit_proxy(e, fn, localScope);
    }
}

void Method::emit_proxy(EmitState& e, const string& fn,
			SymbolTable* localScope) const
{
    emit_prototype_defin(e, fn+"::", localScope);
    e.out << "\n{\n";
    if(reply_required())
	e.out << "    ::Component::PIDL::ReplyEP* _reply=::Component::PIDL::ReplyEP::acquire();\n";
    e.out << "    globus_nexus_startpoint_t _sp;\n";
    e.out << "    _reply->get_startpoint(&_sp);\n\n";
    std::vector<Argument*>& list=args->getList();
    int argNum=0;
    for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	argNum++;
	Argument* arg=*iter;
	if(arg->getMode() != Argument::Out) {
	    std::ostringstream argname;
	    argname << "_arg" << argNum;
	    arg->emit_startpoints(e, argname.str());
	}
    }
    e.out << "    // Size the buffer\n";
    e.out << "    int _size=";
    if(reply_required())
	e.out << "globus_nexus_sizeof_startpoint(&_sp, 1)";
    else
	e.out << "0";
    argNum=0;
    for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	argNum++;
	Argument* arg=*iter;
	if(arg->getMode() != Argument::Out) {
	    std::ostringstream argname;
	    argname << "_arg" << argNum;
	    arg->emit_marshallsize(e, argname.str());
	}
    }
    e.out << ";\n";
    e.out << "    globus_nexus_buffer_t _buffer;\n";
    e.out << "    if(int _gerr=globus_nexus_buffer_init(&_buffer, _size, 0))\n";
    e.out << "        throw ::Component::PIDL::GlobusError(\"buffer_init\", _gerr);\n";
    e.out << "    // Marshall the arguments\n";
    //... emit_marshall...;
    argNum=0;
    for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	argNum++;
	Argument* arg=*iter;
	if(arg->getMode() != Argument::Out) {
	    std::ostringstream argname;
	    argname << "_arg" << argNum;
	    arg->emit_marshall(e, argname.str(), "&_buffer");
	}
    }

    if(reply_required()){
	e.out << "    // Marshall the reply startpoint\n";
	e.out << "    globus_nexus_put_startpoint_transfer(&_buffer, &_sp, 1);\n";
    }
    e.out << "    // Send the message\n";
    e.out << "    ::Component::PIDL::Reference _ref;\n";
    e.out << "    _proxyGetReference(_ref, false);\n";
    e.out << "    int _handler=_ref.getVtableBase()+" << handlerOff << ";\n";
    e.out << "    if(int _gerr=globus_nexus_send_rsr(&_buffer, &_ref.d_sp,\n";
    e.out << "                                       _handler, GLOBUS_TRUE, GLOBUS_FALSE))\n";
    e.out << "        throw ::Component::PIDL::GlobusError(\"send_rsr\", _gerr);\n";
    if(reply_required()){
	e.out << "    globus_nexus_buffer_t _recvbuff=_reply->wait();\n";
	//... emit unmarshall...;
	e.out << "    int _flag;\n";
	e.out << "    globus_nexus_get_int(&_recvbuff, &_flag, 1);\n";
	e.out << "    if(_flag != 0)\n";
	e.out << "        NOT_FINISHED(\"Exceptions not implemented\");\n";
	e.out << "    // Unmarshall the return values\n";
	if(return_type){
	    if(!return_type->isvoid()){
		return_type->emit_unmarshall(e, "_ret", "&_recvbuff");
	    }
	}
	argNum=0;
	for(vector<Argument*>::const_iterator iter=list.begin();iter != list.end();iter++){
	    argNum++;
	    Argument* arg=*iter;
	    if(arg->getMode() != Argument::In) {
		std::ostringstream argname;
		argname << "_arg" << argNum;
		arg->emit_unmarshall(e, argname.str(), "&_recvbuff");
	    }
	}
	e.out << "    ::Component::PIDL::ReplyEP::release(_reply);\n";
	e.out << "    if(int _gerr=globus_nexus_buffer_destroy(&_recvbuff))\n";
	e.out << "        throw ::Component::PIDL::GlobusError(\"buffer_destroy\", _gerr);\n";
	if(return_type){
	    if(!return_type->isvoid()){
		e.out << "    return _ret;\n";
	    }
	}
    }
    e.out << "}\n";
}

void Argument::emit_unmarshall(EmitState& e, const string& arg,
			       const std::string& bufname) const
{
    e.out << "    // " << arg << ": " << fullsignature() << "\n";
    type->emit_unmarshall(e, arg, bufname);
}

void Argument::emit_marshallsize(EmitState& e, const string& arg) const
{
    type->emit_marshallsize(e, arg);
}

void Argument::emit_startpoints(EmitState& e, const string& arg) const
{
    type->emit_startpoints(e, arg);
}

void Argument::emit_marshall(EmitState& e, const string& arg,
			     const std::string& bufname) const
{
    type->emit_marshall(e, arg, bufname);
}

void Argument::emit_prototype(SState& out, SymbolTable* localScope) const
{
    Type::ArgContext ctx;
    switch(mode){
    case In:
	ctx=Type::ArgIn;
	break;
    case Out:
	ctx=Type::ArgOut;
	break;
    case InOut:
	ctx=Type::ArgInOut;
	break;
    }
    type->emit_prototype(out, ctx, localScope);
    if(id != "" && id != "this")
	out << " " << id;
}

void Argument::emit_prototype_defin(SState& out, const std::string& arg,
				    SymbolTable* localScope) const
{
    Type::ArgContext ctx;
    switch(mode){
    case In:
	ctx=Type::ArgIn;
	break;
    case Out:
	ctx=Type::ArgOut;
	break;
    case InOut:
	ctx=Type::ArgInOut;
	break;
    }
    type->emit_prototype(out, ctx, localScope);
    out << " " << arg;
}

void ArrayType::emit_unmarshall(EmitState& e, const string& arg,
				const std::string& bufname) const
{
    cerr << "ArrayType::emit_unmarshall not finished\n";
}

void ArrayType::emit_marshallsize(EmitState& e, const string& arg) const
{
    cerr << "ArrayType::emit_marshallsize not finished\n";
}

void ArrayType::emit_startpoints(EmitState&, const string&) const
{
    // None
}

void ArrayType::emit_marshall(EmitState& e, const string& arg,
			      const std::string& bufname) const
{
    cerr << "ArrayType::emit_marshall not finished\n";
}

void ArrayType::emit_rettype(EmitState& e, const string& arg) const
{
    cerr << "ArrayType::emit_rettype not finished\n";
}

void ArrayType::emit_prototype(SState& out, ArgContext ctx,
			       SymbolTable* localScope) const
{
    if(ctx == ArgIn){
	out << "const ";
    }
    out << "::Component::PIDL::array<";
    subtype->emit_prototype(out, ArrayTemplate, localScope);
    out << ", " << dim << ">";
    if(ctx == ArgOut || ctx == ArgInOut)
	out << "&";
}

void BuiltinType::emit_unmarshall(EmitState& e, const string& arg,
				  const std::string& bufname) const
{
    if(cname == "void"){
	// What?
	cerr << "Trying to unmarshall a void!\n";
	exit(1);
    } else if(cname == "bool"){
	e.out << "    globus_byte_t " << arg << "_tmp;\n";
	e.out << "    globus_nexus_get_byte (" << bufname << ", &" << arg << "_tmp, 1);\n";
	e.out << "    bool " << arg << "=(bool)" << arg << "_tmp;\n";
    } else {
	e.out << "    " << cname << " " << arg << ";\n";
	e.out << "    globus_nexus_get_" << nexusname << "(" << bufname << ", &" << arg << ", 1);\n";
    }
}

void BuiltinType::emit_marshallsize(EmitState& e, const string&) const
{
    if(cname == "void"){
	// What?
	cerr << "Trying to size a void!\n";
	exit(1);
    }
    e.out << "+\n        globus_nexus_sizeof_" << nexusname << "(1)";
}

void BuiltinType::emit_startpoints(EmitState&, const string&) const
{
    // None
}

void BuiltinType::emit_marshall(EmitState& e, const string& arg,
				const std::string& bufname) const
{
    if(cname == "void"){
	// What?
	cerr << "Trying to unmarshall a void!\n";
	exit(1);
    } else if(cname == "bool"){
	e.out << "    globus_byte_t " << arg << "_tmp = " << arg << ";\n";
	e.out << "    globus_nexus_put_byte (" << bufname << ", &" << arg << "_tmp, 1);\n";
    } else {
	e.out << "    globus_nexus_put_" << nexusname << "(" << bufname << ", &" << arg << ", 1);\n";
    }
}

void BuiltinType::emit_rettype(EmitState& e, const string& arg) const
{
    if(cname == "void"){
	// Nothing
	return;
    }
    e.out << cname << " " << arg;
}

void BuiltinType::emit_prototype(SState& out, ArgContext ctx,
				 SymbolTable*) const
{
    if(cname == "void"){
	// Nothing
	if(ctx == ReturnType)
	    out << "void";
	else {
	    cerr << "Illegal void type in argument list\n";
	    exit(1);
	}
    } else if(cname == "string"){
	switch(ctx){
	case ReturnType:
	case ArrayTemplate:
	    out << "::Component::PIDL::string";
	    break;
	case ArgIn:
	    out << "const ::Component::PIDL::string&";
	    break;
	case ArgOut:
	case ArgInOut:
	    out << "::Component::PIDL::string&";
	    break;
	}
    } else {
	switch(ctx){
	case ReturnType:
	case ArgIn:
	case ArrayTemplate:
	    out << cname;
	    break;
	case ArgOut:
	case ArgInOut:
	    out << cname << "&";
	    break;
	}
    }
}

void NamedType::emit_unmarshall(EmitState& e, const string& arg,
				const std::string& bufname) const
{
    e.out << "    int " << arg << "_vtable_base;\n";
    e.out << "    globus_nexus_get_int(" << bufname << ", &" << arg << "_vtable_base, 1);\n";
    e.out << "    " << name->cppfullname(0) << " " << arg << ";\n";
    e.out << "    if(" << arg << "_vtable_base == -1){\n";
    e.out << "        " << arg << "=0;\n";
    e.out << "    } else {\n";
    e.out << "        ::Component::PIDL::Reference _ref;\n";
    e.out << "        globus_nexus_get_startpoint(" << bufname << ", &_ref.d_sp, 1);\n";
    e.out << "        _ref.d_vtable_base=" << arg << "_vtable_base;\n";
    e.out << "        " << arg << "=new " << name->cppfullname(0) << "_proxy(_ref);\n";
    e.out << "    }\n";
}

void NamedType::emit_marshallsize(EmitState& e, const string& arg) const
{
    e.out << "+\n        globus_nexus_sizeof_int(1)";
    e.out << "+\n        (" << arg << "?globus_nexus_sizeof_startpoint(&" << arg << "_ref.d_sp, 1):0)";
}

void NamedType::emit_startpoints(EmitState& e, const string& arg) const
{
    e.out << "    ::Component::PIDL::Reference " << arg << "_ref;\n";
    e.out << "    if(" << arg << "){\n";
    e.out << "        " << arg << "->_getReference(" << arg << "_ref, true);\n";
    e.out << "    }\n";
}

void NamedType::emit_marshall(EmitState& e, const string& arg,
			      const std::string& bufname) const
{
    e.out << "    if(" << arg << "){\n";
    e.out << "        const ::Component::PIDL::TypeInfo* _dt=" << arg << "->_getTypeInfo();\n";
    e.out << "        const ::Component::PIDL::TypeInfo* _bt=" << name->cppfullname(0) << "::_getTypeInfo();\n";
    e.out << "        int _vtable_offset=_dt->computeVtableOffset(_bt);\n";
    e.out << "        int _vtable_base=" << arg << "_ref.getVtableBase()+_vtable_offset;\n";
    e.out << "        globus_nexus_put_int(" << bufname << ", &_vtable_base, 1);\n";
    e.out << "        globus_nexus_put_startpoint_transfer(" << bufname << ", &" << arg << "_ref.d_sp, 1);\n";
    e.out << "    } else {\n";
    e.out << "        int _vtable_base=-1; // Null ptr\n";
    e.out << "        globus_nexus_put_int(" << bufname << ", &_vtable_base, 1);\n";
    e.out << "    }\n";
}

void NamedType::emit_rettype(EmitState& e, const string& arg) const
{
    e.out << name->cppfullname(0) << " " << arg;
}

void NamedType::emit_prototype(SState& out, ArgContext ctx,
			       SymbolTable* localScope) const
{
    // Ensure that it is forward declared...
    name->getSymbol()->emit_forward(*out.e);
    switch(ctx){
    case ReturnType:
    case ArrayTemplate:
	out << name->cppfullname(localScope);
	break;
    case ArgIn:
	out << "const " << name->cppfullname(localScope) << "&";
	break;
    case ArgOut:
    case ArgInOut:
	out << name->cppfullname(localScope) << "&";
	break;
    }
}

//
// $Log$
// Revision 1.1  1999/09/17 05:07:26  sparker
// Added nexus code generation capability
//
//
