/**************************************
 *
 * Parser.cc
 *
 * Written by:
 *   Darby J Van Uitert
 *   SCI Institute
 *   January 2003
 *************************************/

#include "Parser.h"

///////////////////////////////
// Constructors and Destructors
///////////////////////////////
Parser::Parser()
{

}



Parser::~Parser() {

}


//////////////////////////////
// read_input_file
///////////////////////////////  
DOMNode* Parser::read_input_file(string filename)
{
  DOMNode* node;
  try {
    XMLPlatformUtils::Initialize();
    XercesDOMParser* parser = new XercesDOMParser;
    parser->setDoValidation(false);
    // Parse the input file
    parser->parse(filename.c_str());
    DOMDocument* doc = parser->adoptDocument();
    
    if( !doc ) {
      cerr << "Parse failed!\n";
      return 0;
    }
    
    delete parser;

    node = doc->getDocumentElement();
  } catch (const XMLException& toCatch) {
    char* ch = XMLString::transcode(toCatch.getMessage());
    string ex("XML Exception: " + string(ch));
    delete [] ch;
    return 0;
  }
  this->resolve_includes(node);
  this->resolve_imports(node);

  return node;
}

//////////////////////////////
// resolve_includes
//////////////////////////////
void Parser::resolve_includes(DOMNode* root)
{
  DOMNodeList* children = root->getChildNodes();

  for(int i=0; i<children->getLength(); i++) {
    DOMNode* child = children->item(i);
    if(child->getNodeType() == DOMNode::ELEMENT_NODE) {
      char* name = XMLString::transcode(child->getNodeName());
      if(strcmp(name,"include")==0) {
	// get external file
	DOMNamedNodeMap* attr = child->getAttributes();
	unsigned long num_attr = attr->getLength();
	for (unsigned long i = 0; i<num_attr; i++) {
	  char* attrName = XMLString::transcode(attr->item(i)->getNodeName());
	  char* attrValue = XMLString::transcode(attr->item(i)->getNodeValue());
	  if(strcmp(attrName,"href")==0) {
	    // open the file, read it, and replace the index node
	    DOMNode* include = this->read_input_file(attrValue);
	    DOMNode* to_insert = child->getOwnerDocument()->importNode(include, true);
	    //root->insertBefore(to_insert, child);
	    root->appendChild(to_insert);

	    XMLCh* str = XMLString::transcode("\n");
	    DOMText *leader = child->getOwnerDocument()->createTextNode(str);
	    delete [] str;
	    root->appendChild(leader);
	    root->removeChild(child);
	  }
	  delete [] name;
	  // recurse on child's children
	  this->resolve_includes(child);
	}
	child = child->getNextSibling();
      }
    } 
  } 
}

//////////////////////////////
// resolve_imports
//////////////////////////////
void Parser::resolve_imports(DOMNode* root)
{
  DOMNodeList* children = root->getChildNodes();
  
  for(int i=0; i<children->getLength(); i++) {
    DOMNode* child = children->item(i);
    if(child->getNodeType() == DOMNode::ELEMENT_NODE) {
      char* name = XMLString::transcode(child->getNodeName());
      if(strcmp(name,"import")==0) {
	// get external file
	DOMNamedNodeMap* attr = child->getAttributes();
	unsigned long num_attr = attr->getLength();
	
	for (unsigned long i = 0; i<num_attr; i++) {
	  char* attrName = XMLString::transcode(attr->item(i)->getNodeName());
	  char* attrValue = XMLString::transcode(attr->item(i)->getNodeValue());
	  if(strcmp(attrName,"name")==0) {
	    // open the file, read it, and replace the index node
	    DOMNode* include = this->read_input_file(attrValue);
	    DOMNode* to_insert = child->getOwnerDocument()->importNode(include, true);
	    // nodes to be substituted must be enclosed in a 
	    // "name" node
	    //if (strcmp(XMLString::transcode(include->getNodeName()),"name")==0) {
	    root->appendChild(to_insert);
	    XMLCh* str = XMLString::transcode("\n");
	    DOMText *leader = child->getOwnerDocument()->createTextNode(str);
	    delete [] str;
	    root->appendChild(leader);
	    root->removeChild(child);
	  }
	  delete [] name;
	  // recurse on child's children
	  this->resolve_imports(child);
	} 
	child = child->getNextSibling();
      }
    } 
  } 
}

//////////////////////////////
// write_node
//////////////////////////////
void Parser::write_node(ostream& out, DOMNode* toWrite)
{
  // Get the name and value out for convenience
  const char *nodeName = XMLString::transcode(toWrite->getNodeName());
  const char *nodeValue = XMLString::transcode(toWrite->getNodeValue());

  // nodeValue will be sometimes be deleted in outputContent, but
  // will not always call outputContent
  bool valueDeleted = false;

  switch (toWrite->getNodeType()) {
  case DOMNode::TEXT_NODE:
    {
      output_content(out, nodeValue);
      valueDeleted = true;
      break;
    }

  case DOMNode::PROCESSING_INSTRUCTION_NODE :
    {
      out  << "<?"
	      << nodeName
	      << ' '
	      << nodeValue
	      << "?>";
      break;
    }

  case DOMNode::DOCUMENT_NODE :
    {
      // Bug here:  we need to find a way to get the encoding name
      //   for the default code page on the system where the
      //   program is running, and plug that in for the encoding
      //   name.
      //MLCh *enc_name = XMLPlatformUtils::fgTransService->getEncodingName();
      out << "<?xml version='1.0' encoding='ISO-8859-1' ?>\n";
      DOMNode *child = toWrite->getFirstChild();
      while(child != 0)
        {
	  write_node(out, child);
          child = child->getNextSibling();
        }

      break;
    }

  case DOMNode::ELEMENT_NODE :
    {
      // Output the element start tag.
      out << '<' << nodeName;

      // Output any attributes on this element
      DOMNamedNodeMap *attributes = toWrite->getAttributes();
      int attrCount = attributes->getLength();
      for (int i = 0; i < attrCount; i++) {
	DOMNode  *attribute = attributes->item(i);
	char* attrName = XMLString::transcode(attribute->getNodeName());
	out  << ' ' << attrName
		<< "=\"";
	//  Note that "<" must be escaped in attribute values.
	output_content(out, XMLString::transcode(attribute->getNodeValue(\
									   )));
	out << '"';
	delete [] attrName;
      }

      //  Test for the presence of children, which includes both
      //  text content and nested elements.
      DOMNode *child = toWrite->getFirstChild();
      if (child != 0) {
	// There are children. Close start-tag, and output children.
	out << '>';
	while(child != 0) {
	  write_node(out, child);
	  child = child->getNextSibling();
	}

	// Done with children.  Output the end tag.
	out << "</" << nodeName << ">";
      } else {
	//  There were no children.  Output the short form close of the
	//  element start tag, making it an empty-element tag.
	out << "/>";
      }
      break;
    }

  case DOMNode::ENTITY_REFERENCE_NODE:
    {
      DOMNode *child;
      for (child = toWrite->getFirstChild(); child != 0;
	   child = child->getNextSibling())
	write_node(out, child);
      break;
    }

  case DOMNode::CDATA_SECTION_NODE:
    {
      out << "<![CDATA[" << nodeValue << "]]>";
      break;
    }

  case DOMNode::COMMENT_NODE:
    {
      out << "<!--" << nodeValue << "-->";
      break;
    }

  default:
    out << "Unrecognized node type = "
	 << (long)toWrite->getNodeType() << endl;
  }

  delete [] nodeName;
  if (!valueDeleted)
    delete [] nodeValue;
}

//////////////////////////////
// output_content
//////////////////////////////
void Parser::output_content(ostream& out,  string chars )
{
  //const char* chars = strdup(to_char_ptr(to_write));
  for (unsigned int index = 0; index < chars.length(); index++) {
    switch (chars[index]) {
    case chAmpersand :
      out << "&amp;";
      break;

    case chOpenAngle :
      out << "&lt;";
      break;

    case chCloseAngle:
      out << "&gt;";
      break;

    case chDoubleQuote :
      out << "&quot;";
      break;

    default:
      // If it is none of the special characters, print it as such
      out << chars[index];
      break;
    }
  }
}
