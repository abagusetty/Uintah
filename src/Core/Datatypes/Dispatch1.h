//  This is a generated file, DO NOT EDIT!!

#if !defined(Datatypes_Dispatch1_h)
#define Datatypes_Dispatch1_h

// dispatch1 macro follows
#define dispatch1(field1, callback)\
  bool disp_error = false;\
  string disp_msg;\
  string disp_name = field1->get_type_name(0);\
  if (disp_name == "TetVol") {\
    if (field1->get_type_name(1) == "double") {\
      TetVol<double> *f1 = 0;\
      f1 = dynamic_cast<TetVol<double>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<double>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "float") {\
      TetVol<float> *f1 = 0;\
      f1 = dynamic_cast<TetVol<float>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<float>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "int") {\
      TetVol<int> *f1 = 0;\
      f1 = dynamic_cast<TetVol<int>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<int>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned int") {\
      TetVol<unsigned int> *f1 = 0;\
      f1 = dynamic_cast<TetVol<unsigned int>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<unsigned int>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "char") {\
      TetVol<char> *f1 = 0;\
      f1 = dynamic_cast<TetVol<char>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<char>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned char") {\
      TetVol<unsigned char> *f1 = 0;\
      f1 = dynamic_cast<TetVol<unsigned char>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<unsigned char>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "short") {\
      TetVol<short> *f1 = 0;\
      f1 = dynamic_cast<TetVol<short>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<short>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned short") {\
      TetVol<unsigned short> *f1 = 0;\
      f1 = dynamic_cast<TetVol<unsigned short>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<unsigned short>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "bool") {\
      TetVol<bool> *f1 = 0;\
      f1 = dynamic_cast<TetVol<bool>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<bool>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Vector") {\
      TetVol<Vector> *f1 = 0;\
      f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Tensor") {\
      TetVol<Tensor> *f1 = 0;\
      f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "LatticeVol") {\
    if (field1->get_type_name(1) == "double") {\
      LatticeVol<double> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<double>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<double>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "float") {\
      LatticeVol<float> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<float>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<float>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "int") {\
      LatticeVol<int> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<int>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<int>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned int") {\
      LatticeVol<unsigned int> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<unsigned int>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<unsigned int>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "char") {\
      LatticeVol<char> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<char>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<char>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned char") {\
      LatticeVol<unsigned char> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<unsigned char>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<unsigned char>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "short") {\
      LatticeVol<short> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<short>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<short>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned short") {\
      LatticeVol<unsigned short> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<unsigned short>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<unsigned short>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "bool") {\
      LatticeVol<bool> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<bool>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<bool>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Vector") {\
      LatticeVol<Vector> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Tensor") {\
      LatticeVol<Tensor> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "ContourField") {\
    if (field1->get_type_name(1) == "double") {\
      ContourField<double> *f1 = 0;\
      f1 = dynamic_cast<ContourField<double>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<double>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "float") {\
      ContourField<float> *f1 = 0;\
      f1 = dynamic_cast<ContourField<float>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<float>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "int") {\
      ContourField<int> *f1 = 0;\
      f1 = dynamic_cast<ContourField<int>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<int>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned int") {\
      ContourField<unsigned int> *f1 = 0;\
      f1 = dynamic_cast<ContourField<unsigned int>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<unsigned int>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "char") {\
      ContourField<char> *f1 = 0;\
      f1 = dynamic_cast<ContourField<char>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<char>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned char") {\
      ContourField<unsigned char> *f1 = 0;\
      f1 = dynamic_cast<ContourField<unsigned char>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<unsigned char>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "short") {\
      ContourField<short> *f1 = 0;\
      f1 = dynamic_cast<ContourField<short>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<short>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned short") {\
      ContourField<unsigned short> *f1 = 0;\
      f1 = dynamic_cast<ContourField<unsigned short>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<unsigned short>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "bool") {\
      ContourField<bool> *f1 = 0;\
      f1 = dynamic_cast<ContourField<bool>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<bool>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Vector") {\
      ContourField<Vector> *f1 = 0;\
      f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Tensor") {\
      ContourField<Tensor> *f1 = 0;\
      f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "TriSurf") {\
    if (field1->get_type_name(1) == "double") {\
      TriSurf<double> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<double>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<double>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "float") {\
      TriSurf<float> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<float>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<float>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "int") {\
      TriSurf<int> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<int>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<int>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned int") {\
      TriSurf<unsigned int> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<unsigned int>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<unsigned int>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "char") {\
      TriSurf<char> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<char>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<char>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned char") {\
      TriSurf<unsigned char> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<unsigned char>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<unsigned char>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "short") {\
      TriSurf<short> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<short>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<short>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "unsigned short") {\
      TriSurf<unsigned short> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<unsigned short>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<unsigned short>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "bool") {\
      TriSurf<bool> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<bool>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<bool>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Vector") {\
      TriSurf<Vector> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Tensor") {\
      TriSurf<Tensor> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_error) {\
    cerr << "Error: " << disp_msg << endl;\
  }\

#endif //Datatypes_Dispatch1_h
