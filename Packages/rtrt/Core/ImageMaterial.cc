
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>

using namespace rtrt;
using namespace std;

ImageMaterial::ImageMaterial(int, char* texfile, ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode,
			     const Color& ambient, double Kd,
			     const Color& specular, double specpow,
			     double refl)
    : umode(umode), vmode(vmode), ambient(ambient), Kd(Kd), specular(specular),
      specpow(specpow), refl(refl),  transp(0), flip_(false), valid_(false)
{
    read_hdr_image(texfile);
    outcolor=Color(0,0,0);
}

ImageMaterial::ImageMaterial(char* texfile, ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode,
			     const Color& ambient, double Kd,
			     const Color& specular, double specpow,
			     double refl)
    : umode(umode), vmode(vmode), ambient(ambient), Kd(Kd), specular(specular),
      specpow(specpow), refl(refl),  transp(0), flip_(false), valid_(false)
{
    read_image(texfile);
    outcolor=Color(0,0,0);
}

ImageMaterial::ImageMaterial(char* texfile, ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode,
			     const Color& ambient, double Kd,
			     const Color& specular, double specpow,
			     double refl,  double transp)
    : umode(umode), vmode(vmode), ambient(ambient), Kd(Kd), specular(specular),
      specpow(specpow), refl(refl),  transp(transp), flip_(false), 
      valid_(false)
{
    read_image(texfile);
    outcolor=Color(0,0,0);
}

ImageMaterial::~ImageMaterial()
{
}

Color interp_color(Array2<Color>& image,
				  double u, double v)
{
    u *= (image.dim1()-1);
    v *= (image.dim2()-1);
    
    int iu = (int)u;
    int iv = (int)v;

    double tu = u-iu;
    double tv = v-iv;

    Color c = image(iu,iv)*(1-tu)*(1-tv)+
	image(iu+1,iv)*tu*(1-tv)+
	image(iu,iv+1)*(1-tu)*tv+
	image(iu+1,iv+1)*tu*tv;

    return c;
    
}

void ImageMaterial::shade(Color& result, const Ray& ray,
			  const HitInfo& hit, int depth, 
			  double atten, const Color& accumcolor,
			  Context* cx)
{
    UVMapping* map=hit.hit_obj->get_uvmapping();
    UV uv;
    Point hitpos(ray.origin()+ray.direction()*hit.min_t);
    map->uv(uv, hitpos, hit);
    Color diffuse;
    double u=uv.u();
    double v=uv.v();
    switch(umode){
    case None:
	if(u<0 || u>1){
	    diffuse=outcolor;
	    goto skip;
	}
	break;
    case Tile:
	{
	    int iu=(int)u;
	    u-=iu;
	    if (u < 0) u += 1;
	}
        break;
    case Clamp:
	if(u>1)
	    u=1;
	else if(u<0)
	    u=0;
    };
    switch(vmode){
    case None:
	if(v<0 || v>1){
	    diffuse=outcolor;
	    goto skip;
	}
	break;
    case Tile:
	{
	    int iv=(int)v;
	    v-=iv;
	    if (v < 0) v += 1;
	}
        break;
    case Clamp:
	if(v>1)
	    v=1;
	else if(v<0)
	    v=0;
    };
    {
#if 1
      if (flip_)
	diffuse = interp_color(image,u,1-v);
      else
	diffuse = interp_color(image,u,v);
#else
	u*=image.dim1();
	v*=image.dim2();
	int iu=(int)u;
	int iv=(int)v;
        if (flip_)
          iv = image.dim2()-iv;
	diffuse=image(iu, iv);
#endif
    }
skip:
    phongshade(result, ambient, diffuse, specular, specpow, refl,
                ray, hit, depth,  atten,
               accumcolor, cx);
}

static void eat_comments_and_whitespace(ifstream &str)
{
  char c;
  str.get(c);
  for(;;) {
    if (c==' '||c=='\t'||c=='\n') {
      str.get(c);
      continue;
    } else if (c=='#') {
      str.get(c);
      while(c!='\n')
        str.get(c);
    } else {
      str.unget();
      break;
    }
  }
}

void ImageMaterial::read_image(char* filename)
{
  unsigned nu, nv;
  double size;
  ifstream indata(filename);
  unsigned char color[3];
  string token;

  if (!indata.is_open()) {
    cerr << "ImageMaterial: WARNING: I/O fault: no such file: " << filename << endl;
  }
    

  indata >> token; // P6
  eat_comments_and_whitespace(indata);
  indata >> nu >> nv;
  eat_comments_and_whitespace(indata);
  indata >> size;
  eat_comments_and_whitespace(indata);
  image.resize(nu, nv);
  for(unsigned v=0;v<nv;++v){
    for(unsigned u=0;u<nu;++u){
      indata.read((char*)color, 3);
      double r=color[0]/size;
      double g=color[1]/size;
      double b=color[2]/size;
      image(u,v)=Color(r,g,b);
    }
  }

  valid_ = true;
}


void ImageMaterial::read_hdr_image(char* filename)
{
   char buf[200];
   sprintf(buf, "%s.hdr", filename);
   ifstream in(buf);
   if(!in){
     cerr << "Error opening header: " << buf << '\n';
     exit(1);
   }
   int nu, nv;
   in >> nu >> nv;
   if(!in){
     cerr << "Error reading header: " << buf << '\n';
     exit(1);
   }
   ifstream indata(filename);
   image.resize(nu, nv);
   for(int i=0;i<nu;i++){
     for(int j=0;j<nv;j++){
       unsigned char color[3];
       indata.read((char*)color, 3);
       double r=color[0]/255.;
       double g=color[1]/255.;
       double b=color[2]/255.;
       image(i,j)=Color(r,g,b);
     }
   }
   if(!indata){
     cerr << "Error reading image!\n";
     exit(1);
   }
  valid_ = true;
}


