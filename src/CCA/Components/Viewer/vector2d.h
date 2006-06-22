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
 * vector2d.h
 *
 * Written by:
 *  Keming Zhang
 *  Department of Computer Science
 *  University of Utah
 *  June 2002
 *
 * vector2d is a class for 2D vector
 */

#ifndef VECTOR2D_H
#define VECTOR2D_H

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

class vector2d  {
public:
  vector2d() : x(0), y(0) {}
  vector2d(double e0, double e1) : x(e0), y(e1) {}

  vector2d(const vector2d &v) {
    x = v.x;
    y = v.y;
  }

  const vector2d& operator+() const { return *this; }
  vector2d operator-() const { return vector2d(-x, -y); }

  vector2d& operator+=(const vector2d &v2);
  vector2d& operator-=(const vector2d &v2);
  vector2d& operator*=(const double t);
  vector2d& operator/=(const double t);

  // inner product
  inline double operator%(const vector2d &v) { return (x * v.x) + (y * v.y); }

  double length() const { return sqrt((x * x) + (y * y)); }
  double squaredLength() const { return (x * x) + (y * y); }

  void normalize();

private:
  double x, y;
};

inline void vector2d::normalize() {
  //double k = 1.0 / sqrt(x*x + y*y);
  double k = 1.0 / length();
  x *= k; y *= k;
}

//////////////////////////////////////////////////////////////////////////
// vector2d operators

inline vector2d& vector2d::operator+=(const vector2d &v){
  x  += v.x;
  y  += v.y;
  return *this;
}

inline vector2d& vector2d::operator-=(const vector2d& v) {
  x  -= v.x;
  y  -= v.y;
  return *this;
}

inline vector2d& vector2d::operator*=(const double t) {
  x  *= t;
  y  *= t;
  return *this;
}

inline vector2d& vector2d::operator/=(const double t) {
  x  /= t;
  y  /= t;
  return *this;
}

//////////////////////////////////////////////////////////////////////////
// convenience functions that use vector2d

inline bool operator==(const vector2d &t1, const vector2d &t2) {
  return ((t1.x == t2.x) && (t1.y == t2.y));
}

inline bool operator!=(const vector2d &t1, const vector2d &t2) {
  return ((t1.x != t2.x) || (t1.y != t2.y));
}

inline std::istream &operator>>(std::istream &is, vector2d &t) {
  is >> t.x >> t.y;
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const vector2d &t) {
  os << t.x << " " << t.y;
  return os;
}

inline vector2d unit_vector(const vector2d& v) {
  double k = 1.0 / sqrt(v.x * v.x + v.y * v.y);
  return vector2d(v.x*k, v.y*k);
}

inline vector2d operator+(const vector2d &v1, const vector2d &v2) {
  return vector2d( v1.x + v2.x, v1.y + v2.y);
}

inline vector2d operator-(const vector2d &v1, const vector2d &v2) {
  return vector2d( v1.x - v2.x, v1.y - v2.y);
}

inline vector2d operator*(double t, const vector2d &v) {
  return vector2d(t * v.x, t * v.y);
}

inline vector2d operator*(const vector2d &v, double t) {
  return vector2d(t * v.x, t * v.y);
}


inline vector2d operator*(const vector2d &v1, const vector2d &v2) {
  return vector2d(v1.x * v2.x, v1.y * v2.y);
}


inline vector2d operator/(const vector2d &v, double t) {
  return vector2d(v.x / t, v.y / t);
}

// inner product
inline double operator%(const vector2d &v1, const vector2d &v2) {
  return v1.x * v2.x + v1.y * v2.y ;
}


#endif
