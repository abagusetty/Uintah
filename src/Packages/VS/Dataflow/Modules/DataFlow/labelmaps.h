/******************************************************************************
 * File: labelmaps.h
 *
 * Description: C header source class definitions to provide an API for
 *		Visible Human segmentation information:
 *		* The Master Anatomy Label Map
 *		* Spatial Adjacency relations for the anatomical structures
 *		and
 *		* Bounding Boxes for each anatomical entity.
 *
 * Author: Stewart Dickson <mailto:dicksonsp@ornl.gov>
 *	   <http://www.csm.ornl.gov/~dickson>
 ******************************************************************************/

#ifndef LABELMAPS_H
#define LABELMAPS_H

#include <vector>

using namespace std;

#define VH_LM_NUM_NAMES 512

/* misc string manip functions */
char *space_to_underbar(char *dst, char *src);

char *capitalize(char *dst, char *src);

/******************************************************************************
 * class VH_MasterAnatomy
 *
 * description: Two parallel arrays -- names <-> indices
 *
 *		The map from anatomical volume (e.g., C-T, MRI) tags to
 *		anatomical names.
 ******************************************************************************/

class VH_MasterAnatomy {
public:
	VH_MasterAnatomy();
	~VH_MasterAnatomy();
	void readFile(char *infilename);
	void readFile(FILE *infileptr);
	char *get_anatomyname(int labelindex);
	int get_labelindex(char *anatomyname);
	int get_num_names() { return num_names; };
private:
	char **anatomyname;
	int *labelindex;
	int num_names;
};

/******************************************************************************
 * class VH_AdjacencyMapping
 *
 * description: An array of integers for each anatomical entity -- the
 *		relation describes every entity spatially adjacent to the
 *		indexed entity.
 ******************************************************************************/

#define VH_FILE_MAXLINE 2048

class VH_AdjacencyMapping {
public:
	VH_AdjacencyMapping();
	~VH_AdjacencyMapping();
	void readFile(char *infilename);
        void readFile(FILE *infileptr);
	int get_num_rel(int index);
	int get_num_names() { return num_names; };
	int *adjacent_to(int index);
private:
	int **rellist;
	int *numrel;
	int num_names;
};

/******************************************************************************
 * class VH_AnatomyBoundingBox
 *
 * description: A doubly-linked list of nodes consisting of an ASCII
 *		char *name -- matching a tissue in the MasterAnatomy
 *              and the X-Y-Z extrema of the segmentation of that
 *              tissue.  Note: dimensions are integer Voxel addresses
 *		referring to the original segmented volume.
 ******************************************************************************/
class VH_AnatomyBoundingBox {
private:
	char *anatomyname_;
	int minX_, maxX_, minY_, maxY_, minZ_, maxZ_, minSlice_, maxSlice_;
	VH_AnatomyBoundingBox *blink, *flink;
public:
        VH_AnatomyBoundingBox() { flink = blink = this; };
	void append(VH_AnatomyBoundingBox *newNode);
        VH_AnatomyBoundingBox * next() { return flink; };
        void readFile(FILE *infileptr);
	char *get_anatomyname() { return(anatomyname_); };
	void set_anatomyname(char *newName) { anatomyname_ = newName; };
	int get_minX() { return minX_; };
        void set_minX(int new_minX) { minX_ = new_minX; };
        int get_maxX() { return maxX_; };
        void set_maxX(int new_maxX) { maxX_ = new_maxX; };
        int get_minY() { return minY_; };
        void set_minY(int new_minY) { minY_ = new_minY; };
        int get_maxY() { return maxY_; };
        void set_maxY(int new_maxY) { maxY_ = new_maxY; };
        int get_minZ() { return minZ_; };
        void set_minZ(int new_minZ) { minZ_ = new_minZ; };
        int get_maxZ() { return maxZ_; };
        void set_maxZ(int new_maxZ) { maxZ_ = new_maxZ; };
        int get_minSlice() { return minSlice_; };
        void set_minSlice(int newMinSlice) { minSlice_ = newMinSlice; };
        int get_maxSlice() { return maxSlice_; };
        void set_maxSlice(int newMaxSlice) { maxSlice_ = newMaxSlice; };
};

/******************************************************************************
 * Read an ASCII AnatomyBoundingBox file into a linked list
 ******************************************************************************/
VH_AnatomyBoundingBox *
VH_Anatomy_readBoundingBox_File(char *infilename);

/******************************************************************************
 * Find the boundingBox of a named anatomical entity
 ******************************************************************************/
VH_AnatomyBoundingBox *
VH_Anatomy_findBoundingBox(VH_AnatomyBoundingBox *list, char *anatomyname);

/******************************************************************************
 * Find the largest bounding volume of the segmentation
 ******************************************************************************/
VH_AnatomyBoundingBox *
VH_Anatomy_findMaxBoundingBox(VH_AnatomyBoundingBox *list);

/******************************************************************************
 * class VH_injuryList
 *
 * description: A doubly-linked list of nodes consisting of nodes containing
 *              the name of an injured tissue and iconic geometry to display
 *              to indicate the extent of the injury.
 ******************************************************************************/

class VH_injury
{
  public:
  string anatomyname;
  int geom_type; // sphere, cylinder, hollow cylinder
  float axisX0, axisY0, axisZ0; // center axis endpoint 0
  float axisX1, axisY1, axisZ1; // center axis endpoint 1
  float rad0, rad1;

  VH_injury() { };
  VH_injury(char *newName) { anatomyname = string(newName); };
};

bool
is_injured(char *targetName, vector<VH_injury> &injured_tissue_list);

#endif
