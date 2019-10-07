/*
 *  File name:   base.h
 *  Author:      Ai Kagawa
 *  Description: a header file for LPBase and Data classes
 */

 #ifndef Base_h
 #define Base_h

//#include "gurobi_c++.h"
#include "Time.h"
#include "allParams.h"

#include <pebbl_config.h>
#include <pebbl/utilib/ParameterList.h>
#include <pebbl/utilib/memdebug.h>
#include <pebbl/utilib/seconds.h>
#include <pebbl/utilib/CommonIO.h>
#include <pebbl/bb/pebblParams.h>
#include <pebbl/pbb/parPebblParams.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>


namespace base {

  enum GreedyLevel { EXACT, NotOptimal, Greedy};
  enum OuterInnerCV {INNER, OUTER};
  enum TestTrainData {TRAIN, TEST, VALID};

using namespace utilib;
using namespace std;

struct IntMinMax { double minOrigVal, maxOrigVal; };

struct Feature {vector<IntMinMax> vecIntMinMax;};


/////////////////////////  LPB base class /////////////////////////
class Base :
     public allParams,
     virtual public pebbl::pebblParams,
     virtual public pebbl::parallelPebblParams {
  //virtual public pebbl::pebblParams,
  //virtual public pebbl::parallelPebblParams  {

public:

  Base() : parameters_registered(false), min_num_required_args(0) {
		cout << setprecision(6) << fixed;
	}

	virtual ~Base() {};

	/// Setup the solver parameters using command-line information.
	/// This returns false if there is a problem in the setup, and true
	/// if the setup appeared to work normally.
	bool setup(int& argc, char**& argv);

  // Parameter-related methods
	void write_usage_info(char const* progName,std::ostream& os) const;

  void writeCommandUsage(char const* progName,std::ostream& os) const;

  bool processParameters(int& argc, char**& argv,
         unsigned int min_num_required_args__=0);

  /// Register the parameters into a ParameterList object
	void register_parameters() { plist.register_parameters(*this); }

  /// Check parameters for setup problems and perform debugging I/O
  bool checkParameters(char const* progName = "");

	bool setupProblem(int argc, char** argv) { true; }

	virtual void setName(const char* cname);

	ParameterList plist;

	bool parameters_registered;

	string problemName;

	std::string solver_name;

	unsigned int min_num_required_args;

	Time tc;
	double wallTime;
  double cpuTime;

};


//////////////////// a clsss for integerized dataset ////////////////////
class DataXw {

public:

	DataXw() : w(0.0) {}
	DataXw( const vector<int>& X_ ) :	X(X_) { }

	int read(istream& is) { is >> X >> w; return 0; }
	int write(ostream& os) const { os << X << w;	return 0; }

	vector<int> X;	// explanatory variables
	double w;				// weight of each observation

};


//////////////////// a clsss for original dataset ////////////////////
class DataXy {

public:

	DataXy() {}
	DataXy(  const vector<double>& X_, const int & y_ ) :
		X(X_), y(y_) { }

	int read(istream& is) { is >> X >> y; return 0; }
	int write(ostream& os) const { os << X << " " << y;	return 0; }

	vector<double> X;		// independent variables
	double y;							// dependent variable

};


/////////////////////////// Data class ///////////////////////////
class Data : public Base { //, public Base1 {

friend class CrossValidation;
friend class LPB;

public:

  Data() {}

  bool readData(int argc, char** argv);

	bool readRandObs(int argc, char** argv);

  void setDataDimensions();

  void integerizeData() ;

  void setStandData();

  void setPosNegObs();

  void integerizeFixedLengthData();

  void writeIntObs();
  void writeOrigObs();

	void setXStat();

//protected:

  double avgY, sdY;
	vector<double> avgX, sdX;
  vector<double> minX, maxX;

	int numOrigObs;	  // # of observations in original data
	int numTrainObs;	// # of distinct observation after discretization
  int numTestObs;   // # of testing observations
	int numAttrib;	  // # of attributes
  int numPosTrainObs;
  int numNegTrainObs;

	int numTotalCutPts; // # of cutpoints for RMA
	int maxL;						// maximum distinct value among attributes

  vector<DataXy> origData;			// original datasets X and y

  vector<DataXw> intData;				// discretized data X abd w (weight)

  vector<DataXy> standData;

	vector<int> distFeat;					// distinct features after discretization

	vector<int> vecRandObs;       // contains randomize all observations

	vector<int> vecTrainData;			// contains only training dataset observations

  vector<int> vecTestData;			// contains only training dataset observations

  vector<Feature> vecFeature;		// contains features original and integeried values

};


} // boosting namespace

ostream& operator<<(ostream& os, const deque<bool>& v);
ostream& operator<<(ostream& os, const vector<int>& v);
ostream& operator<<(ostream& os, const vector<double>& v);
ostream& operator<<(ostream& os, const vector<vector<int> >& v);
ostream& operator<<(ostream& os, const vector<vector<double> >& v);

ostream& operator<<(ostream& os, base::DataXw& obj);
istream& operator>>(istream& is, base::DataXw& obj);
ostream& operator<<(ostream& os, base::DataXy& obj);
istream& operator>>(istream& is, base::DataXy& obj);


#endif
