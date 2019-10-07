/*
 *  File name:   CrossValid.h
 *  Author:      Ai Kagawa
 *  Description: a header file for CrossValidation classes
 */

#ifndef CrossValid_h
#define CrossValid_h

//#include "gurobi_c++.h"
#include "lpbr.h"
#include "repr.h"
#include "compModel.h"

#ifdef ACRO_HAVE_MPI
#include <pebbl/pbb/parBranching.h>
#define outstream ucout
#define IO(action) if (uMPI::iDoIO) { CommonIO::end_tagging(); action; }
#else // ACRO_HAVE_MPI
typedef void parRMA;
#define outstream cout
#define IO(action) action;
#endif // ACRO_HAVE_MPI

namespace crossvalidation {

using namespace base;
using namespace boosting;
using namespace comparison;

/////////////////  Cross Calidation class /////////////////
class CrossValidation : public Data {

public:

	CrossValidation() {}// base = this; };

	~CrossValidation() {
#ifdef ACRO_HAVE_MPI
		CommonIO::end();
		uMPI::done();
#endif // ACRO_HAVE_MPI
	};

	void setupData(int& argc, char**& argv) {

#ifdef ACRO_HAVE_MPI
    uMPI::init(&argc, &argv, MPI_COMM_WORLD);
#endif // ACRO_HAVE_MPI

		setup(argc, argv);
		readData(argc, argv);
		lpbr = new LPBR(argc, argv, this);
    repr = new REPR(argc, argv, this);
	}

  void runOuterCrossValidation();
	void setParamComb();
	void selectParamters(const int& i);

  void setOuterPartition();
  void setInnerPartition(int i);

	void setTrainNTestData(int i, int remain, int NumObs, int NumEachPart,
			int &NumTrainData, int &NumTestData,
			vector<int> &vecTrainData, vector<int> &vecTestData, vector<int> Data) ;

	void setCurrnetDataSets(const bool &isOuter, const int &j);

  void printOuterScore(); // outer cross validation score
  void printInnerScore(int i);	// inner cross validation score

	void writePredictions(const int& isTest);

  vector<vector<int> > trainDataOut;
	vector<vector<int> >  testDataOut;

	vector<vector<int> > trainDataIn;
	vector<vector<int> >  testDataIn;

  vector<int> NumTrainDataOut;
	vector<int> NumTestDataOut;

	vector<int> NumTrainDataIn;
	vector<int> NumTestDataIn;

  vector<double> vecParams;	   // parameters to select in inner cross-validation
	vector<double> avgInTestErr;
	vector<double> avgInTrainErr;

	int NumComb;
	int bestParamComb;  // best parameter combination

  // # of  outer and inner partitions
  int NumOutPartition;
	int NumInPartition;

  // # of observation in each outer and inner partition
  int NumEachOutPartition;
	int NumEachInPartition;

	double avgOutTrainErr;
	double avgOutTestErr;

	double avgTime;
	double avgTimePerFold;

	LPBR* lpbr;
  REPR* repr;

	compREPR* rcm;
	compLPBR* lcm;

};

} // namespace crossvalidation

#endif
