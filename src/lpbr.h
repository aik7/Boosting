/*
 *  File name:   lpbr.h
 *  Author:      Ai Kagawa
 *  Description: a header file for LPBR class
 */

#ifndef LPBR_h
#define LPBR_h

//*
#include <pebbl_config.h>
#include <pebbl/utilib/ParameterList.h>
#include <pebbl/utilib/memdebug.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <map>

//#include "gurobi_c++.h"
#include "Time.h"
#include "baseBoost.h"
#include "serRMA.h"
#include "boosting.h"

#ifdef ACRO_HAVE_MPI
#include <pebbl/pbb/parBranching.h>
#include "parRMA.h"
#define outstream ucout
#define IO(action) if (uMPI::iDoIO) { CommonIO::end_tagging(); action; }
#else // ACRO_HAVE_MPI
typedef void parRMA;
#define outstream cout
#define IO(action) action;
#endif // ACRO_HAVE_MPI

namespace boosting {


using namespace utilib;
using namespace std;


struct SimpleRule {
public:
	vector<double> vecLower;
	vector<double> vecUpper;
};


class LPBR : public Boosting {

public:

	LPBR() {}
	LPBR(int argc, char** argv, Data* d) ;
	~LPBR() {}

	void initBoostingData();

	//////////////////////// training data //////////////////////////////
	void trainData(const bool& isOuter, const int& iter, const int& greedyLevel);
	void setInitialMaster();
	void setDataWts();
	void insertColumns();
	void insertGreedyColumns();

	void set1DRules();
	void setOriginal1DRule(const int& j, const int& l, const int& u) ;
	double getLowerBound1D(int j, int value, bool isUpper) ;
	double getUpperBound1D(int j, int value, bool isUpper) ;

	void setSimpleRules();
	bool isCoveredBySR(const int& obs, const int& j);
	void setOriginalRule();
	double getLowerBoundSR(int k, int j, int value, bool isUpper);
	double getUpperBoundSR(int k, int j, int value, bool isUpper);

	void printRMPSolution();	// print restricted mater problem solution
	void printRMAInfo();			// print RMA problem info

	double evaluateEachIter(const int& isTest);
	double evaluateAtFinal(const int& isTest);

	void printEachIterAllErrs();

//private:

	double D;	     // LPBR parameter
	double alpha;  // dual variable alpha
	int P;

	vector<double> unknownRate;
	vector<double> preAdjErr;

	vector<SimpleRule> vecSimpleRule;

	vector<bool> vecSMIsPositive;
	vector<double> vec1DRuleLower;
	vector<double> vec1DRuleUpper;
	vector<vector<bool> > vecCoveredObsBySimpleRule;  // size: m x n

};


} // namespace LPBR


#endif
