/*
 *  File name:   repr.h
 *  Author:      Ai Kagawa
 *  Description: a header file for REPR class
 */

#ifndef REPR_h
#define REPR_h

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

#include "gurobi_c++.h"
#include "Time.h"
#include "base.h"
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


class REPR : public Boosting {

public:

	REPR() {}
	REPR(int argc, char** argv, Data* d) ;
	~REPR() {}

	void initBoostingData();

	//////////////////////// training data //////////////////////////////
	void trainData(const bool& isOuter, const int& iter, const int& greedyLevel);
	void setInitialMaster();
	//void solveInitialMaster();
	void setDataWts();
	void insertColumns(); //const int& GreedyLevel
	void insertGreedyColumns();

	void printRMPSolution();	// restricted mater problem solution
	void printRMAInfo();			// print RMA problem info

	double evaluateEachIter(const int& isTest);
	double evaluateAtFinal(const int& isTest);

	void printEachIterAllErrs() {}

//private:

	// parameters for REPR
	int P;
	double C, E;
	double D, F;

	//GRBQuadExpr obj; //  GUROBI variables

};


} // namespace boosting

#endif
