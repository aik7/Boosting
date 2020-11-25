/*
 *  File name:   repr.h
 *  Author:      Ai Kagawa
 *  Description: a header file for REPR class
 */

#ifndef REPR_h
#define REPR_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <map>

#include <pebbl_config.h>
#include <pebbl/utilib/ParameterList.h>
#include <pebbl/utilib/memdebug.h>

#ifdef ACRO_HAVE_MPI
#include <pebbl/pbb/parBranching.h>
#include "parRMA.h"
#define outstream ucout
//#define IO(action) if (uMPI::iDoIO) { CommonIO::end_tagging(); action; }
#else // ACRO_HAVE_MPI
typedef void parRMA;
#define outstream cout
#define IO(action) action;
#endif // ACRO_HAVE_MPI

#include "Time.h"
#include "boosting.h"
#include "driverRMA.h"


namespace boosting {

  class REPR : virtual public Boosting {

  public:

    REPR() {}
    REPR(int argc, char** argv) : Boosting(argc, argv) {};
    // REPR() : rma::DriverRMA(), Boosting() {};
    ~REPR() {}

    //////////////////////// Training methods //////////////////////////////

    void setBoostingParameters();

    void setInitRMP();
    void setDataWts();

    bool isStoppingCondition();
    void insertExactColumns();
    void insertGreedyColumns();

    //////////////////////// Evaluating methods //////////////////////////////

    double evaluateEachIter(const int& isTest, vector<DataXy> origData);
    double evaluateAtFinal (const int& isTest, vector<DataXy> origData);

    //////////////////////// Printing methods //////////////////////////////

    void printRMPSolution();	// restricted mater problem solution
    void printRMAInfo();			// print RMA problem info
    //void printEachIterAllErrs() {}

  private:

    // parameters for REPR
    int P;
    double C, E, D, F;

  };


} // namespace boosting

#endif
