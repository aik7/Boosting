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

#include "Time.h"
#include "boosting.h"


namespace boosting {

  class REPR : virtual public Boosting {

  public:

    REPR() {}
    REPR(int argc, char** argv) : Boosting(argc, argv) {};
    virtual ~REPR() {}

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
