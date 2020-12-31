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

using namespace std;
using namespace rma;

namespace boosting {

  class REPR : virtual public Boosting {

  public:

    REPR() {}
    REPR(int argc, char** argv) : Boosting(argc, argv) {};
    virtual ~REPR() {}


    // set REPR parameters
    void setBoostingParameters();

    /*************** set initial RMP ***************/
    void setInitRMP();
    void setInitRMPVariables();
    void setInitRMPObjective();
    void setInitRMPColumnBound();
    void setInitRMPRowBound();
    void setConstraintsLHS();
    void printClpElements();
    void setInitRMPClpModel();

    void setDataWts();

    bool isStoppingCondition();

    /************* insert columns ****************/
    void insertExactColumns();
    void insertGreedyColumns();
    void setVecIsCovered(const vector<unsigned int> &a,
                         const vector<unsigned int> &b);
    void setVecIsObjValPos(const unsigned int &k, const bool &isPosObjVal);
    void setMatIntBounds(const unsigned int &k,
                         const vector<unsigned int> &lower,
                         const vector<unsigned int> &upper);
    void insertColumnClpModel(const bool &isPosIncumb);
    void setPebblRMASolutions();

    //////////////////////// Evaluating methods //////////////////////////////

    double evaluateEachIter(const bool& isTest, vector<DataXy> origData);
    double evaluateAtFinal (const bool& isTest, vector<DataXy> origData);

    //////////////////////// Printing methods //////////////////////////////

    void printRMPSolution();	// restricted mater problem solution
    void printRMAInfo();	// print RMA problem info
    //void printEachIterAllErrs() {}

    /************* save a model ****************/
    void saveModel();

  private:

    // parameters for REPR
    unsigned int P;              // the exponent P for the REPR model
    double C, E, D, F;  // coefficients C, E, D, F for the REPR model

  };


} // namespace boosting

#endif
