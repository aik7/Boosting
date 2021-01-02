/*
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
    void setInitRMP();             // set initial RMP
    void setInitRMPVariables();    // set variables for initial RMP
    void setClpParameters();       // set the CLP parameters
    void setInitRMPObjective();    // set objective for initial RMP
    void setInitRMPColumnBound();  // set column bounds for initial RMP
    void setInitRMPRowBound();     // set row bounds for initial RMP
    void setConstraintsLHS();      // set LHS constraints, elements
    void setInitRMPClpModel();     // set initial RMP CLP model

    void setDataWts();             // set data weights in DataRMA class

    bool isStoppingCondition();    // whether or not stopping condition

    /************************* insert columns **************************/
    void insertPebblColumns();     // insert columns using PEBBL RMA solution
    void insertGreedyColumns();    // insert columns using Greedy RMA solution

    // void setVecIsCovered(const vector<unsigned int> &a,
    //                      const vector<unsigned int> &b);

    // set vecIsObjValPos,
    // a vector of whether or not each solution, k, is positive
    void setVecIsObjValPos(const unsigned int &k, const bool &isPosObjVal);

    // set matIntLower and matIntUpper for k-th box, and lower and upper boudns
    void setMatIntBounds(const unsigned int &k,
                         const vector<unsigned int> &lower,
                         const vector<unsigned int> &upper);

    // set matIsCvdObsByBox, a matrix of each observation is covered by k-th box
    void setMatIsCvdObsByBox(const unsigned int &k);

    // insert a column in the CLP model using k-th RMA solution
    void insertColumnClpModel(const unsigned int &k);

    /************************* Evaluating methods **************************/

    // evaluate the current model each observation
    double evaluateEachIter(const bool& isTest, vector<DataXy> origData);

    // evaluate the current model in the end of Boosting iteration
    double evaluateAtFinal (const bool& isTest, vector<DataXy> origData);

    /************************* Printing methods **************************/

    void printRMPSolution();  // restricted mater problem solution
    void printRMAInfo();      // print RMA problem info
    //void printEachIterAllErrs() {}
    void printClpElements();  // print CLP elements

    /************* save a model ****************/
    void saveModel();

  private:

    // parameters for REPR
    unsigned int P;              // the exponent P for the REPR model
    double C, E, D, F;  // coefficients C, E, D, F for the REPR model

  };  // end REPR class


} // namespace boosting

#endif  // REPR_h
