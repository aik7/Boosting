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

#include "boosting.h"
#include "utility.h"

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

    void setWeights(int interaction);   // set data weights in DataRMA class

    bool isStoppingCondition();    // whether or not stopping condition

    /************************* insert columns **************************/
    // insert columns using PEBBL or Greedy RMA solutions
    // The argument specifies an interaction, -1 meaning no interaction
    void insertColumns(int interaction = -1);

    // insert columns using Greedy RMA solution
    void insertEachColumn(const int & k, const bool &isPosObjVal,
                          const vector<unsigned int> &vecLower,
                          const vector<unsigned int> &vecUpper,
                          int interaction);

    // insert a column in the CLP model using k-th RMA solution
    void insertColumnClpModel(const unsigned int &k, int interaction = -1);

#ifdef HAVE_GUROBI
    void setGurobiRMP();
    void insertColumnGurobiModel(const unsigned int &k, int interaction = -1);
#endif
    /************************* Evaluating methods **************************/

    // evaluate the current model each observation
    double evaluate(const bool& isTrain, const vector<DataXy> &origData,
                    const deque<deque<bool> > &matIsCvdObsByBox);

    /************************* Printing methods **************************/

    void printRMPCheckInfo();  // restricted mater problem solution
    void printRMAInfo();       // print RMA problem info

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
