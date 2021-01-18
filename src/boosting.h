/*
 *  Author:      Ai Kagawa
 *  Description: a header file for Boosting class
 */

#ifndef Boosting_h
#define Boosting_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <map>
#include <iomanip>
#include <cassert>
#include <ctime>

#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
        
#include <ClpSimplex.hpp>
#include <CoinTime.hpp>
#include <CoinBuild.hpp>
#include <CoinModel.hpp>
#include <CoinPackedMatrix.hpp>
#include <CoinPackedVector.hpp>
#include <CoinHelperFunctions.hpp>

#include "boosting_config.h"

#ifdef HAVE_GUROBI
  #include "gurobi_c++.h"
#endif // HAVE_GUROBI

#include "argBoost.h"
#include "solveRMA.h"

using namespace std;


namespace boosting {

  enum GreedyLevel   {EXACT, GREEDY}; // NotOptimal,
  enum TestTrainData {TEST, TRAIN, VALID};

  class Boosting : public arg::ArgBoost, virtual public rma::SolveRMA {

  public:

    Boosting()
#ifdef HAVE_GUROBI
      : modelGrb(env)
#endif // HAVE_GUROBI
    {};

    Boosting(int& argc, char**& argv);
    virtual ~Boosting() {}

    // reset and resize Boosting variables
    void resetBoosting();

    virtual void setBoostingParameters() = 0;

    //////////////////////// training data //////////////////////////////
    void train(const unsigned int& iter, // const bool& isOuter,
               const unsigned int & greedyLevel);

    void setStandardizedData();

    ////void discretizeData();
    //// virtual void   resetMaster() = 0;
    virtual void setInitRMP() = 0;

    void         solveRMP();

    // solve the restricted master problem,
    // and print out the solution and run time
    void         solveClpRMP();

#ifdef HAVE_GUROBI
    virtual void setGurobiRMP() = 0;
    void         solveGurobiRMP();
    void         resetGurobi();
#endif // HAVE_GUROBI

    virtual void setWeights() = 0;

    virtual bool isStoppingCondition() = 0;

    void         setStoppingCondition();

    // insert column
    virtual void insertColumns() = 0;

    // set vecIsObjValPos,
    // a vector of whether or not each solution, k, is positive
    void setVecIsObjValPos(const unsigned int &k, const bool &isPosObjVal);

    // set matIntLower and matIntUpper for k-th box, and lower and upper boudns
    void setMatIntBounds(const unsigned int &k,
                         const vector<unsigned int> &lower,
                         const vector<unsigned int> &upper);

    // set matIsCvdObsByBox, a matrix of each observation is covered by k-th box
    void setMatIsCvdObsByBox(const unsigned int &k);

    void setMatIsCvdObsByBox(const int &k, const bool &isTrain,
                             const vector<DataXy> &origData,
                             deque<deque<bool> > &matIsCvdObsByBox);

    void setMatIsCvdObsByBoxTestPerIter();

    // set original lower and upper bounds
    // using integerized lower and upper bounds
    void   setOriginalBounds();

    // set PEBBL RMA solutions
    void setPebblRMASolutions();

    // resize vecERMAObjVal and vecGRMAObjVal
    void   resetVecRMAObjVals();

    // set vecERMAObjVal and vecGRMAObjVal for current iteration
    void   setVecRMAObjVals();

    /************************ Evaluating methods ************************/

    // evaluate the model for training and testing data
    void evaluateModel();

    // evaluate the error in each Boosting iteration
    virtual double evaluate(const bool& isTrain, const vector<DataXy> &origData,
                            const deque<deque<bool> > &matIsCvdObsByBox)= 0;

    /************************ Checking methods ************************/
    bool checkDuplicateBoxes(vector<unsigned int> vecIntLower,
                             vector<unsigned int> vecIntUpper);

    void checkObjValue(vector<DataXw> intData,
                       vector<unsigned int> lower,
                       vector<unsigned int> upper);

    // void checkObjValue(const unsigned int &k, vector<DataXw> intData);	// double-check objevtive value for (a, b)

    /************************ Printing functions ************************/

    // print RMP objectiva value and CPU run time
    void printRMPSolution();

    // print curret iteration, testing and testing errors
    void printBoostingErr();

    // print restricted mater problem solution
    virtual void printRMPCheckInfo() = 0;

    // print pritinc problem, RMA, info
    virtual void printRMAInfo()     = 0;
    //virtual void printEachIterAllErrs() = 0;

    /************************ Saving functions ************************/

    // save the train and/or test MSE in each boosting itearation
    void    saveErrors();

    // save actual and predicted response value
    void    savePredictions(const bool &isTrain, vector<DataXy> origData);

    // save the greedy and/or PEBBL RMA objective values
    // in each boosting itearation
    void    saveGERMAObjVals();

    // save weights of all observations for each boosting iteration
    void    saveWeights(const unsigned int &curIter);

    virtual void saveModel() = 0;

    GreedyLevel greedyLevel;

  protected:

    // // return the index for the training observation
    // inline unsigned int idxTrain(const unsigned int &i) {
    //   return data->vecTrainObsIdx[i];
    // }
    //
    // // return the index for the testing observation
    // inline unsigned int idxTest(const unsigned int &i) {
    //   return data->vecTestObsIdx[i];
    // }

    ///////////////////// Boosting variables /////////////////////

    // unsigned int  numIter;  // # of iterations
    unsigned int  curIter;    // the current iteration number

    unsigned int  numRows;    // # of constraints / rows in the mater problem
    unsigned int  numCols;    // # of variables / columns in the mater problem

    unsigned int  numBoxesSoFar;  // # of total boxes entered so far
    unsigned int  numBoxesIter;   // # of boxes entered in the current interaction

    int isStopCond;  // stopping condition

    ///////////////////// CLP variables /////////////////////
    ClpSimplex       modelClp;   // CLP model
    CoinPackedMatrix *matrix;    // CLP matrix
    ClpPackedMatrix  *clpMatrix; // CLP matrix
    // CoinPackedVector row;

    // indices for columns
    int              *colIndex;

    // coefficients of objective variables
    double           *objective;

    // lower and upper bounds of columns variables
    double           *columnLower, *columnUpper;

    // lower and upper bounds of row variables
    double           *rowLower,    *rowUpper;

    // # of LHS constraints variables
    unsigned long long int numElements;

    // coefficients of LHS constraints variables
    double       *elements;

    // stores row index for each element
    int          *rows;

    // starting indices of each column
    CoinBigIndex *starts;

    // # of rows in each column
    int          *lengths;

    // column to insert
    double       *columnInsert;

    /*************************** Gurobi variables ******************/
#ifdef HAVE_GUROBI
    GRBEnv      env;       // Gurobi environment
    GRBModel    modelGrb;  // Gurobi model
    GRBLinExpr  lhs;       // Gurobi linear variables
    GRBConstr*  constr;    // Gurobi constraints
    GRBVar*     vars;      // Gurobi variables
    GRBColumn   col;
    GRBQuadExpr obj;
#endif // HAVE_GUROBI

    // store solution infomation for the master problem
    double *vecPrimalVars;     // dual variables
    double *vecDualVars;       // primal variables
    double primalObjVal;       // primal solution value

    ///////////////////// For RMA /////////////////////

    BasicArray<pebbl::solution*>       s;   // PEBBL solution objects
    BasicArray<pebblRMA::rmaSolution*> sl;  // PEBBL RMA solution objects

    // store lower and upper bound of rules (boxes) in integer and original values
    vector<vector<unsigned int> >    matIntLower;
    vector<vector<unsigned int> >    matIntUpper;
    vector<vector<double> >          matOrigLower;
    vector<vector<double> >          matOrigUpper;

    ///////////////////// For Evaluation /////////////////////

    // whether or not each box is positive or negative box variable
    // (size: # of boxes inserted)
    deque<bool>             vecIsObjValPos;

    // whether or not each train or test observation is covered by each box
    // ( size: [# of train or test obs] * [# of boxes found so far] )
    deque<deque<bool> >     matIsCvdObsByBoxTrain;
    deque<deque<bool> >     matIsCvdObsByBoxTest;

    // errors for the train and test data
    double                  errTrain, errTest;

    // vectors of Greery RMA and PEBBL RMA solutions for all iterations
    // (size: # of Boosting iterations)
    vector<double>          vecErrTrain;
    vector<double>          vecErrTest;

    // predictions of the training and test data by this boosting model
    // (size: # of train or test observations)
    vector<double>          vecPredTrain;
    vector<double>          vecPredTest;

    // vectors of Greery RMA and PEBBL RMA solutions for all iterations
    // (size: # of Boosting iterations)
    vector<double>          vecERMAObjVal;
    vector<double>          vecGRMAObjVal;

    // TODO:: put this function somewhere else
    string getDateTime() {

      time_t rawtime;
      struct tm * timeinfo;
      char buffer[80];

      time (&rawtime);
      timeinfo = localtime(&rawtime);

      strftime(buffer,sizeof(buffer),"%m%d%Y%H%M",timeinfo);
      string str(buffer);

      // cout << str;
      return str;

    } // end getDataTime function

  }; // end Boosting class

} // namespace boosting

#endif  // Boosting_h
