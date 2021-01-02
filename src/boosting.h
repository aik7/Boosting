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

#include <ClpSimplex.hpp>
#include <CoinTime.hpp>
#include <CoinBuild.hpp>
#include <CoinModel.hpp>
#include <CoinPackedMatrix.hpp>
#include <CoinPackedVector.hpp>
#include <CoinHelperFunctions.hpp>

#include "Time.h"
#include "argRMA.h"
#include "argBoost.h"
#include "dataRMA.h"
#include "baseRMA.h"
#include "serRMA.h"
#include "greedyRMA.h"
#include "solveRMA.h"

using namespace std;


namespace boosting {

  enum GreedyLevel   {EXACT, Greedy}; // NotOptimal,
  enum TestTrainData {TRAIN, TEST, VALID};

  class Boosting : public arg::ArgBoost, virtual public rma::SolveRMA {

  public:

    Boosting() {};
    Boosting(int& argc, char**& argv);
    virtual ~Boosting() {}

    // set up to run Boosting algorithm
    void setupBoosting(int& argc, char**& argv);

    // reset and resize Boosting variables
    void resetBoosting();

    virtual void setBoostingParameters() = 0;

    //////////////////////// training data //////////////////////////////
    void train(const unsigned int& iter, // const bool& isOuter,
               const unsigned int & greedyLevel);

    ////void discretizeData();
    //// virtual void   resetMaster() = 0;
    virtual void setInitRMP() = 0;
    void         solveRMP();
    virtual void setDataWts() = 0;

    virtual bool isStoppingCondition() = 0;
    void         insertColumns();
    virtual void insertPebblColumns()  = 0;
    virtual void insertGreedyColumns() = 0;

    void   setOriginalBounds();

    void   resetVecRMAObjVals();
    void   setVecRMAObjVals();

    // set PEBBL RMA solutions
    void setPebblRMASolutions();

    //////////////////////// Evaluating methods /////////////////////////////

    // evaluate the error in each Boosting iteration
    virtual double evaluateEachIter(const bool &isTest,
                                    vector<DataXy> origData) = 0;

    // evaluate the error in the end of Boosting iterations
    virtual double evaluateAtFinal(const bool &isTest,
                                   vector<DataXy> origData)  = 0;

    void    saveWts(const unsigned int &curIter);
    void    savePredictions(const bool &isTest, vector<DataXy> origData); // write predictions

    /************************ Checking methods ************************/
    bool checkDuplicateBoxes(vector<unsigned int> vecIntLower,
                             vector<unsigned int> vecIntUpper);

    void checkObjValue(vector<DataXw> intData);

    void checkObjValue(const unsigned int &k, vector<DataXw> intData);	// double-check objevtive value for (a, b)

    /************************ Printing functions ************************/
    void printCLPsolution();
    void printBoostingErr();

    //////////////////////// Printing methods /////////////////////////////

    virtual void printRMPSolution() = 0;  		// restricted mater problem solution
    virtual void printRMAInfo()     = 0;		  // pritinc problem, RMA
    //virtual void printEachIterAllErrs() = 0;

    /************************ Saving functions ************************/

    void saveGERMAObjVals();

    virtual void saveModel() = 0;

    GreedyLevel greedyLevel;

  protected:

    inline unsigned int idxTrain(const unsigned int &i) {
      return data->vecTrainObsIdx[i];
    };

    inline unsigned int idxTest(const unsigned int &i) {
      return data->vecTestObsIdx[i];
    };

    ///////////////////// Boosting variables /////////////////////

    unsigned int  numObs;     // # of observation
    unsigned int  numAttrib;  // # of attribute

    // unsigned int  numIter;  // # of iterations
    unsigned int  curIter;    // the current iteration number

    unsigned int  numRows;    // # of constraints / rows in the mater problem
    unsigned int  numCols;    // # of variables / columns in the mater problem

    unsigned int  numBoxesSoFar;  // # of total boxes entered so far
    unsigned int  numBoxesIter;   // # of boxes entered in the current interaction

    ///////////////////// CLP variables /////////////////////
    ClpSimplex       model;   // CLP model
    CoinPackedMatrix *matrix; // CLP matrix
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

    // vector<double> vecPrimalVars;
    // vector<double> vecDualVars;

    // store solution infomation for the master problem
    double *vecPrimalVars;     // dual variables
    double *vecDualVars;       // primal variables
    double primalSol;	        // primal solution value

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

    // whether or not each observation is covered by each box
    // ( size: [# of observations] * [# of boxes found so far] )
    vector<vector<bool> >   matIsCvdObsByBox;

    // errors for the train and test data
    double                  errTrain, errTest;

    // a vector of PEBBL RMA solutions for all iterations
    // (size: # of Boosting iterations)
    vector<double>          vecERMAObjVal;

    // a vector of Greery RMA solutions for all iterations
    // (size: # of Boosting iterations)
    vector<double>          vecGRMAObjVal;

    // predictions of training data by the current model
    vector<double>          predTrain;

    // predictions of testing data by model
    vector<double>          predTest;

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


    // double getLowerBound(const unsigned int &k,     const unsigned int &j,
    //                      const unsigned int &value, const bool &isUpper) ;
    // double getUpperBound(const unsigned int &k,     const unsigned int &j,
    //                      const unsigned int &value, const bool &isUpper)


  }; // end Boosting class


} // namespace boosting


#endif  // Boosting_h
