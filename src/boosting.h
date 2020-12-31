/*
 *  File name:   boosting_h
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

namespace boosting {

  enum GreedyLevel   {EXACT, NotOptimal, Greedy};
  enum TestTrainData {TRAIN, TEST, VALID};
  //enum OuterInnerCV  {INNER, OUTER};

  class Boosting : public arg::ArgBoost, virtual public rma::SolveRMA {

  public:

    Boosting() {};
    Boosting(int& argc, char**& argv);
    virtual ~Boosting() {}

    // set up to run Boosting algorithm
    void setupBoosting(int& argc, char**& argv);

    void reset();

    virtual void setBoostingParameters() = 0;

    //////////////////////// training data //////////////////////////////
    void train(const bool& isOuter, const unsigned int& iter,
               const unsigned int & greedyLevel);

    ////void discretizeData();
    //// virtual void   resetMaster() = 0;
    virtual void setInitRMP() = 0;
    void         solveRMP();
    virtual void setDataWts() = 0;

    virtual bool isStoppingCondition() = 0;
    void         insertColumns();
    virtual void insertExactColumns()  = 0;
    virtual void insertGreedyColumns() = 0;

    void   setOriginalBounds();
    double getLowerBound(int k, int j, int value, bool isUpper) ;
    double getUpperBound(int k, int j, int value, bool isUpper) ;

    //////////////////////// Printing methods /////////////////////////////

    virtual void printRMPSolution() = 0;  		// restricted mater problem solution
    virtual void printRMAInfo()     = 0;		  // pritinc problem, RMA
    //virtual void printEachIterAllErrs() = 0;

    // void         printRMASolutionTime();

    void         printIterInfo();
    void         printBoostingErr();
    void         printCLPsolution();

    void         saveGERMAObjVals();

    //////////////////////// Evaluating methods /////////////////////////////

    void setCoveredTrainObs();
    void setCoveredTestObs();

    void evaluateEach();  // evaluate each iteration
    void evaluateFinal(); // evaluate in the end

    virtual double evaluateEachIter(const bool &isTest,
                                    vector<DataXy> origData) = 0;
    virtual double evaluateAtFinal(const bool &isTest,
                                   vector<DataXy> origData)  = 0;

    void    saveWts(const unsigned int &curIter);
    void    savePredictions(const bool &isTest, vector<DataXy> origData); // write predictions

    virtual void saveModel() = 0;  // save model

    //////////////////////// Checking methods ///////////////////////
    bool isDuplicate();
    void checkObjValue(vector<DataXw> intData);
    void checkObjValue(const unsigned int &k, vector<DataXw> intData);	// double-check objevtive value for (a, b)

    GreedyLevel greedyLevel;

    // TODO:: put this function somewhere else
    std::string getDateTime(){

      time_t rawtime;
      struct tm * timeinfo;
      char buffer[80];

      time (&rawtime);
      timeinfo = localtime(&rawtime);

      strftime(buffer,sizeof(buffer),"%m%d%Y%H%M",timeinfo);
      std::string str(buffer);

      // std::cout << str;
      return str;

    }

  protected:

    inline unsigned int idxTrain(const unsigned int &i) {
      return data->vecTrainObsIdx[i];
    };

    ///////////////////// Boosting variables /////////////////////

    unsigned int  numObs;     // # of observation
    unsigned int  numAttrib;  // # of attribute

    // unsigned int  numIter;    // # of iterations, observation, features, and variables
    unsigned int  curIter;    // the current iteration number

    unsigned int  numRows;    // # of constraints / rows in the mater problem
    unsigned int  numCols;    // # of variables / columns in the mater problem

    unsigned int  numBoxesSoFar;  // # of total boxes entered so far
    unsigned int  numBoxesIter;   // # of boxes entered in the current interaction

    bool isOuter;     // whether or not it is the outer iteration of the cross validation

    ///////////////////// CLP variables /////////////////////
    ClpSimplex       model;   // CLP model
    CoinPackedMatrix *matrix;
    CoinPackedVector row;

    double *objective;
    double *columnLower, *columnUpper;
    double *rowLower,     *rowUpper;
    int *colIndex, *rowIndex;  // indices for columns and rows
    unsigned long long int numElements;
    double *elements;
    CoinBigIndex *starts;
    int *rows;
    int *lengths;

    // store solution infomation for the master problem
    double *vecPrimalVars;    // dual variables
    double *vecDualVars;      // primal variables
    double primalSol;	      // primal solution value
    double *columnObjective;

    deque<bool>  vecIsCovered;	// whether or not each observation is covered

    ///////////////////// For RMA /////////////////////

    BasicArray<pebbl::solution*>       s;   // PEBBL solution objects
    BasicArray<pebblRMA::rmaSolution*> sl;  // PEBBL RMA solution objects

    // store lower and upper bound of rules (boxes)
    vector<vector<unsigned int> >    matIntLower;
    vector<vector<unsigned int> >    matIntUpper;
    vector<vector<double> >          matOrigLower;
    vector<vector<double> >          matOrigUpper;

    ///////////////////// For Evaluation /////////////////////

    vector<bool>            vecIsObjValPos;     // size: m (originalObs) x |K'|
    vector<vector<bool> >   vecCoveredObsByBox; // size: m (originalObs) x |K'|

    vector<double>          predTrain;          // predictions of training data by model
    vector<double>          predTest;           // predictions of testing data by model

    // a vector of PEBBL RMA solutions for all iterations
    vector<double>          vecERMAObjVal;

    // a vector of Greery RMA solutions for all iterations
    vector<double>          vecGRMAObjVal;

    double errTrain; // error for the train data
    double errTest;  // error for the test data

  }; // end Boosting class

} // namespace boosting


#endif
