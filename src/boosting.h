/*
 *  File name:   boosting_h
 *  Author:      Ai Kagawa
 *  Description: a header file for Boosting class
 */

#ifndef Boosting1_h
#define Boosting1_h

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

#include <pebbl_config.h>
#include <pebbl/utilib/ParameterList.h>
#include <pebbl/utilib/memdebug.h>
#include <pebbl/utilib/CommonIO.h>

#ifdef ACRO_HAVE_MPI
#include <pebbl/pbb/parBranching.h>
#include "parRMA.h"
#define outstream ucout
// #define IO(action) if (uMPI::iDoIO) { CommonIO::end_tagging(); action; }
#else // ACRO_HAVE_MPI
typedef void parRMA;
#define outstream cout
#define IO(action) action;
#endif //

#include "Time.h"
#include "argRMA.h"
#include "argBoost.h"
#include "dataRMA.h"
// #include "dataBoost.h"
#include "baseRMA.h"
#include "serRMA.h"
#include "greedyRMA.h"
#include "driverRMA.h"

namespace boosting {

  enum GreedyLevel   {EXACT, NotOptimal, Greedy};
  enum TestTrainData {TRAIN, TEST, VALID};
  //enum OuterInnerCV  {INNER, OUTER};

  class Boosting : public arg::ArgBoost, virtual public rma::DriverRMA {

  public:

    Boosting() {};
    Boosting(int& argc, char**& argv);
    virtual ~Boosting() {
#ifdef ACRO_HAVE_MPI
      if (isParallel) { CommonIO::end(); uMPI::done(); }
#endif // ACRO_HAVE_MPI
    }

    void setupBoosting(int& argc, char**& argv);

    void         reset();

    virtual void setBoostingParameters() = 0;

    //////////////////////// training data //////////////////////////////
    void train(const bool& isOuter, const int& iter, const int & greedyLevel);

    void saveModel();

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

    /////////////////void discretizeData();
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

    void         writeGERMA();

    //////////////////////// Evaluating methods /////////////////////////////

    void setCoveredTrainObs();
    void setCoveredTestObs();

    void evaluateEach();
    void evaluateFinal();

    virtual double evaluateEachIter(const int& isTest, vector<DataXy> origData) = 0;
    virtual double evaluateAtFinal(const int& isTest, vector<DataXy> origData)  = 0;

    void    writeWts(const int& curIter);
    void    writePredictions(const int& isTest, vector<DataXy> origData); // write predictions

    //////////////////////// Checking methods ///////////////////////
    bool isDuplicate();
    void checkObjValue(vector<DataXw> intData);
    void checkObjValue(int k, vector<DataXw> intData);	// double-check objevtive value for (a, b)

    GreedyLevel greedyLevel;

  protected:

    ///////////////////// Boosting variables /////////////////////

    int  NumIter;	  // # of iterations, observation, features, and variables
    int  curIter;		// the current iteration number

    int  numRows;    // # of constraints / rows in the mater problem
    int  numCols;    // # of variables / columns in the mater problem
    int  numBox;     // # of total boxes entered so far
    unsigned int  numRMASols; // # of boxes entered in the current interaction

    int  NumObs;
    int  NumAttrib;

    bool flagDuplicate;
    bool isOuter;

    ///////////////////// CLP variables /////////////////////
    ClpSimplex       model;
    CoinPackedMatrix *matrix;
    CoinPackedVector row;

    double *dataWts;
    double *objValue;
    double *lowerColumn, *upperColumn;
    double *lowerRow,    *upperRow;
    int    *colIndex,    *rowIndex;

    // store solution infomation for the master problem
    double *vecPrimalVars;  // dual variables
    double *vecDualVars;    // primal variables
    double primalSol;	      // primal solution value
    double *columnObjective;

    deque<bool>    vecIsCovered;	// each observation is covered or not

    ///////////////////// For RMA /////////////////////

    BasicArray<pebbl::solution*>       s;
    BasicArray<pebblRMA::rmaSolution*> sl;

    // store lower and upper bound of rules (boxes)
    vector<vector<unsigned int> >    matIntLower;
    vector<vector<unsigned int> >    matIntUpper;
    vector<vector<double> >          matOrigLower;
    vector<vector<double> >          matOrigUpper;

    ///////////////////// For Evaluation /////////////////////

    vector<bool>            vecCoveredSign;     // size: m (originalObs) x |K'|
    vector<vector<bool> >   vecCoveredObsByBox; // size: m (originalObs) x |K'|

    vector<double>          predTrain;          // predictions of training data by model
    vector<double>          predTest;           // predictions of testing data by model

    vector<double>          vecERMA;
    vector<double>          vecGRMA;

    double errTrain;
    double errTest;

    Time   tc;


  };

} // namespace boosting


#endif
