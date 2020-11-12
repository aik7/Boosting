/*
 *  File name:   boosting_h
 *  Author:      Ai Kagawa
 *  Description: a header file for Boosting class
 */

#ifndef Boosting1_h
#define Boosting1_h

//#include <direct.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <map>
#include <iomanip>
#include <cassert>

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
#define IO(action) if (uMPI::iDoIO) { CommonIO::end_tagging(); action; }
#else // ACRO_HAVE_MPI
typedef void parRMA;
#define outstream cout
#define IO(action) action;
#endif // ACRO_HAVE_MPI

#include "Time.h"
#include "argRMA.h"
#include "dataRMA.h"
#include "dataBoost.h"
#include "baseRMA.h"
#include "serRMA.h"
#include "greedyRMA.h"


namespace boosting {

  enum GreedyLevel   {EXACT, NotOptimal, Greedy};
  enum TestTrainData {TRAIN, TEST, VALID};
  //enum OuterInnerCV  {INNER, OUTER};

  class Boosting : public arg::ArgBoost, public base::BaseRMA { //public base::BaseBoost { //public DriverRMA,

  public:

    Boosting(int& argc, char**& argv); //  rma(NULL), prma(NULL), parallel(false)  { }; //: DriverRMA{argc, argv} {};   //:  rma(NULL), prma(NULL), parallel(false) {}; //, model(env) {};
    virtual ~Boosting();

    void         reset();
    void         setData(int& argc, char**& argv);
    void         setupPebblRMA(int& argc, char**& argv);
    virtual void setBoostingParameters() = 0;

    //////////////////////// training data //////////////////////////////
    void train(const bool& isOuter, const int& iter, const int & greedyLevel);

    /////////////////void discretizeData();
    //// virtual void   resetMaster() = 0;
    virtual void setInitRMP() = 0;
    void         solveRMP();
    virtual void setDataWts()      = 0;

    void   resetExactRMA();

    void   solveRMA();
    void   solveExactRMA();
    void   solveGreedyRMA();

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
    void         printRMASolutionTime();
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
    int  numRMASols; // # of boxes entered in the current interaction

    int  NumObs;
    int  NumAttrib;

    bool parallel;	// is parallel or not
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
    double *vecPrimal;  // dual variables
    double *vecDual;    // primal variables
    double primalVal;	 // primal solution value
    double *columnObjective;

    deque<bool>    vecIsCovered;	// each observation is covered or not

    ///////////////////// For RMA /////////////////////

    BasicArray<pebbl::solution*>       s;
    BasicArray<pebblRMA::rmaSolution*> sl;

    // store lower and upper bound of rules (boxes)
    vector<vector<unsigned int> >    matIntLower;
    vector<vector<unsigned int> >    matIntUpper;
    vector<vector<double> > matOrigLower;
    vector<vector<double> > matOrigUpper;

    ///////////////////// For Evaluation /////////////////////

    vector<bool>            vecCoveredSign;     // size: m (originalObs) x |K'|
    vector<vector<bool> >   vecCoveredObsByBox; // size: m (originalObs) x |K'|

    vector<double>          predTrain;          // predictions of training data by model
    vector<double>          predTest;           // predictions of testing data by model

    vector<double> vecERMA;
    vector<double> vecGRMA;

    double errTrain;
    double errTest;

    Time   tc;

    data::DataBoost*      data;
    pebblRMA::RMA*        rma;   // serial RMA instance
    pebblRMA::parRMA*     prma;  // parallel RMA instance
    greedyRMA::GreedyRMA* grma;  // greedy RMA instance

  };

} // namespace boosting


#endif
