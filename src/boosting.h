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

//#include "gurobi_c++.h"
#include "Time.h"
#include "argRMA.h"
#include "dataRMA.h"
#include "dataBoost.h"
#include "baseRMA.h"
//#include "baseBoost.h"
//#include "driverRMA.h"
#include "serRMA.h"
#include "greedyRMA.h"

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


namespace base {

  enum GreedyLevel   {EXACT, NotOptimal, Greedy};
  //enum OuterInnerCV  {INNER, OUTER};
  //enum TestTrainData {TRAIN, TEST, VALID};

  struct IntMinMax { double minOrigVal, maxOrigVal; };

  struct Feature {vector<IntMinMax> vecIntMinMax;};

/////////////////////////  Boosting Base class /////////////////////////
  class BaseBoost : public arg::ArgBoost, public BaseRMA, // {  //{ //, public ArgRMA,
    virtual public pebbl::pebblParams,
    virtual public pebbl::parallelPebblParams {
  public:
      BaseBoost() {}
      ~BaseBoost() {};
    };

} // base namespace



namespace boosting {


  enum TestTrainData {TRAIN, TEST, VALID};

  using namespace utilib;
  using namespace pebbl;
  using namespace std;
  using namespace arg;
  using namespace data;
  using namespace base;
  using namespace pebblRMA;
  using namespace greedyRMA;


  class Boosting : public BaseBoost { //public DriverRMA,

public:

  Boosting(int& argc, char**& argv); //  rma(NULL), prma(NULL), parallel(false)  { }; //: DriverRMA{argc, argv} {};   //:  rma(NULL), prma(NULL), parallel(false) {}; //, model(env) {};

  // model.set(GRB_IntAttr_ModelSense,1); // minimization
  virtual ~Boosting() {
#ifdef ACRO_HAVE_MPI
    if (parallel) {
      CommonIO::end();
      uMPI::done();
    }
#endif // ACRO_HAVE_MPI
  };

  void setData(int& argc, char**& argv) {
    data = new DataBoost(argc, argv, (BaseRMA *) this, (ArgBoost *) this);
  }

  void setupRMA(int& argc, char**& argv);

  virtual void initBoostingData() = 0;

  //////////////////////// training data //////////////////////////////
  virtual void trainData(const bool& isOuter, const int& iter,
			                   const int & greedyLevel) = 0;
  /////////////////void discretizeData();
  virtual void setInitialMaster() = 0;
  //void solveInitialMaster();
  virtual void setDataWts() = 0;

  void         solveRMA();
  void         solveExactRMA();
  void         solveGreedyRMA();

  virtual void insertColumns() = 0; //const int& GreedyLevel
  void         solveMaster();

  void   setOriginalBounds();
  double getLowerBound(int k, int j, int value, bool isUpper) ;
  double getUpperBound(int k, int j, int value, bool isUpper) ;

  void   resetGurobi();

  virtual void printRMPSolution() = 0;  		// restricted mater problem solution
  virtual void printRMAInfo()     = 0;		  // pritinc problem, RMA

  //////////////////////// checking duplicate ///////////////////////
  bool isDuplicate();
  void checkObjValue(vector<DataXw> intData);
  void checkObjValue(int k, vector<DataXw> intData);	// double-check objevtive value for (a, b)

  //////////////////////// Evaluation  /////////////////////////////

  void setCoveredTrainObs();
  void setCoveredTestObs();

  void evaluateEach();
  void evaluateFinal();

  virtual double evaluateEachIter(const int& isTest, vector<DataXy> origData) = 0;
  virtual double evaluateAtFinal(const int& isTest, vector<DataXy> origData)  = 0;

  virtual void printEachIterAllErrs() = 0;

  void writePredictions(const int& isTest, vector<DataXy> origData); // write predictions

  //private:

  int NumIter;	  // # of iterations, observation, features, and variables
  int curIter;		// the current iteration number

  int numRows;    // # of constraints / rows in the mater problem
  int numCols;    // # of variables / columns in the mater problem
  int numBox;     // # of total boxes entered so far
  int numRMASols; // # of boxes entered in the current interaction

  int NumObs;
  int NumAttrib;

  bool parallel;	// is parallel or not
  bool flagDuplicate;
  bool isOuter;

  /*
  ///////////////////// GUROBI variables /////////////////////
  GRBEnv      env;
  GRBModel    model;
  GRBLinExpr  lhs;
  GRBConstr*  constr;
  GRBVar*     vars;
  GRBColumn   col;
  GRBQuadExpr obj;
  */

  // store solution infomation for the master problem
  vector<double> vecPrimal;  // dual variables
  vector<double> vecDual;    // primal variables
  double         primalVal;	 // primal solution value

  deque<bool> vecIsCovered;	// each observation is covered or not

  ///////////////////// For RMA /////////////////////

  BasicArray<solution*>    s;
  BasicArray<rmaSolution*> sl;

  // store lower and upper bound of rules (boxes)
  vector<vector<int> >    matIntLower;
  vector<vector<int> >    matIntUpper;
  vector<vector<double> > matOrigLower;
  vector<vector<double> > matOrigUpper;

  ///////////////////// For Evaluation /////////////////////

  vector<bool> vecCoveredSign;  // size: m (originalObs) x |K'|
  vector<vector<bool> > vecCoveredObsByBox;  // size: m (originalObs) x |K'|
  vector<double> predTrain;  // predictions of training data by model
  vector<double> predTest;   // predictions of testing data by model

  double errTrain;
  double errTest;

  double timeWall;
  double timeCPU;

  Time   tc;

  DataBoost* data;

  RMA*       rma;   // serial RMA instance
  parRMA*    prma;  // parallel RMA instance
  GreedyRMA* grma;  // greedy RMA instance

};

} // namespace boosting


#endif
