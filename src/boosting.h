/*
 *  File name:   boosting_h
 *  Author:      Ai Kagawa
 *  Description: a header file for Boosting class
 */

#ifndef Boosting1_h
#define Boosting1_h

#include <pebbl_config.h>
#include <ParameterList.h>
#include <memdebug.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <map>

#include "gurobi_c++.h"
#include "Time.h"
#include "base.h"
#include "serRMA.h"
#include "greedyRMA.h"

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


namespace boosting {


using namespace utilib;
using namespace std;
using namespace pebblRMA;
using namespace greedyRMA;


class Boosting {

public:

  Boosting() : rma(NULL), prma(NULL), parallel(false), model(env) {};


  // model.set(GRB_IntAttr_ModelSense,1); // minimization
	virtual ~Boosting() {
    #ifdef ACRO_HAVE_MPI
      if (parallel) {
        CommonIO::end();
        uMPI::done();
      }
    #endif // ACRO_HAVE_MPI
  };

	virtual void initBoostingData() = 0;

	//////////////////////// training data //////////////////////////////
	virtual void trainData(const bool& isOuter, const int& iter,
    const int& greedyLevel) = 0;
	/////////////////void discretizeData();
	virtual void setInitialMaster() = 0;
  //void solveInitialMaster();
	virtual void setDataWts() = 0;
	void solveRMA();
	virtual void insertColumns() = 0; //const int& GreedyLevel
	void solveMaster();

	void setOriginalBounds();
	double getLowerBound(int k, int j, int value, bool isUpper) ;
	double getUpperBound(int k, int j, int value, bool isUpper) ;

  void resetGurobi();

	virtual void printRMPSolution() = 0;  		// restricted mater problem solution
	virtual void printRMAInfo() = 0;					// pritinc problem, RMA

	//////////////////////// checking duplicate ///////////////////////
	bool isDuplicate();
  void checkObjValue();
	void checkObjValue(int k);	// double-check objevtive value for (a, b)

  //////////////////////// Evaluation  /////////////////////////////

	void setCoveredTrainObs();
	void setCoveredTestObs();

	void evaluateEach();
	void evaluateFinal();

	virtual double evaluateEachIter(const int& isTest) = 0;
  virtual double evaluateAtFinal(const int& isTest) = 0;

  virtual void printEachIterAllErrs() = 0;

  void writePredictions(const int& isTest); // write predictions

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

	///////////////////// GUROBI variables /////////////////////
	GRBEnv       env;
	GRBModel    model;
  GRBLinExpr 	lhs;
  GRBConstr* 	constr;
	GRBVar* 		vars;
	GRBColumn 	col;
  GRBQuadExpr obj;

	// store solution infomation for the master problem
	vector<double> vecPrimal; // dual variables
	vector<double> vecDual;		// primal variables
	double  primalVal;				// primal solution value

	deque<bool> vecIsCovered;	// each observation is covered or not

	///////////////////// For RMA /////////////////////

	BasicArray<solution*> s;
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

	Time tc;
	Data* data;

  RMA*    rma;		  // serial RMA instance
  parRMA* prma;	    // parallel RMA instance
	GreedyRMA* grma;  // greedy RMA instance

};

} // namespace boosting


#endif


/*
// call GUROBI to solve Master Problems
void REPR::solveInitialMaster() {

	int i;

	model.optimize();

  constr = model.getConstrs();

  vecPrimal.resize(numCols);
	vecDual.resize(numRows);

  if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
    primalVal = model.get(GRB_DoubleAttr_ObjVal);
    for (i = 0; i < numCols; ++i)
      vecPrimal[i] = vars[i].get(GRB_DoubleAttr_X);
    for (i = 0; i < numRows; ++i)
      vecDual[i] = constr[i].get(GRB_DoubleAttr_Pi);
  }

#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
	ucout << "Initial Master Problem Solution: " << primalVal << "\n";
  if (!data->shuffleObs()) printRMPSolution();
#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI

	if (data->evaluateEachIter()) {
		ucout << "i: 0\t";
		evaluateEach();
	}

} // solveInitialMaster()
*/
