
/*
 *  File name: allParams.cpp
 *  Author:    Ai Kagawa
 */

//#include <acro_config.h>
#include "allParams.h"


namespace base {

using utilib::ParameterLowerBound;
using utilib::ParameterBounds;
using utilib::ParameterNonnegative;

allParams::allParams():

    _delta(0),
    _shrinkDelta(.95),
    _limitInterval(inf),
    _fixedSizeBin(-1),

    _outerCV(false),
    _innerCV(false),
    _validation(false),

    _isLPBoost(false),
    _isREPR(true),
    _iterations(1),
    _exponentP(2),
    _printBoost(false),

    _coeffD(0.5),
    _nu(.5),
    _noSoftMargin(false),
    _initRules(false),
    _init1DRules(false),
    _lowerRho(-inf),
    _upperRho(inf),

    _coeffC(1),
    _coeffE(1),
    _coeffF(0),

    _SeqCoverValue(false),
    _greedyRMA(true),
    _exactRMA(true),
    //_greedyExactRMA(false),

    _initGuess(true),
    _randSeed(true),

    _evalEachIter(false),
    _evalFinalIter(false),
    _compModels(false),
    _compModelIters(100),
    _writePredictions(false),

    _shuffleObs(true),
    _readShuffledObs(false),
    _writeShuffledObs(false),
    _numLimitedObs(intInf) //,

    //debug_solver_params1(false)

    {

///////////////////// Discretization parameters /////////////////////

  create_categorized_parameter("delta", _delta, "<double>",
    "0", "delta for recursive discretization", "Data");

  create_categorized_parameter("shrinkDelta", _shrinkDelta, "<double>",
    ".95", "shrink delta for recursive discretization", "Data");

  create_categorized_parameter("limitInterval", _limitInterval, "<double>",
    "inf", "limit Interval length of bouneded subproblems", "Data");

  create_categorized_parameter("fixedSizeBin", _fixedSizeBin, "<int>",
    "-1",	"integerized by the fixed size bin", "Data");

///////////////////// Corss Validation parameters /////////////////////

  create_categorized_parameter("outerCV", _outerCV, "<bool>", "false",
    "Do outer cross validation", "CrossValidation");

  create_categorized_parameter("innerCV", _innerCV, "<bool>", "false",
    "Do inner cross validation", "CrossValidation");

  create_categorized_parameter("validation", _validation, "<bool>",
     "false",	"using validation set for early stopping", "CrossValidation");

///////////////////// Shuffle observation parameters /////////////////////

  create_categorized_parameter("shuffleObs", _shuffleObs, "<bool>", "true",
       "Shuffle Observations", "CrossValidation");

  create_categorized_parameter("readShuffledObs", _readShuffledObs, "<bool>",
      "false", "Read shuffled observations", "CrossValidation");

  create_categorized_parameter("writeShuffledObs", _writeShuffledObs, "<bool>",
      "false", "Write shuffled observations", "CrossValidation");

  create_categorized_parameter("numLimitedObs", _numLimitedObs, "<int>",
      "inf", "limit number of observations to use", "CrossValidation");

///////////////////// Boosting parameters /////////////////////

  create_categorized_parameter("lpboost", _isLPBoost, "<bool>",
      "false",	"Run LPBoost", "LPBoost");

  create_categorized_parameter("repr", _isREPR, "<bool>",
      "true",	"Run REPR", "REPR");

  create_categorized_parameter("iterations", _iterations, "<int>", "1",
      "Number of LP-boosting iterations to run.  "
      "Each iteration runs\n\ta full branch and "
      "bound with different observation weights", "Boosting",
      utilib::ParameterLowerBound<int>(0));

  create_categorized_parameter("p", _exponentP, "<int>",
     "1", "exponent p", "Boosting");

  create_categorized_parameter("printBoost", _printBoost, "<bool>", "false",
    "print out more details to cehck boosting procedures", "Boosting");

///////////////////// LPBoost parameters /////////////////////

  create_categorized_parameter("d", _coeffD, "<double>",
      "0.5", "coefficient D", "LPBoost");

  create_categorized_parameter("nu", _nu, "<double>",
      "0.00", "coefficient D = 1/(m*nu)", "LPBoost");

  create_categorized_parameter("noSoftMargin", _noSoftMargin, "<bool>",
      "false", "No soft margin, all episilon_i has to be 0", "LPBoost");

  create_categorized_parameter("initRules", _initRules, "<bool>",
      "false", "LPBR has initial simple rules", "LPBoost");

  create_categorized_parameter("init1DRules", _init1DRules, "<bool>",
      "false", "LPBR has initial 1 dimentional rules", "LPBoost");

  create_categorized_parameter("lowerRho", _lowerRho, "<double>",
      "-inf", "lower bound of rho in LPBR", "LPBoost");

  create_categorized_parameter("upperRho", _upperRho, "<double>",
      "-inf", "upper bound of rho in LPBR", "LPBoost");

///////////////////// REPR parameters /////////////////////

  create_categorized_parameter("c", _coeffC, "<double>",
      "1.0", "coefficient C", "REPR");

  create_categorized_parameter("e", _coeffE, "<double>",
      "1.0", "coefficient E", "REPR");

  create_categorized_parameter("f", _coeffF, "<double>",
      "1.0", "coefficient F", "REPR");

///////////////////// RMA Greedy level parameters /////////////////////

  create_categorized_parameter("exactRMA", _exactRMA, "<bool>",
      "true",	"Exact B&B", "GreedyRMA");

  create_categorized_parameter("greedyRMA", _greedyRMA, "<bool>",
      "true",	"Greedy Range Search for RMA",
      "GreedyRMA");

  //create_categorized_parameter("greedyExactRMA", _greedyExactRMA, "<bool>",
  //    "false",	"Greedy Range Search for RMA",
  //    "GreedyRMA");

  create_categorized_parameter("SeqCoverValue", _SeqCoverValue, "<bool>",
      "false",	"Weighted Sequential Coering for Value ",
      "GreedyRMA");

///////////////////// inititial guess for RMA /////////////////////

  create_categorized_parameter("initGuess", _initGuess, "<bool>",
      "true", "enable the initial guess computation", "GreedyRMA");

  create_categorized_parameter("randSeed", _randSeed, "<bool>",
      "true", "random seed for tied solutions", "RMA");


///////////////////// Evaluation parameters /////////////////////

  create_categorized_parameter("evalEachIter", _evalEachIter, "<bool>",
      "true",	"Evaluate each iteration ", "Boosting");

  create_categorized_parameter("evalFinalIter", _evalFinalIter, "<bool>",
      "false", "evaluate model in the final iteration ", "Boosting");

  create_categorized_parameter("writePredictions", _writePredictions, "<bool>",
       "false", "Write predictions for each model ", "Boosting");

  create_categorized_parameter("compModels", _compModels, "<bool>",
       "false", "comparing our model with the other models", "Boosting");

  create_categorized_parameter("compModelIters", _compModelIters, "<int>",
      "100", "the nunmber of iteration or trees for compting models", "Boosting");

///////////////////// Debugging parameters /////////////////////
/*
  create_categorized_parameter("debug-solver-params1", debug_solver_params1,
        "<bool>", "false", "If true, print solver parameters", "Debugging");

  create_categorized_parameter("debug",debug, "<int>","0",
      "Debugging output level", "Debugging", ParameterNonnegative<int>());
*/
}

} // namespace lpboost


/*
    _countingSort(false),
    _branchSelection(0),
    _perCachedCutPts(0.000001),
    _binarySearchCutVal(false),

    _writeNodeTime(false),
    _writeCutPts(false),
    _rampUpSizeFact(1.0),
    _perLimitAttrib(1.0),
    _maxBoundedSP(intInf),
    _checkObjVal(false),
//*/

/*
  create_categorized_parameter("branchSelection", _branchSelection, "<int>",
    "0", "Among tied cutpoints, 0: randomize cutpoint to select, "
    "1: always select the first one, 2: always slect the last one", "RMA");

  ///////////////////// Non-strong branching parameters /////////////////////

  create_categorized_parameter("perCachedCutPts", _perCachedCutPts,
    "<double>", "false", "check only cut-points from the cache"
    "if the cache has at least x% of live cut-points out of total cut points",
    "RMA");

  create_categorized_parameter("binarySearchCutVal", _binarySearchCutVal,
    "<bool>", "false", "binary search cut values in each feature", "RMA");

///////////////////// RMA Limitation parameters /////////////////////

  create_categorized_parameter("countingSort", _countingSort, "<bool>",
    "false", "Use counting sort instead of bucket sort", "RMA");

  create_categorized_parameter("perLimitAttrib", _perLimitAttrib, "<double>",
      "1.00", "limit number of attributes to check ", "RMA");

  create_categorized_parameter("checkObjVal", _checkObjVal, "<bool>",
    "false",	"check the optimal solution in the end ", "RMA");

  create_categorized_parameter("writeCutPts", _writeCutPts, "<bool>",
    "false", "Write cut points chosen in the solution file ", "RMA");

  create_categorized_parameter("writeInstances", _writeInstances, "<bool>",
    "false", "Write an input file for each weighted problem solved", "RMA");

  create_categorized_parameter("writeNodeTime", _writeNodeTime, "<bool>",
    "false", "Write an input file for the number of B&B node and "
    "CPU time for each iteration", "RMA");

  create_categorized_parameter("testWt", _testWt, "<bool>", "false",
    "testing with specified test weights data, testWt.data", "RMA");

  create_categorized_parameter("maxBoundedSP", _maxBoundedSP, "<int>",
    "intInf", "maximum number of bouneded subproblems", "RMA");

  create_categorized_parameter("rampUpSizeFact", _rampUpSizeFact, "<double>",
    "1.00", "if (#storedCutPts) <= rampUpSizeFact * (#processors),"
    "get out the ramp-up", "RMA");
//*/
