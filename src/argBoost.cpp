\
/*
 *  File name: allParams.cpp
 *  Author:    Ai Kagawa
 */

//#include <acro_config.h>
#include "argBoost.h"


namespace arg {

using utilib::ParameterLowerBound;
using utilib::ParameterBounds;
using utilib::ParameterNonnegative;

ArgBoost::ArgBoost():

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
    //_greedyRMA(true),
    //_exactRMA(true),
    //_greedyExactRMA(false),

    //_initGuess(true),
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

  //create_categorized_parameter("exactRMA", _exactRMA, "<bool>",
  //    "true",	"Exact B&B", "GreedyRMA");

  //create_categorized_parameter("greedyRMA", _greedyRMA, "<bool>",
  //    "true",	"Greedy Range Search for RMA",
  //    "GreedyRMA");

  //create_categorized_parameter("greedyExactRMA", _greedyExactRMA, "<bool>",
  //    "false",	"Greedy Range Search for RMA",
  //    "GreedyRMA");

  create_categorized_parameter("SeqCoverValue", _SeqCoverValue, "<bool>",
      "false",	"Weighted Sequential Coering for Value ",
      "GreedyRMA");

///////////////////// inititial guess for RMA /////////////////////

  //create_categorized_parameter("initGuess", _initGuess, "<bool>",
  //    "true", "enable the initial guess computation", "GreedyRMA");

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

  }
}
