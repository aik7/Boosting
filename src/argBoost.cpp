/*
 *  File name: argBoost.cpp
 *  Author:    Ai Kagawa
 */

#include "argBoost.h"


namespace arg {

  using utilib::ParameterLowerBound;
  using utilib::ParameterBounds;
  using utilib::ParameterNonnegative;

  ArgBoost::ArgBoost():

    _isREPR(true),
    _numIterations(1),
    _exponentP(2),
    _isPrintBoost(false),

    _coeffD(0.5),
    _nu(.5),
    _isNoSoftMargin(false),
    _isInitRules(false),
    _isInit1DRules(false),
    _lowerRho(-getInf()),
    _upperRho(getInf()),

    _coeffC(1),
    _coeffE(1),
    _coeffF(0),

    _isSeqCoverValue(false),
    _numLimitedObs(getIntInf()),
    _maxBoundedSP(10000000),

    _isEvalEachIter(false),
    _isEvalFinalIter(false),
    _isSaveWts(false),
    _isSavePredictions(false)

    {

      ///////////////////// Boosting parameters /////////////////////

  // create_categorized_parameter("lpboost", _isLPBoost, "<bool>",
  //     "false",	"Run LPBoost", "LPBoost");

  create_categorized_parameter("repr", _isREPR, "<bool>",
      "true",	"Run REPR", "REPR");

  create_categorized_parameter("numIterations", _numIterations, "<unsigned int>", "1",
      "Number of LP-boosting iterations to run.  "
      "Each iteration runs\n\ta full branch and "
      "bound with different observation weights", "Boosting",
      utilib::ParameterLowerBound<unsigned>(0));

  create_categorized_parameter("p", _exponentP, "<unsigned int>",
     "1", "exponent p", "Boosting");

  create_categorized_parameter("isPrintBoost", _isPrintBoost, "<bool>", "false",
    "print out more details to cehck boosting procedures", "Boosting");

///////////////////// LPBoost parameters /////////////////////

  create_categorized_parameter("d", _coeffD, "<double>",
      "0.5", "coefficient D", "LPBoost");

  create_categorized_parameter("nu", _nu, "<double>",
      "0.00", "coefficient D = 1/(m*nu)", "LPBoost");

  create_categorized_parameter("isNoSoftMargin", _isNoSoftMargin, "<bool>",
      "false", "No soft margin, all episilon_i has to be 0", "LPBoost");

  create_categorized_parameter("isInitRules", _isInitRules, "<bool>",
      "false", "LPBR has initial simple rules", "LPBoost");

  create_categorized_parameter("isInit1DRules", _isInit1DRules, "<bool>",
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

  create_categorized_parameter("isSeqCoverValue", _isSeqCoverValue, "<bool>",
      "false",	"Weighted Sequential Coering for Value ",
      "GreedyRMA");

  create_categorized_parameter("numLimitedObs", _numLimitedObs, "<unsigned>",
      "inf", "limit number of observations to use", "CrossValidation");

///////////////////// Evaluation parameters /////////////////////

  create_categorized_parameter("isEvalEachIter", _isEvalEachIter, "<bool>",
      "true",	"Evaluate each iteration ", "Boosting");

  create_categorized_parameter("isEvalFinalIter", _isEvalFinalIter, "<bool>",
      "false", "evaluate model in the final iteration ", "Boosting");

  create_categorized_parameter("isSaveWts", _isSaveWts, "<bool>",
       "false", "Wether or not to save weights for each boosting iteration ",
       "Boosting");

  create_categorized_parameter("_isSavePredictions", _isSavePredictions, "<bool>",
       "false", "Write predictions for each model ", "Boosting");

  }
}
