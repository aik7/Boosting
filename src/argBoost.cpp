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
    _lowerRho(-getInf()),
    _upperRho(getInf()),

    _coeffC(1),
    _coeffE(1),
    _coeffF(0),

    _SeqCoverValue(false),
    _numLimitedObs(getIntInf()),
    _maxBoundedSP(10000000),

    _evalEachIter(false),
    _evalFinalIter(false),
    _saveWts(false),
    _writePredictions(false)

    {

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

  create_categorized_parameter("SeqCoverValue", _SeqCoverValue, "<bool>",
      "false",	"Weighted Sequential Coering for Value ",
      "GreedyRMA");


  create_categorized_parameter("numLimitedObs", _numLimitedObs, "<int>",
      "inf", "limit number of observations to use", "CrossValidation");

///////////////////// Evaluation parameters /////////////////////

  create_categorized_parameter("evalEachIter", _evalEachIter, "<bool>",
      "true",	"Evaluate each iteration ", "Boosting");

  create_categorized_parameter("evalFinalIter", _evalFinalIter, "<bool>",
      "false", "evaluate model in the final iteration ", "Boosting");

  create_categorized_parameter("saveWts", _writePredictions, "<bool>",
       "false", "Write weights for each boosting iteration ", "Boosting");

  create_categorized_parameter("writePredictions", _writePredictions, "<bool>",
       "false", "Write predictions for each model ", "Boosting");

  }
}
