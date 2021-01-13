/*
 *  Author:      Ai Kagawa
 *  Description: a hedare file for Boosintg arguments
 */

#ifndef ARG_BOOST_h
#define ARG_BOOST_h

#include <pebbl_config.h>
#include <pebbl/utilib/ParameterSet.h>
#include <limits>
#include "argRMA.h"
#include "utility.h"


namespace arg {

  //  Boosting parameters class
  class ArgBoost :
    virtual public utilib::ParameterSet,
    virtual public utilib::CommonIO {

  public:

    ArgBoost();
    virtual ~ArgBoost(){};

    /////////////////// parameters for Boosting ///////////////////

    // bool         isLPBoost()        const {return _isLPBoost;}
    bool         isUseGurobi()      const {return _isUseGurobi;}
    bool         isREPR()           const {return _isREPR;}
    unsigned int getNumIterations() const {return _numIterations;}
    unsigned int getExponentP()     const {return _exponentP;}
    bool         isPrintBoost()     const {return _isPrintBoost;}

    /////////////////// parameters for LPBR class ///////////////////

    double getCoefficientD() const {return _coeffD;}
    double getNu()           const {return _nu;}
    bool   isNoSoftMargin()  const {return _isNoSoftMargin; }
    double getLowerRho()     const {return _lowerRho; }
    double getUpperRho()     const {return _upperRho;}
    bool   isInitRules()     const {return _isInitRules; }
    bool   isInit1DRules()   const {return _isInit1DRules; }

    /////////////////// parameters for REPR class ///////////////////

    double getCoefficientC() const {return _coeffC;}
    double getCoefficientE() const {return _coeffE;}
    double getCoefficientF() const {return _coeffF;}

    /////////// Parameters for Greedy level of pricing problems ///////////////

    // TODO: what is this variable?
    bool         isSeqCoverValue() const {return _isSeqCoverValue;}
    unsigned int getNumLimitedObs() const {return _numLimitedObs;}

    //////////////////////// Evaluation parameters ////////////////////////

    bool isEvalEachIter()   const {return _isEvalEachIter;}
    bool isEvalFinalIter()  const {return _isEvalFinalIter;}

    bool isSaveWts()        const {return _isSaveWts;}
    bool isSavePred()       const {return _isSavePredictions;}
    bool isSaveAllRMASols() const {return _isSaveAllRMASols;}
    bool isSaveClpMps()     const {return _isSaveClpMps;}

    bool isStandData()      const {return _isStandData;}

  protected:

    /////////////////// Parameters for Boosting ///////////////////

    // if this is true, use Gurobi to solve RMP else use CLP
    bool         _isUseGurobi;

    // if isREPR=true, run REPR; else, run LPBosot
    bool          _isREPR;

    // # of iterations in column generation
    unsigned int  _numIterations;

    // exponent of residuals
    unsigned int  _exponentP;

    // whether or not to print more details for boosting
    bool          _isPrintBoost;

    /////////////////// Parameters for LPBoost class ///////////////////

    double _coeffD;	  // coefficients parameters
    double _nu;            // D = 1 / (m * nu)
    bool   _isNoSoftMargin;
    bool   _isInitRules;
    bool   _isInit1DRules;
    double _lowerRho;
    double _upperRho;

    /////////////////// Parameters for LPBoost class ///////////////////

    double _coeffC;  // coefficients parameters
    double _coeffE;  // coefficients parameters
    double _coeffF;  // coefficients parameters

    //////////// Parameters for Greedy level of pricing problems /////////////

    bool         _isSeqCoverValue;
    unsigned int _numLimitedObs;

    // set a maximum number of bounded subproblems to check
    unsigned int _maxBoundedSP;

    //////////////////////// Evaluation parameters ////////////////////////

    // whether or not to evaluate solutions in each iteration
    bool _isEvalEachIter;

    // whether or not to evaluate solutions in the final column generation
    bool _isEvalFinalIter;

    // whether or not to save weights for each boosting iteration
    bool _isSaveWts;

    // whether or not to save actual and predicted y-values
    bool _isSavePredictions;

    // whether or not to save all RMA solutions for each Boosting iterations
    bool _isSaveAllRMASols;

    // whether or not to save ClpMps files for each boosting iteration
    bool _isSaveClpMps;

    // whether or not to standerdize the data
    bool _isStandData;

  }; // end ArgBoost class

} // namespace arg

#endif
