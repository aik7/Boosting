/*
 *  File name: argBoost.h
 *  Author:    Ai Kagawa
 */

#ifndef ARG_BOOST_h
#define ARG_BOOST_h

#include <pebbl_config.h>
#include <pebbl/utilib/ParameterSet.h>
#include <limits>
#include "argRMA.h"
#include "utilRMA.h"


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

    /////////////////// Parameters for Greedy level of pricing problems ///////////////////

    // TODO: what is this variable?
    bool         isSeqCoverValue() const {return _isSeqCoverValue;}
    unsigned int getNumLimitedObs() const {return _numLimitedObs;}

    //////////////////////// Evaluation parameters ////////////////////////

    bool isEvalEachIter()  const {return _isEvalEachIter;}
    bool isEvalFinalIter() const {return _isEvalFinalIter;}

    bool isSaveWts()       const {return _isSaveWts;}
    bool isSavePred()    const {return _isSavePredictions;}

  protected:

    /////////////////// Parameters for Boosting ///////////////////

    bool          _isREPR;
    unsigned int  _numIterations;  // the number of iterations in column generation
    unsigned int  _exponentP;	    // exponent of residuals
    bool          _isPrintBoost;    // print out more details for boosting

    /////////////////// Parameters for LPBoost class ///////////////////

    double _coeffD;	  // coefficients parameters
    double _nu;            // D = 1 / (m * nu)
    bool   _isNoSoftMargin;
    bool   _isInitRules;
    bool   _isInit1DRules;
    double _lowerRho;
    double _upperRho;

    /////////////////// Parameters for LPBoost class ///////////////////

    double _coeffC;	// coefficients parameters
    double _coeffE;	// coefficients parameters
    double _coeffF;	// coefficients parameters

    /////////////////// Parameters for Greedy level of pricing problems ///////////////////

    bool         _isSeqCoverValue;
    unsigned int _numLimitedObs;
    unsigned int _maxBoundedSP;  // set a maximum number of bounded subproblems to check

      //////////////////////// Evaluation parameters ////////////////////////

    bool _isEvalEachIter;       // evaluate solutions in each iteration
    bool _isEvalFinalIter;      // evaluate solutions in the final column generation
    bool _isSaveWts;            // save weights for each boosting iteration
    bool _isSavePredictions;    // print prediction

  }; // end ArgBoost class

} // namespace arg

#endif
