/*
 *  File name: allParams.h
 *  Author:    Ai Kagawa
 */

#ifndef ARG_BOOST_h
#define ARG_BOOST_h

#include <pebbl_config.h>
#include <pebbl/utilib/ParameterSet.h>
#include <limits>
#include "argRMA.h"


namespace arg {

  // static double inf = numeric_limits<double>::infinity();
  static int intInf = numeric_limits<int>::max();
  
  
  //  Boosting parameters class
  class ArgBoost :
    virtual public utilib::ParameterSet,
    virtual public utilib::CommonIO {
      
  public:
      
      ArgBoost();
      virtual ~ArgBoost(){};
    
      /////////////////// parameters for Boosting ///////////////////

      bool isLPBoost()     const {return _isLPBoost;}
      bool isREPR()        const {return _isREPR;}
      int  getIterations() const {return _iterations;}
      int  getExponentP()  const {return _exponentP;}
      bool printBoost()    const {return _printBoost;}

      /////////////////// parameters for LPBR class ///////////////////

      double getCoefficientD() const {return _coeffD;}
      double getNu()           const {return _nu;}
      bool   getNoSoftMargin() const {return _noSoftMargin; }
      double getLowerRho()     const {return _lowerRho; }
      double getUpperRho()     const {return _upperRho;}
      bool   initRules()       const {return _initRules; }
      bool   init1DRules()     const {return _init1DRules; }

      /////////////////// parameters for REPR class ///////////////////

      double getCoefficientC() const {return _coeffC;}
      double getCoefficientE() const {return _coeffE;}
      double getCoefficientF() const {return _coeffF;}

      /////////////////// Parameters for Greedy level of pricing problems ///////////////////

      bool SeqCoverValue() const {return _SeqCoverValue;}
      int  getNumLimitedObs() const {return _numLimitedObs;}

      //////////////////////// Evaluation parameters ////////////////////////

      bool evalEachIter()  const {return _evalEachIter;}
      bool evalFinalIter() const {return _evalFinalIter;}
      bool writePred()         const {return _writePredictions;}

  protected:

      /////////////////// Parameters for Boosting ///////////////////

      bool _isLPBoost;
      bool _isREPR;
      int  _iterations;					// the number of iterations in column generation
      int  _exponentP;	// exponent of residuals
      bool _printBoost; // print out more details for boosting

      /////////////////// Parameters for LPBoost class ///////////////////

      double _coeffD;	// coefficients parameters
      double _nu;     // D = 1 / (m * nu)
      bool   _noSoftMargin;
      bool   _initRules;
      bool   _init1DRules;
      double _lowerRho;
      double _upperRho;

      /////////////////// Parameters for LPBoost class ///////////////////

      double _coeffC;	// coefficients parameters
      double _coeffE;	// coefficients parameters
      double _coeffF;	// coefficients parameters

      /////////////////// Parameters for Greedy level of pricing problems ///////////////////

      bool _SeqCoverValue;
      int  _numLimitedObs;
      int  _maxBoundedSP;				// set a maximum number of bounded subproblems to check

      //////////////////////// Evaluation parameters ////////////////////////

      bool _evalEachIter;   		// evaluate solutions in each iteration
      bool _evalFinalIter;      // evaluate solutions in the final column generation
      bool _writePredictions;		// print prediction

    };

} // namespace arg

#endif
