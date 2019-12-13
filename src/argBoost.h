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

using namespace std;
//using namespace pebblRMA;


namespace arg {

  // static double inf = numeric_limits<double>::infinity();
  static int intInf = numeric_limits<int>::max();


//  Boosting parameters class
class ArgBoost :
  public ArgRMA,
  //virtual public pebbl::pebblParams,
  //virtual public pebbl::parallelPebblParams,
  virtual public utilib::ParameterSet,
  virtual public utilib::CommonIO {

public:

  ArgBoost();
  virtual ~ArgBoost(){};

/////////////////////// Data parameters ///////////////////////
  double getDelta()         const {return _delta;}
  double shrinkDelta()      const {return _shrinkDelta;}
  double getLimitInterval() const {return _limitInterval;}
  int    fixedSizeBin()      const {return _fixedSizeBin;}
  
/////////////////// parameters for CrossValidation class ///////////////////

  bool outerCV()    const {return _outerCV;}
  bool innerCV()    const {return _innerCV;}
  bool validation() const {return _validation;}
  
  int getNumLimitedObs() const {return _numLimitedObs;}
  
  bool shuffleObs()       const {return _shuffleObs;}
  bool readShuffledObs()  const {return _readShuffledObs;}
  bool writeShuffledObs() const {return _writeShuffledObs;}

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
  //bool greedyRMA()     const {return _greedyRMA;}
  //bool exactRMA()      const {return _exactRMA;}
  //bool greedyExactRMA() const {return _greedyExactRMA;}

  /////////////////////// RMA parameters ///////////////////////

  //bool initGuess() const {return _initGuess;}
  bool randSeed()  const {return _randSeed;}

  //////////////////////// Evaluation parameters ////////////////////////

  bool compModels()        const {return _compModels;}
  int  getCompModelIters() const {return _compModelIters;}
  bool evalEachIter()  const {return _evalEachIter;}
  bool evalFinalIter() const {return _evalFinalIter;}
  bool writePred()         const {return _writePredictions;}

 protected:

  /////////////////// parameters for Data class ///////////////////

  double _delta;
  double _shrinkDelta;
  double _limitInterval;
  int    _fixedSizeBin;

  /////////////////// Parameters for CrossValidation class ///////////////////

  bool _outerCV;						// enable outer crossvalitaion
  bool _innerCV; 						// enable inner cross valitaion (outerCV must be enabled)
  bool _validation;

  bool _shuffleObs;					// shuffle observations to choose test and train observations
  bool _readShuffledObs;    // read shuffled observation indices from a file
  bool _writeShuffledObs;   // write shuffled observation indices to a file

  /////////////////// Parameters for Boosting ///////////////////

  bool _isLPBoost;
  bool _isREPR;
  int _iterations;					// the number of iterations in column generation
  int _exponentP;	// exponent of residuals
  bool _printBoost; // print out more details for boosting

  /////////////////// Parameters for LPBoost class ///////////////////

  double _coeffD;	// coefficients parameters
  double _nu;     // D = 1 / (m * nu)
  bool _noSoftMargin;
  bool _initRules;
  bool _init1DRules;
  double _lowerRho;
  double _upperRho;

  /////////////////// Parameters for LPBoost class ///////////////////

  double _coeffC;	// coefficients parameters
  double _coeffE;	// coefficients parameters
  double _coeffF;	// coefficients parameters

  /////////////////// Parameters for Greedy level of pricing problems ///////////////////

  //bool _exactRMA;
  //bool _greedyRMA;
  bool _SeqCoverValue;
  int _numLimitedObs;
  int _maxBoundedSP;				// set a maximum number of bounded subproblems to check

  /////////////////// Parameters for RMA class  ///////////////////

  //bool _initGuess;		// compute an initial incumbent
  bool _randSeed;     // random seed for tied solution or bound
  
  //////////////////////// Evaluation parameters ////////////////////////

  bool _evalEachIter;   		// evaluate solutions in each iteration
  bool _evalFinalIter;      // evaluate solutions in the final column generation
  bool _compModels;		// compare out moredl to different models
  int _compModelIters;          // iterations or the number of trees for competing models
  bool _writePredictions;		// print prediction

  //////////////////////// Debugging parameters ////////////////////////

  //bool debug_solver_params1;

  };


} // namespace arg

#endif

