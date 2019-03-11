/*
 *  File name: rmaParams.h
 *  Author:    Ai Kagawa
 */

#ifndef rma_Params_h
#define rma_Params_h

#include <pebbl/utilib/ParameterSet.h>
#include <pebbl_config.h>
#include <limits>

using namespace std;

namespace pebblRMA {

 //static double inf = numeric_limits<double>::infinity();
 //static int intInf = numeric_limits<int>::max();

 //  LPBoost parameters class
 class rmaParams :
   virtual public utilib::ParameterSet,
   virtual public utilib::CommonIO {

 public:

   rmaParams();
   ~rmaParams(){};

   /////////////////////// RMA parameters ///////////////////////
   double perCachedCutPts() const {return _perCachedCutPts;}
   bool binarySearchCutVal() const {return _binarySearchCutVal;}
   double perLimitAttrib() const {return _perLimitAttrib;}
   bool checkObjVal() const {return _checkObjVal;}
   bool writingInstances() const {return _writeInstances;}
   bool writingNodeTime() const {return _writeNodeTime;}
   bool writingCutPts() const {return _writeCutPts;}
   bool testWeight() const {return _testWt;}
   double rampUpSizeFact() const {return _rampUpSizeFact;}
   bool countingSort() const {return _countingSort;}
   int branchSelection() const {return _branchSelection;}

 //protected:

   /////////////////// Parameters for RMA class  ///////////////////

   // for non-strong branching ...
   double _perCachedCutPts;		// check only stored cuts points which is x % of total cut points
   bool _binarySearchCutVal;	// binarySearchCutVal
   double _perLimitAttrib;			// percentages of features to check

   bool _checkObjVal;				// check the solution is right in the end

   bool _writeInstances;
   bool _writeNodeTime;			// make an output file containing BoundedSP and run time
   bool _writeCutPts;

   bool _testWt;

   double _rampUpSizeFact;

   bool _countingSort;
   int _branchSelection;

 };


} // namespace pebblRMA

 #endif
