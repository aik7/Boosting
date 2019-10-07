/*
 *  File name: rmaParams.h
 *  Author:    Ai Kagawa
 */


#include "rmaParams.h"


namespace pebblRMA {

using utilib::ParameterLowerBound;
using utilib::ParameterBounds;
using utilib::ParameterNonnegative;

rmaParams::rmaParams():
    _perCachedCutPts(0.000001),
    _binarySearchCutVal(false),
    _perLimitAttrib(1.0),
    _writeNodeTime(false),
    _writeCutPts(false),
    _rampUpSizeFact(1.0),
    _checkObjVal(false),
    _countingSort(false),
    _branchSelection(0) {

  create_categorized_parameter("perCachedCutPts", _perCachedCutPts,
    "<double>", "false", "check only cut-points from the cache"
    "if the cache has at least x% of live cut-points out of total cut points",
    "RMA");

  create_categorized_parameter("binarySearchCutVal", _binarySearchCutVal,
    "<bool>", "false", "binary search cut values in each feature", "RMA");

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

  create_categorized_parameter("rampUpSizeFact", _rampUpSizeFact, "<double>",
    "1.00", "if (#storedCutPts) <= rampUpSizeFact * (#processors),"
    "get out the ramp-up", "RMA");

  create_categorized_parameter("countingSort", _countingSort, "<bool>",
    "false", "Use counting sort instead of bucket sort", "RMA");

  create_categorized_parameter("branchSelection", _branchSelection, "<int>",
    "0", "Among tied cutpoints, 0: randomize cutpoint to select, "
    "1: always select the first one, 2: always slect the last one", "RMA");

}

} // namespace lpboost
