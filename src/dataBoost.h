#ifndef DATA_BOOST_H
#define DATA_BOOST_H

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

//#include "Time.h"
#include "dataRMA.h"
#include "argBoost.h"

namespace data {

  using namespace std;
  using namespace arg;
  
  class DataBoost : public DataRMA {
    
  public:
    
    DataBoost() {}
    //DataBoost(int& argc, char**& argv, ArgBoost *args);
  DataBoost(int& argc, char**& argv, ArgRMA *args_rma, ArgBoost *args_boost) :
    DataRMA(argc, argv, args_rma) {
      numTestObs = numOrigObs;
      vecTestData.resize(numTestObs);
    }
    
    //private:
    int numTestObs;                    // # of testing observations
    vector<int>     vecTestData;       // contains only testing dataset observations
    vector<DataXy>  origTestData;      // original datasets X and y
    vector<DataXw>  intTestData;       // discretized data X abd w (weight)
    vector<DataXy>  standTestData;
    
  };
  
} // end nemespace data

#endif // end DATA_BOOST_H
