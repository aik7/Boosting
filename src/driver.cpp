/*
 *  File name:   main.cpp
 *  Author:      Ai Kagawa
 *  Description: a dirver to run LPBoost or REPR with cross-validation
 */

#include "repr.h"


int main(int argc, char** argv) {

  boosting::REPR repr(argc, argv);
  repr.train(true, repr.getIterations(), repr.greedyLevel); //isOuter=false, NumIter=10, greedyLevel=EXACT

  return 0;

}
