/*
 *  File name:   main.cpp
 *  Author:      Ai Kagawa
 *  Description: a dirver to run LPBoost or REPR with cross-validation
 */

#include "repr.h"


int main(int argc, char** argv) {

  boosting::REPR repr(argc, argv);
  repr.train(true, repr.getIterations(), repr.greedyLevel);

  // boosting::REPR* repr = new boosting::REPR(argc, argv);
  // repr->train(true, repr->getIterations(), repr->greedyLevel);
  // delete repr;

  return 0;

}
