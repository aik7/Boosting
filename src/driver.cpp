/*
 *  Author:      Ai Kagawa
 *  Description: a dirver to run LPBoost or REPR with cross-validation
 */

#include "repr.h"


int main(int argc, char** argv) {

  boosting::REPR repr(argc, argv);
  repr.train(repr.getNumIterations(), repr.greedyLevel);

  return 0;

}
