#include "runTrainedModel.h"


int main(int argc, char** argv) {

  TrainedREPR repr(argc, argv);

  // compute the predicted values and print them
  repr.predict();
  // repr.printVecPredY();

  // compute MSE and print mse
  repr.computeMSE();
  repr.printMSE();

  return 0;
}
