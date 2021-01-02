#include "runTrainedModel.h"


int main(int argc, char** argv) {
  TrainedREPR repr(argc, argv);
  repr.predict();
  return 0;
}
