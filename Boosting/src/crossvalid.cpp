/*
 *  File name:   CrossValid.cpp
 *  Author:      Ai Kagawa
 *  Description: a source file for CrossValidation classe
 */

#include "crossvalid.h"


namespace crossvalidation {

/////////////////////////////// Cross Validation ///////////////////////////////

void CrossValidation::runOuterCrossValidation() {

  avgOutTestErr=0.0;
  avgOutTrainErr=0.0;
  avgTime = 0;
  avgTimePerFold = 0;

  setOuterPartition();

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
  if (compModels()) {
    if (isLPBoost()) {
      lcm = new compLPBR((Data*)this);
      lcm->setCompModelsCV();
    } else {
      rcm = new compREPR((Data*)this);
      rcm->setCompModelsCV();
    }
  }
#ifdef ACRO_HAVE_MPI
  }
#endif //  ACRO_HAVE_MPI

  for (int i=0; i<NumOutPartition; ++i) {    ////// outer iteration i  ////////

    setCurrnetDataSets(OUTER, i);

    Time tCV;
    if (innerCV()) {
      tCV.startTime();
      selectParamters(i);
    }

    (isLPBoost()) ? lpbr->isOuter = true : repr->isOuter = true ;

    Time t;
    t.startTime();

    if (isLPBoost()) lpbr->trainData(OUTER, getIterations(), EXACT);
    else             repr->trainData(OUTER, getIterations(), EXACT);

  #ifdef ACRO_HAVE_MPI
  	if (uMPI::rank==0) {
  #endif //  ACRO_HAVE_MPI
    ucout << "RMA Boosting ";
    avgTime += t.endCPUTime();
    if (innerCV()) {
      ucout << "RMA Boosting with Param Selection ";
      avgTimePerFold += tCV.endCPUTime();
    }
  #ifdef ACRO_HAVE_MPI
    }
  #endif //  ACRO_HAVE_MPI

    avgOutTrainErr += (isLPBoost()) ? lpbr->errTrain : repr->errTrain ;
    avgOutTestErr  += (isLPBoost()) ? lpbr->errTest  : repr->errTest ;

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
    if (compModels())
      if (isLPBoost()) {
        lcm->runCompModels();
        lcm->printCompModels();
      } else {
        rcm->runCompModels();
        rcm->printCompModels();
      }

    if (writePred()) {
      writePredictions(TRAIN);
      writePredictions(TEST);
    }
#ifdef ACRO_HAVE_MPI
  }
#endif //  ACRO_HAVE_MPI

    if (!outerCV()) break;

  } // end outer partition

#ifdef ACRO_HAVE_MPI
if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
  if (outerCV()) printOuterScore();
#ifdef ACRO_HAVE_MPI
}
#endif //  ACRO_HAVE_MPI

}


// select parameters by inner cross-validation for outer iteration i
void CrossValidation::selectParamters(const int& i) {

  (isLPBoost()) ? lpbr->isOuter = false : repr->isOuter = false ;

	double minAvgParamMSE = inf;
  int innerCVIter=getIterations();

  for (int m=0; m<NumComb; ++m) {
    avgInTestErr[m] = 0 ;
    avgInTrainErr[m]  = 0 ;
  }

  if (innerCVIter>20) innerCVIter=20;

	setInnerPartition(i);

  for (int j=0; j<NumInPartition; ++j) {  ////// inner iteration j (starts) ///////

    setCurrnetDataSets(INNER, j);

  	for (int m=0; m<NumComb; ++m) {  // iteration number of coefficient combination m (starts) //

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
  		ucout << "j: " << j  << " param: " << vecParams[m] ;
  			//	 << " Test/Train MSE:    " << REPRmseTest2 << "\t" << REPRmseTrain2 << "\n";
#ifdef ACRO_HAVE_MPI
  }
#endif //  ACRO

  		if (isLPBoost()) {
        lpbr->D = vecParams[m];
        lpbr->trainData(INNER, innerCVIter, Greedy);
        avgInTestErr[m]  += lpbr->errTest;
        avgInTrainErr[m] += lpbr->errTrain;
      } else {
        repr->C = vecParams[m];
        repr->E = vecParams[m];
        repr->trainData(INNER, innerCVIter, Greedy);
        avgInTestErr[m]  += repr->errTest;
        avgInTrainErr[m] += repr->errTrain;
      }

    } //////// iteration  number of coefficient combination, m (ends) ///////

  } /////////////////// inner iteration j (ends) ////////////////

  for (int m=0; m<NumComb; ++m) {  // iteration number of coefficient combination m (starts) //

    avgInTestErr[m] /= (double)NumInPartition;
    avgInTrainErr[m] /= (double)NumInPartition;

    // select parameters which has the lowest avgREPRmseTest2
    if ( minAvgParamMSE > avgInTestErr[m] ) {
      minAvgParamMSE = avgInTestErr[m] ;
      bestParamComb = m;
    }

  } /////////////////// each parameter (ends) ////////////////

  if (isLPBoost()) {
    lpbr->D = vecParams[bestParamComb];
  } else {
    repr->C = vecParams[bestParamComb];
    repr->E = vecParams[bestParamComb];
  }

	printInnerScore(i);
/*
  #ifdef ACRO_HAVE_MPI
    static MPI_Datatype mpiType;
    MPI_Bcast(&repr->C, 1, mpiType, 0, MPI_COMM_WORLD);
    repr->E = repr->C;
  #endif //  ACRO_HAVE_MPI
*/

/*
  for (i = 0; i < world_size; i++) {
    if (i != world_rank) {
      MPI_Send(data, count, datatype, i, 0, communicator);
    }
  }
  } else {
  // If we are a receiver process, receive the data from the root
  MPI_Recv(data, count, double, root, 0, communicator,
           MPI_COMM_WORLD);
}*/

}


void CrossValidation::setCurrnetDataSets(const bool &isOuter, const int &j) {

  int numTestData;
  int numTrainData;

  if (isOuter) {
    numTestData = NumTestDataOut[j];
    numTrainData = NumTrainDataOut[j];
    vecTestData.resize(numTestData);
    vecTrainData.resize(numTrainData);
    copy(testDataOut[j].begin(),  testDataOut[j].end(), vecTestData.begin());
    copy(trainDataOut[j].begin(), trainDataOut[j].end(), vecTrainData.begin());
  } else {
    numTestData = NumTestDataIn[j];
    numTrainData = NumTrainDataIn[j];
    vecTestData.resize(numTestData);
    vecTrainData.resize(numTrainData);
    copy(testDataIn[j].begin(),  testDataIn[j].end(), vecTestData.begin());
    copy(trainDataIn[j].begin(), trainDataIn[j].end(), vecTrainData.begin());
  }

  numTrainObs = numTrainData;

}


void CrossValidation::setOuterPartition() {

  vecRandObs.resize(numOrigObs);

  if (vecRandObs.size() > 0 && !readShuffledObs()) {
    for (int i=0; i < numOrigObs; ++i) vecRandObs[i]=i;
    if (shuffleObs() )  {
      srand (time(NULL));
      random_shuffle(vecRandObs.begin(), vecRandObs.end());
    }
  }

  if (writeShuffledObs()) {
    stringstream s;
    s << "randObsList" << '.' << problemName;
    ofstream os(s.str().c_str());
    for (int i=0; i<numOrigObs; ++i)
      os << vecRandObs[i] << " ";
    os.close();
  }

  if (getNumLimitedObs()<numOrigObs) {
    numOrigObs = getNumLimitedObs();
    vecRandObs.resize(numOrigObs);
  }

  NumOutPartition = 5;
  NumInPartition = 3;
  NumEachOutPartition = numOrigObs / NumOutPartition ;
  int remainderOut        = numOrigObs % NumOutPartition;

  if (innerCV()) setParamComb();

  testDataOut.resize(NumOutPartition);
  trainDataOut.resize(NumOutPartition);

  NumTestDataOut.resize(NumOutPartition);
  NumTrainDataOut.resize(NumOutPartition);

  if (innerCV()) {
    NumTestDataIn.resize(NumInPartition);
    NumTrainDataIn.resize(NumInPartition);
  }

  for (int i=0; i<NumOutPartition; ++i) {
    setTrainNTestData(i, remainderOut, numOrigObs,
            NumEachOutPartition, NumTrainDataOut[i], NumTestDataOut[i],
            trainDataOut[i], testDataOut[i], vecRandObs);

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
    DEBUGPRX(1, this,  "testDataOut: " << testDataOut[i]);
    DEBUGPRX(1, this,  "trainDataOut: " << trainDataOut[i]);
#ifdef ACRO_HAVE_MPI
  }
#endif //  ACRO_HAVE_MPI

  }

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
    DEBUGPRX(1, this, "vecRandObs: " << vecRandObs;);
#ifdef ACRO_HAVE_MPI
  }
  #endif //  ACRO_HAVE_MPI

  //delete[] vars;

}


void CrossValidation::setInnerPartition(int i) {

  NumEachInPartition = NumTrainDataOut[i] / NumInPartition;
  int remainderIn    = NumTrainDataOut[i] % NumInPartition;

  testDataIn.clear();
  trainDataIn.clear();

  testDataIn.resize(NumInPartition);
  trainDataIn.resize(NumInPartition);

  for (int j=0; j<NumInPartition; ++j) {

    setTrainNTestData(j, remainderIn, NumTrainDataOut[i], NumEachInPartition,
      NumTrainDataIn[j], NumTestDataIn[j], trainDataIn[j], testDataIn[j], trainDataOut[i]);

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
    DEBUGPRX(1, this, "testDataIn: "  << testDataIn[j] );
    DEBUGPRX(1, this, "trainDataIn: " << trainDataIn[j] );
#ifdef ACRO_HAVE_MPI
  }
#endif //  ACRO_HAVE_MPI

  }

}


void CrossValidation::setTrainNTestData(int i, int remain,
    int NumObs, int NumEachPart, int &NumTrainData, int &NumTestData,
    vector<int> &trainData, vector<int> &testData, vector<int> Data) {

  int l1=-1, l2=-1; 			// l1 and l2 for counters for trainData1 and testData1
  int lower, upper; 	// lower and upper for index for lower bound and upper bound of testData1

  NumTrainData = NumObs-NumEachPart;
  NumTestData = NumEachPart;

  if (remain !=0 ) {
    if (i < remain ) {
      NumTrainData--;
      NumTestData++;
    }
  }

  trainData.resize(NumTrainData);
  testData.resize(NumTestData);

  if (remain == 0) {
    lower = i*NumEachPart;
    upper = (i+1)*NumEachPart-1;
  } else if (remain != 0) {
    lower = i*(NumEachPart+1);
    upper = (i+1)*(NumEachPart+1)-1;
    if ( i == remain ) {
      upper--;
    } else if ( i > remain ) {
      lower = i*(NumEachPart+1) - (i-remain);
      upper = lower + NumEachPart -1;
    }
  }

  for (int p=0; p<NumObs; ++p) {
  if ( lower <= p && p <= upper )
    testData[++l2] = Data[p];
  else
    trainData[++l1] = Data[p];
  }
}


void CrossValidation::setParamComb() { // vector<ParamComb> &pc, int &NumComb

  (isLPBoost()) ? NumComb = 3 : NumComb = 5;

  vecParams.resize(NumComb);
  avgInTestErr.resize(NumComb);
  avgInTrainErr.resize(NumComb);

  double vecNu1[] = {.05, .1, .15 };
  double vecNu2[] = {.0001, .001, .01 };
  double vecCandidate[] = {0, .5, 1, 1.5, 2};

  for (int i=0; i<NumComb; ++i) {
    if (isLPBoost()) {
      if ( getExponentP()==1 ) {
        vecParams[i] =  1 / ( vecNu1[i] * NumEachOutPartition * (NumOutPartition-1) );  //vecD[i];
      } else {
        vecParams[i] =  1 / ( vecNu2[i] * NumEachOutPartition * (NumOutPartition-1) ); //vecD[i];
      }
    } else {
      vecParams[i] = vecCandidate[i];
    }

    DEBUGPRX(1, this, "m:" << i << " param=(" << vecParams[i] << ")\n");
  }

}


void CrossValidation::writePredictions(const int& isTest) {

  stringstream s;
  int obs, size;

  if (isTest) s << "predTest" << '.' << problemName;
  else        s << "predTrain" << '.' << problemName;

  ofstream os(s.str().c_str());
  // appending to its existing contents
  //ofstream os(s.str().c_str(), ofstream::app);

  os << "ActY  LPBR  ";
  if (compModels()) os << "AdaBoost  RandFore  ";
  os << " \n";

  (isTest) ? size = vecTestData.size() : size = vecTrainData.size();

  for (int i=0; i < size; ++i) {

    (isTest) ? obs = vecTestData[i] : obs = vecTrainData[i];
    os << origData[obs].y << "\t"  ;

    if (isTest) {
      os << lpbr->predTest[i] << "\t";
      if (compModels())
        os << lcm->predTest[0][i] << "\t" << lcm->predTest[1][i];
      os << "\n";
		} else {
      os << lpbr->predTrain[i] << "\t";
      if (compModels())
        os << lcm->predTrain[0][i] << "\t" << lcm->predTrain[1][i];
      os << "\n";
    }

  }

  os.close();
}


void CrossValidation::printOuterScore() {
  (isLPBoost()) ? ucout << "LPBR: " : ucout << "REPR: ";
  ucout << "\tAvg Test/Train Error:     "
    << avgOutTestErr / (double) NumOutPartition << "\t"
    << avgOutTrainErr / (double) NumOutPartition << "\n";

  if (compModels())
    if (isLPBoost()) lcm->printCompModelsCV();
    else             rcm->printCompModelsCV();

  (isLPBoost()) ? ucout << "LPBR: " : ucout << "REPR: ";
  ucout << "\tAvg Time:       " << avgTime/(double) NumOutPartition << "\n";
  if (innerCV()) {
    (isLPBoost()) ? ucout << "LPBR: " : ucout << "REPR: ";
    ucout << "\tAvg Time fold:  " << avgTimePerFold/(double) NumOutPartition << "\n";
  }
}


void CrossValidation::printInnerScore(int i) {

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

    for (int m=0; m<NumComb; ++m)
      ucout << "m: " << m << " Inner Avg Test/Train Errors: "
        << avgInTestErr[m] << "\t" << avgInTrainErr[m] << "\n";

  	ucout << "bestParamComb: " << bestParamComb
  	//	 << ", minAvgParamErr: " << minAvgParamMSE
  		 << ", param: " << vecParams[bestParamComb] << "\n";

#ifdef ACRO_HAVE_MPI
  }
#endif //  ACRO_HAVE_MPI
}


} // nemespace crossvalidation
