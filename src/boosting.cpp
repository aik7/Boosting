/*
 *  File name:   boosting.cpp
 *  Author:      Ai Kagawa
 *  Description: a source file for Boosting class
 */

#include "boosting.h"


namespace boosting {

  ///////////////////////// Set-up Boosting /////////////////////////

  Boosting::Boosting(int& argc, char**& argv) {

    setup(argc, argv);     // setup all paramaters

    setData(argc, argv);   // set data from SolveRMA class

    (isPebblRMA()) ? greedyLevel=EXACT : greedyLevel=Greedy;

    if (isPebblRMA()) setupPebblRMA(argc, argv);  // setup PEBBL RMA

    reset(); // reste Boosting

    if (prma!=NULL) prma->printConfiguration(); // TODO: do not know why...

  } // end Boosting constructor


  // reset Boosting variables
  void Boosting::reset() {

    numBoxesSoFar = 0;                  // # of boxes
    numBoxesIter  = 0;                  // # of RMA solutions
    numObs        = data->numTrainObs;  // # of training observations
    numAttrib     = data->numAttrib;    // # of attributes or features

    //vecDualVars.resize(numObs);
    vecIsCovered.resize(numObs);     // a vector indicate each observation is covered or not
    if (isPebblRMA()) rma->incumbentValue = -getInf();  // set the pebbl RMA incumbent value to be negative infinity

    matIntLower.clear();    // matrix containes lower bound of box in integerized value
    matIntUpper.clear();    // matrix containes upper bound of box in integerized value

    matOrigLower.clear();   // matrix containes lower bound of box in original value
    matOrigUpper.clear();   // matrix containes upper bound of box in original value

    if (isEvalEachIter()) { // evaluate each iteration
      vecCoveredObsByBox.clear();
      vecCoveredObsByBox.resize(data->numOrigObs);
    } // end if eacluate each iteration

  } // end reset function


  ///////////////////////// Training methods /////////////////////////

  // training process of boosting
  void Boosting::train(const bool& isOuter,
                       const unsigned int& numIter,
                       const unsigned int& greedyLevel) {

    int flagStop = 0;

    curIter=-1;

    setBoostingParameters();  // set Boosting parameters

    vecERMAObjVal.resize(numIter);
    vecGRMAObjVal.resize(numIter);

    try {

      // standadize data for L1 regularization
      data->setDataStandY();  // set data->dataStandTrain
      data->setDataStandX();

      // TODO: thiw was done at RMA constructor
      // integerize data for RMA
      // data->integerizeEpsData();  // set data->dataIntTrain

#ifdef ACRO_HAVE_MPI
      if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
        // if (isPebblRMA()) rma->setData(data);
        setInitRMP();  // set the initial RMP
        solveRMP();    // set the RMP
#ifdef ACRO_HAVE_MPI
      }
#endif //  ACRO_HAVE_MPI

      for (curIter=0; curIter<numIter; ++curIter) { // for each column generation iteration

        //ucout << "\nColGen Iter: " << curIter << "\n";

        setDataWts();
        //if (saveWts())  // TODO: fix this later
        saveWts(curIter);

        solveRMA();

        if (isPebblRMA()) vecERMAObjVal[curIter] = rma->workingSol.value;
        vecGRMAObjVal[curIter] = grma->getObjVal();

#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

          if (isStoppingCondition()) flagStop = 1;

          // If we are the root process, send our data to everyone
          for (int k = 0; k < uMPI::size; ++k)
            if (k != 0)
              MPI_Send(&flagStop, 1, MPI_INT, k, 0, MPI_COMM_WORLD);

          if (flagStop==1)  break;

          insertColumns(); // add RMA solutions and check duplicate

          //setOriginalBounds();  // map back from the discretized data into original

          solveRMP();

#ifdef ACRO_HAVE_MPI
	} else {
#endif //  ACRO_HAVE_MPI

          if ((uMPI::rank!=0)) {
            // If we are a receiver process, receive the data from the root
            MPI_Recv(&flagStop, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (flagStop==1) break;
          }

#ifdef ACRO_HAVE_MPI
        }
#endif //  ACRO_HAVE_MPI

      } // end for each column generation iteration

      //printBoostingErr();

      //if ( evalFinalIter() && !(isEvalEachIter()) ) evaluateFinal();

      saveGERMAObjVals();

    } catch(...) {
      ucout << "Exception during training" << "\n";
      return; // EXIT_FAILURE;
    } // end try ... catch

#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
    saveModel();
#ifdef ACRO_HAVE_MPI
  }
#endif //  ACRO_HAVE_MPI

  } // end trainData function


  // save both greedy and PEBBL RMA solutions for all Boosting iterations
  void Boosting::saveGERMAObjVals() {

    // set the output file name
    stringstream s;
    s << "GERMA_" << problemName << ".out";
    ofstream os(s.str().c_str());

    os << "iter\tGRMA\tERMA\n";
    for (unsigned int i=0; i<getNumIterations(); ++i ) // for each iteration
      os << i << "\t" << vecGRMAObjVal[i] << "\t" << vecERMAObjVal[i] << "\n" ;

    os.close();

} // end saveGERMAObjVals function


  // call CLP to solve Master Problems
  void Boosting::solveRMP() {

    unsigned int i;

    DEBUGPR(10, cout <<  "Solve Restricted Master Problem!\n");

    tc.startTime();

    model.dual();
    if (debug>=1) model.writeMps("clp.mps");

    vecPrimalVars = model.primalColumnSolution();
    vecDualVars   = model.dualRowSolution();

    DEBUGPR(10,
	    for (i=0; i<numCols; ++i) cout << vecPrimalVars[i];
	    for (i=0; i<numRows; ++i) cout << vecDualVars[i]; );

    primalSol = model.objectiveValue();
    //printCLPsolution();

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

      std::cout << std::fixed << std::setprecision(4)
            << "Master Solution: " << primalSol << "\t";
      std::cout << std::fixed << std::setprecision(2)
            << " CPU Time: " << tc.getCPUTime() << "\n";

      DEBUGPR(1, printRMPSolution());

#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    if (isEvalEachIter()) {
      for (i=numBoxesSoFar-numBoxesIter; i<numBoxesSoFar; ++i) {
      	setCoveredTrainObs();
      	setCoveredTestObs();
      }
      //ucout << "Iter: " << curIter+1 << "\t";
      evaluateEach();
    }

  } // end function Boosting::solveMaster()


  // print function for CLP solutions
  void Boosting::printCLPsolution() {

    // Print column solution
    int numberColumns = model.numberColumns();

    // Alternatively getColSolution()
    double * columnPrimal = model.primalColumnSolution();
    // Alternatively getReducedCost()
    double * columnDual = model.dualColumnSolution();
    // Alternatively getColLower()
    double * columnLower = model.columnLower();
    // Alternatively getColUpper()
    double * columnUpper = model.columnUpper();
    // Alternatively getObjCoefficients()
    double * columnObjective = model.objective();

    int iColumn;

    std::cout << "               Primal          Dual         Lower         Upper          Cost"
              << std::endl;

    for (iColumn = 0; iColumn < numberColumns; iColumn++) {
      double value;
      std::cout << std::setw(6) << iColumn << " ";
      value = columnPrimal[iColumn];
      if (fabs(value) < 1.0e5)
	std::cout << std::setiosflags(std::ios::fixed | std::ios::showpoint) << std::setw(14) << value;
      else
	std::cout << std::setiosflags(std::ios::scientific) << std::setw(14) << value;
      value = columnDual[iColumn];
      if (fabs(value) < 1.0e5)
	std::cout << std::setiosflags(std::ios::fixed | std::ios::showpoint) << std::setw(14) << value;
      else
	std::cout << std::setiosflags(std::ios::scientific) << std::setw(14) << value;
      value = columnLower[iColumn];
      if (fabs(value) < 1.0e5)
	std::cout << std::setiosflags(std::ios::fixed | std::ios::showpoint) << std::setw(14) << value;
      else
	std::cout << std::setiosflags(std::ios::scientific) << std::setw(14) << value;
      value = columnUpper[iColumn];
      if (fabs(value) < 1.0e5)
	std::cout << std::setiosflags(std::ios::fixed | std::ios::showpoint) << std::setw(14) << value;
      else
	std::cout << std::setiosflags(std::ios::scientific) << std::setw(14) << value;
      value = columnObjective[iColumn];
      if (fabs(value) < 1.0e5)
	std::cout << std::setiosflags(std::ios::fixed | std::ios::showpoint) << std::setw(14) << value;
      else
	std::cout << std::setiosflags(std::ios::scientific) << std::setw(14) << value;

      std::cout << std::endl;
    }
  }


  // call insert column in sub-class
  void Boosting::insertColumns() {
    // greedyLevel=EXACT; // TODO: fix this later
    (greedyLevel==EXACT) ? insertExactColumns() : insertGreedyColumns();
  }


  // set original lower and upper bounds matrices
  void Boosting::setOriginalBounds() {

    multimap<int, double>::iterator it;
    double lower, upper, tmpLower, tmpUpper;
    matOrigLower.resize(matIntLower.size());
    matOrigUpper.resize(matIntUpper.size());

    // for each distinct value
    for (unsigned int k = matIntLower.size()-numBoxesIter;
         k<matIntLower.size(); ++k) {

      matOrigLower[k].resize(numAttrib);
      matOrigUpper[k].resize(numAttrib);

      for (unsigned int j=0; j<numAttrib; ++j) { // for each attribute

        ///////////////////////////// mid point rule //////////////////////////////
        if ( matIntLower[k][j] > 0 ) { // lowerBound

          tmpLower =  getUpperBound(k, j, -1, false);
          tmpUpper =  getLowerBound(k, j, 0, false);
          lower =  (tmpLower + tmpUpper) / 2.0;

          DEBUGPR(10, cout << "(k,j): (" << k << ", " << j
                  << ") matIntLower[k][j]-1: " << matIntLower[k][j]-1
                  << " LeastLower: " << tmpLower << "\n"
                  << " matIntLower[k][j]: " << matIntLower[k][j]
                  << " GreatestLower: " << tmpUpper << "\n");

        } else lower=-getInf(); // if matIntLower[k][j] < 0 and matIntLower[k][j] != rma->vecNumDistVals[j]

        if ( matIntUpper[k][j] < data->vecNumDistVals[j] ) { // upperBound

          tmpLower = getUpperBound(k, j, 0, true);
          tmpUpper =  getLowerBound(k, j, 1, true);
          upper = (tmpLower + tmpUpper) / 2.0;

          DEBUGPR(10, cout << "(k,j): (" << k << ", " << j
                  << ") matIntUpper[k][j]: " << matIntUpper[k][j]
                  << " LeastUpper: " << tmpLower << "\n"
                  << " matIntUpper[k][j]+1: " << matIntUpper[k][j]+1
                  << " GreatestUpper: " << tmpUpper << "\n");

        } else upper=getInf(); // if matIntUpper[k][j] < rma->vecNumDistVals[j] and matIntUpper[k][j] != 0

        // store values
        matOrigLower[k][j]=lower;
        matOrigUpper[k][j]=upper;

      } // end for each attribute, j

    }

    DEBUGPR(10, cout << "Integerized Lower: \n" << matIntLower );
    DEBUGPR(10, cout << "Integerized Upper: \n" << matIntUpper );
    DEBUGPR(10, cout << "Real Lower: \n" << matOrigLower );
    DEBUGPR(10, cout << "Real Upper: \n" << matOrigUpper );

  } // end function REPR::setOriginalBounds()


  double Boosting::getLowerBound(int k, int j, int value, bool isUpper) {
    int boundVal;
    // double min = getInf();
    if (isUpper) boundVal = matIntUpper[k][j];
    else         boundVal = matIntLower[k][j];
    return data->vecAttribIntInfo[j].vecBins[boundVal+value].lowerBound;
  }


  double Boosting::getUpperBound(int k, int j, int value, bool isUpper)  {
    int boundVal;
    // double max = -getInf();
    if (isUpper) boundVal = matIntUpper[k][j];
    else         boundVal = matIntLower[k][j];
    return data->vecAttribIntInfo[j].vecBins[boundVal+value].upperBound;
  }



  void Boosting::printIterInfo() {
#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      if (isOuter) ucout << "Outer Iter: " << curIter+1;
      else 				 ucout << "Inner Iter: " << curIter+1;
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI
  }


  void Boosting::printBoostingErr() {

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      if (isOuter) ucout << "OutREPR ";
      else         ucout << "InnREPR ";
      ucout << curIter << ":\tTest/Train Errors: " << errTest << " " << errTrain << "\n";
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

  }

  void Boosting::evaluateEach() {

    errTrain = evaluateEachIter(TRAIN, data->dataOrigTrain);
    errTest  = evaluateEachIter(TEST,  data->dataOrigTest);
    //if (!isREPR() && isPrintBoost()) printEachIterAllErrs();

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      (isOuter) ? ucout << "Outer " : ucout << "Inner ";
      ucout << "Iter: " << curIter+1 << " ";
      ucout << "Test/Train Errors: " << errTest << " " << errTrain ;
      if (!isREPR()) ucout << " " << "rho: "<< vecPrimalVars[numObs] << "\n";
      else           ucout << "\n";
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

  }


  void Boosting::evaluateFinal() {

    errTrain = evaluateAtFinal(TRAIN, data->dataOrigTrain);
    errTest  = evaluateAtFinal(TEST,  data->dataOrigTest);

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      (isOuter) ? ucout << "Outer " : ucout << "Inner ";
      ucout << "Iter: " << curIter+1 << " ";
      ucout << "Test/Train Errors: " << errTest << " " << errTrain << "\n";
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI


  }


  // set covered train observations
  void Boosting::setCoveredTrainObs() {

    if (isPrintBoost()) ucout << "vecCoveredObsByBox Train:\n";

    for (unsigned int i=0; i<data->vecTrainObsIdx.size(); ++i) {

      vecCoveredObsByBox[i].resize(numBoxesSoFar);
      vecCoveredObsByBox[i][numBoxesSoFar-1] = vecIsCovered[i];

      if (isPrintBoost()) {
        for (unsigned int j=0; j<numBoxesSoFar; ++j)
          ucout << vecCoveredObsByBox[i][j] << " ";
        ucout << "\n";
      }

    } // end each train observation

  } // end setCoveredTrainObs function


  // set covered test observations
  void Boosting::setCoveredTestObs() {

    unsigned int j, obs;

    if (isPrintBoost()) ucout << "\nvecCoveredObsByBox Test:\n";
    for (unsigned int i=0; i<data->vecTestObsIdx.size(); ++i) { // for each test dataset
      obs = data->vecTestObsIdx[i];
      vecCoveredObsByBox[obs].resize(numBoxesSoFar);

      for (j=0; j<numAttrib; ++j) { // for each attribute

        if (matOrigLower[numBoxesSoFar-1][j] <= data->dataOrigTest[obs].X[j] &&
            data->dataOrigTest[obs].X[j] <= matOrigUpper[numBoxesSoFar-1][j]  ) {
          if (j==numAttrib-1) { // all features are covered by the box
            vecCoveredObsByBox[obs][numBoxesSoFar-1] = true;
          }
        } else {
          vecCoveredObsByBox[obs][numBoxesSoFar-1] = false;
          break; // this observation is not covered
        }

      } // end for each attribute, j

      if (isPrintBoost()) {
        for (j=0; j<numBoxesSoFar; ++j)
          ucout << vecCoveredObsByBox[obs][j] << " ";
        ucout << "\n";
      }

    } // for each test dataset

  } // setCoveredTestObs function


  void Boosting::saveWts(const unsigned int& curIter) {

    // TODO if this directory exists, create the directory
    //mkdir("./wts")

    stringstream s;
    s << "./wts/wt_" << problemName << "_" << curIter;
    ofstream os(s.str().c_str());

    for (unsigned int i=0; i < numObs ; ++i)
      os << data->dataIntTrain[idxTrain(i)].w << ", ";

  } // end saveWts function


  void Boosting::savePredictions(const bool &isTest, vector<DataXy> origData) {

    stringstream s;
    unsigned int obs, size;

    if (isTest) s << "predictionTest" << '.' << problemName;
    else        s << "predictionTrain" << '.' << problemName;

    ofstream os(s.str().c_str());
    // appending to its existing contents
    //ofstream os(s.str().c_str(), ofstream::app);

    os << "ActY \t Boosting  \n";

    (isTest) ? size = data->vecTestObsIdx.size() : size = data->vecTrainObsIdx.size();

    for (unsigned int i=0; i < size; ++i) { // for each observation
      (isTest) ? obs = data->vecTestObsIdx[i] : obs = data->vecTrainObsIdx[i];
      if (isTest) os << origData[obs].y << " " << predTest[i] << "\n";
      else        os << origData[obs].y << " " << predTrain[i] << "\n";
    } // end for each observation

    os.close();
  } // for savePredictions function

  //////////////////////// Checking methods ///////////////////////

  // check wether or not the "k"-th box in the current iteration is duplicated
  bool Boosting::isDuplicate() {

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

      // if the cuurent box is only one, no duplicates
      if ( matIntUpper.size()==1 )
        return false;

      // for each box in this iteration
      for (unsigned int k=0; k < numBoxesIter; ++k) {
        // for each boxes already in model
        for (unsigned int l=0; l < numBoxesSoFar; ++l)
          for (unsigned int j=0; j<numAttrib; ++j) // for each attribute
            // if no duplicates boxes
            if ( matIntLower[l][j] != sl[k]->a[j]
                 || matIntUpper[l][j] != sl[k]->b[j] )
              return false;
      ucout << "Duplicate Solution Found!! \n" ;
      }

#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    return true;

  } // end isDuplicate function


  void Boosting::checkObjValue(vector<DataXw> intData) {

    int obs;
    double wt=0.0;

    for (unsigned int i=0; i<numObs; i++) { // for each training data
      obs = data->vecTrainObsIdx[i];
      for (unsigned int j=0; j<numAttrib; ++j) { // for each feature
        if ( (grma->getLowerBounds()[j] <= intData[obs].X[j])
             && (intData[obs].X[j] <= grma->getUpperBounds()[j]) ) {
          if (j==numAttrib-1)  // if this observation is covered by this solution
            wt+=intData[obs].w;   //dataWts[i];
        } else break; // else go to the next observation
      }  // end for each feature
    }  // end for each training observation

    cout << "grma->L: " << grma->getLowerBounds() ;
    cout << "grma->U: " << grma->getLowerBounds() ;
    cout << "GRMA ObjValue=" << wt << "\n";
  }


  // check objective value in a brute force way (k: k-th box)
  void Boosting::checkObjValue(const unsigned int &k, vector<DataXw> intData) {

    unsigned int obs;
    double wt = 0.0;

    for (unsigned int i=0; i<numObs; i++) { // for each training data

      for (unsigned int j=0; j<numAttrib; ++j) { // for each attribute

        // if this solution is covered by the current lower and upper bound
        if ( (matIntLower[k][j] <= intData[idxTrain(i)].X[j])
             && (intData[idxTrain(i)].X[j] <= matIntUpper[k][j]) ) {

          // if this observation is covered by this solution
          // if this observation satisfies all attributes' constraints
          if (j==numAttrib-1)  // TODO: specify obs!
            wt+=intData[obs].w;   // add weights

        } else break; // else go to the next observation

      }  // end for each attribute

    }  // end for each training observation

    cout << "matIntLower[k]: " << matIntLower[k] ;
    cout << "matIntUpper[k]: " << matIntUpper[k] ;
    cout << "RMA ObjValue=" << wt << "\n";

  } // end checkObjValue function


} // namespace boosting
