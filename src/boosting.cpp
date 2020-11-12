/*
 *  File name:   boosting.cpp
 *  Author:      Ai Kagawa
 *  Description: a source file for Boosting class
 */

#include "boosting.h"


namespace boosting {


  Boosting::Boosting(int& argc, char**& argv): rma(NULL), prma(NULL), isParallel(false) {

#ifdef ACRO_HAVE_MPI
    uMPI::init(&argc, &argv, MPI_COMM_WORLD);
#endif // ACRO_HAVE_M

    setup(argc, argv);     // setup all paramaters

    setData(argc, argv);   // set data

    (exactRMA()) ? greedyLevel=EXACT : greedyLevel=Greedy;

    if (exactRMA()) setupPebblRMA(argc, argv);  // setup RMA

    reset();

  }


  Boosting::~Boosting() {
#ifdef ACRO_HAVE_MPI
    if (isParallel) {
      CommonIO::end();
      uMPI::done();
    }
#endif // ACRO_HAVE_MPI
  };


  ///////////////////////// Set-up Boosting /////////////////////////

  // set data for Boosting class
  void Boosting::setData(int& argc, char**& argv) {
    data = new data::DataBoost(argc, argv, (BaseRMA *) this, (arg::ArgBoost *) this);
  }


  // set up PEBBL RMA
  void Boosting::setupPebblRMA(int& argc, char**& argv) {

#ifdef ACRO_HAVE_MPI
    int nprocessors = uMPI::size;
    /// Do parallel optimization if MPI indicates that we're using more than one processor
    if (parallel_exec_test<parallelBranching>(argc, argv, nprocessors)) {
      /// Manage parallel I/O explicitly with the utilib::CommonIO tools
      CommonIO::begin();
      CommonIO::setIOFlush(1);
      isParallel = true;
      prma     = new pebblRMA::parRMA(MPI_COMM_WORLD);
      rma      = prma;
    } else {
#endif // ACRO_HAVE_MPI
      rma = new pebblRMA::RMA;
#ifdef ACRO_HAVE_MPI
    }
#endif // ACRO_HAVE_MPI

    rma->setParameters(this); // passing arguments
    rma->setData(data);

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      rma->setSortedObsIdx(data->vecTrainData);
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    //exception_mngr::set_stack_trace(false);
    rma->setup(argc,argv);
    //exception_mngr::set_stack_trace(true);

  }


  // reset Boosting variables
  void Boosting::reset() {

    numBox     = 0;
    numRMASols = 0;
    NumObs     = data->numTrainObs;
    NumAttrib  = data->numAttrib;

    //vecDual.resize(NumObs);
    vecIsCovered.resize(NumObs);
    if (exactRMA()) rma->incumbentValue = getInf();

    matIntLower.clear();
    matIntUpper.clear();

    matOrigLower.clear();
    matOrigUpper.clear();

    if (evalEachIter()) {
      vecCoveredObsByBox.clear();
      vecCoveredObsByBox.resize(data->numOrigObs);
    }

  }


  ///////////////////////// Training methods /////////////////////////

  // training process of boosting
  void Boosting::train(const bool& isOuter, const int& NumIter, const int& greedyLevel) {

    int flagStop = 0;

    curIter=-1;

    setBoostingParameters();
    flagDuplicate=false;

    vecERMA.resize(NumIter);
    vecGRMA.resize(NumIter);

    try {

      //data->setStandDataY(data->origTrainData, data->standTrainData);					// standadize data for L1 regularization
      //data->setStandDataX(data->origTrainData, data->standTrainData);
      //data->integerizeData(data->origTrainData, data->intTrainData); 	// integerize features
#ifdef ACRO_HAVE_MPI
      if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
	data->standTrainData = data->origTrainData;
	// if (exactRMA()) rma->setData(data);
	setInitRMP();
	solveRMP();  //solveInitialMaster();
#ifdef ACRO_HAVE_MPI
      }
#endif //  ACRO_HAVE_MPI

      for (curIter=0; curIter<NumIter; ++curIter) { // for each column generation iteration

	//ucout << "\nColGen Iter: " << curIter << "\n";

        setDataWts();
        //if (saveWts())  // TODO: fix this later
        writeWts(curIter);

        solveRMA();

        if (exactRMA()) vecERMA[curIter] = rma->workingSol.value;
        vecGRMA[curIter] = grma->maxObjValue;

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

      //if ( evalFinalIter() && !(evalEachIter()) ) evaluateFinal();

      // clean up GUROBI for the next crossvalidation set
      //resetGurobi();

      writeGERMA();

    } catch(...) {
      ucout << "Exception during training" << "\n";
      return; // EXIT_FAILURE;
    } // end try ... catch

  } // trainData function


  void Boosting::writeGERMA() {

    stringstream s;
    s << "GERMA" << '_' << problemName;
    ofstream os(s.str().c_str());

    os << "iter\tGRMA\tERMA\n";
    for (int i=0; i<getIterations(); ++i )
      os << i << "\t" << vecGRMA[i] << "\t" << vecERMA[i] << "\n" ;

    os.close();

  }


  // call CLP to solve Master Problems
  void Boosting::solveRMP() {

    int i;

    DEBUGPR(10, cout <<  "Solve Restricted Master Problem!\n");

    tc.startTime();

    model.dual();
    if (debug>=1) model.writeMps("a.mps");

    vecPrimal = model.primalColumnSolution();
    vecDual   = model.dualRowSolution();

    DEBUGPR(10,
	    for (i=0; i<numCols; ++i) cout << vecPrimal[i];
	    for (i=0; i<numRows; ++i) cout << vecDual[i]; );

    primalVal = model.objectiveValue();
    //printCLPsolution();

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

      ucout << "Master Solution: " << primalVal << "\t"
            << " CPU Time: " << tc.getCPUTime() << "\n";

      //printRMPSolution();

#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    if (evalEachIter()) {
      for (i=numBox-numRMASols; i<numBox; ++i) {
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


  void Boosting::solveRMA() {

    if (exactRMA()) {

      resetExactRMA();

      if (BaseRMA::initGuess()) {
#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
	  solveGreedyRMA();
	  rma->setInitialGuess(grma->isPosIncumb, grma->maxObjValue,
			       grma->L, grma->U);
#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI
      }
      solveExactRMA();
    } else {
#ifdef ACRO_HAVE_MPI
      if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
	solveGreedyRMA();
#ifdef ACRO_HAVE_MPI
      }
#endif //  ACRO_HAVE_MPI
    }
  }


  // TODO: make reset function for GreedyRMA
  void Boosting::solveGreedyRMA() {
    grma = new greedyRMA::GreedyRMA((BaseRMA *) this, (DataRMA *) data);
    grma->runGreedyRangeSearch();
  }


  void Boosting::resetExactRMA() {

#ifdef ACRO_HAVE_MPI
    if (isParallel) {
      prma->reset();
      if (printBBdetails()) prma->printConfiguration();
      CommonIO::begin_tagging();
    } else {
#endif //  ACRO_HAVE_MPI
      rma->reset();
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    rma->mmapCachedCutPts.clear();
    rma->workingSol.value = -getInf();
    //rma->numDistObs       = data->numTrainObs;	    // only use training data
    //rma->setSortedObsIdx(data->vecTrainData);

  }


  void Boosting::solveExactRMA() {

    rma->resetTimers();
    InitializeTiming();

    tc.startTime();

    if (BaseRMA::printBBdetails()) rma->solve();  // print out B&B details
    else                           rma->search();

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      tc.getCPUTime();
      tc.getWallTime();
      printRMASolutionTime();
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    // CommonIO::end();
    // uMPI::done();

  } // end function solveExactRMA()


  // set original lower and upper bounds matrices
  void Boosting::setOriginalBounds() {

    multimap<int, double>::iterator it;
    double lower, upper, tmpLower, tmpUpper;
    matOrigLower.resize(matIntLower.size());
    matOrigUpper.resize(matIntUpper.size());

    for (unsigned int k = matIntLower.size()-numRMASols; k<matIntLower.size(); ++k) {

      matOrigLower[k].resize(NumAttrib);
      matOrigUpper[k].resize(NumAttrib);

      for (int j=0; j<NumAttrib; ++j) { // for each attribute

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

	} else lower=-getInf(); // if matIntLower[k][j] < 0 and matIntLower[k][j] != rma->distFeat[j]

	if ( matIntUpper[k][j] < data->distFeat[j] ) { // upperBound

	  tmpLower = getUpperBound(k, j, 0, true);
	  tmpUpper =  getLowerBound(k, j, 1, true);
	  upper = (tmpLower + tmpUpper) / 2.0;

	  DEBUGPR(10, cout << "(k,j): (" << k << ", " << j
		  << ") matIntUpper[k][j]: " << matIntUpper[k][j]
		  << " LeastUpper: " << tmpLower << "\n"
		  << " matIntUpper[k][j]+1: " << matIntUpper[k][j]+1
		  << " GreatestUpper: " << tmpUpper << "\n");

	} else upper=getInf(); // if matIntUpper[k][j] < rma->distFeat[j] and matIntUpper[k][j] != 0

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
    return data->vecFeature[j].vecIntMinMax[boundVal+value].minOrigVal;
  }


  double Boosting::getUpperBound(int k, int j, int value, bool isUpper)  {
    int boundVal;
    // double max = -getInf();
    if (isUpper) boundVal = matIntUpper[k][j];
    else         boundVal = matIntLower[k][j];
    return data->vecFeature[j].vecIntMinMax[boundVal+value].maxOrigVal;
  }


  void Boosting::printRMASolutionTime() {
    ucout << "ERMA Solution: " << rma->workingSol.value
          << "\tCPU time: "    << tc.getCPUTime()
          << "\tNum of Nodes: " << rma->subCount[2]
          << "\n";
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

    errTrain = evaluateEachIter(TRAIN, data->origTrainData);
    errTest  = evaluateEachIter(TEST,  data->origTestData);
    //if (isLPBoost() && printBoost()) printEachIterAllErrs();

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      (isOuter) ? ucout << "Outer " : ucout << "Inner ";
      ucout << "Iter: " << curIter+1 << " ";
      ucout << "Test/Train Errors: " << errTest << " " << errTrain ;
      if (isLPBoost()) ucout << " " << "rho: "<< vecPrimal[NumObs] << "\n";
      else           ucout << "\n";
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

  }


  void Boosting::evaluateFinal() {

    errTrain = evaluateAtFinal(TRAIN, data->origTrainData);
    errTest  = evaluateAtFinal(TEST,  data->origTestData);

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
    int obs;
    if (printBoost()) ucout << "vecCoveredObsByBox Train:\n";
    for (unsigned int i=0; i<data->vecTrainData.size(); ++i) {
      obs = data->vecTrainData[i];
      vecCoveredObsByBox[obs].resize(numBox);
      vecCoveredObsByBox[obs][numBox-1] = vecIsCovered[i];
      if (printBoost()) {
	for (int j=0; j<numBox; ++j)
	  ucout << vecCoveredObsByBox[obs][j] << " ";
	ucout << "\n";
      }
    }

  }


  // set covered test observations
  void Boosting::setCoveredTestObs() {

    int obs;
    if (printBoost()) ucout << "\nvecCoveredObsByBox Test:\n";
    for (unsigned int i=0; i<data->vecTestData.size(); ++i) { // for each test dataset
      obs = data->vecTestData[i];
      vecCoveredObsByBox[obs].resize(numBox);

      for (int j=0; j<NumAttrib; ++j) { // for each attribute

	if (matOrigLower[numBox-1][j] <= data->origTestData[obs].X[j] &&
	    data->origTestData[obs].X[j] <= matOrigUpper[numBox-1][j]  ) {
	  if ( j==NumAttrib-1) { // all features are covered by the box
	    vecCoveredObsByBox[obs][numBox-1] = true;
	  }
	} else {
	  vecCoveredObsByBox[obs][numBox-1] = false;
	  break; // this observation is not covered
	}

      } // end for each attribute, j

      if (printBoost()) {
	for (int j=0; j<numBox; ++j)
	  ucout << vecCoveredObsByBox[obs][j] << " ";
	ucout << "\n";
      }

    } // for each test dataset

  } // setCoveredTestObs function


  void Boosting::writeWts(const int& curIter) {

    int obs;

    //mkdir("./wts")

    stringstream s;
    s << "./wts/wt_" << problemName << "_" << curIter;

    ofstream os(s.str().c_str());
    for (int i=0; i < NumObs ; ++i) {
      obs = data->vecTrainData[i];
      os << data->intTrainData[obs].w << ", ";
    }

  }



  void Boosting::writePredictions(const int& isTest, vector<DataXy> origData) {

    stringstream s;
    int obs, size;

    if (isTest) s << "predictionTest" << '.' << problemName;
    else        s << "predictionTrain" << '.' << problemName;

    ofstream os(s.str().c_str());
    // appending to its existing contents
    //ofstream os(s.str().c_str(), ofstream::app);

    os << "ActY \t Boosting  \n";

    (isTest) ? size = data->vecTestData.size() : size = data->vecTrainData.size();
    for (int i=0; i < size; ++i) {
      (isTest) ? obs = data->vecTestData[i] : obs = data->vecTrainData[i];
      if (isTest) os << origData[obs].y << " " << predTest[i] << "\n";
      else        os << origData[obs].y << " " << predTrain[i] << "\n";
    }

    os.close();
  }

  //////////////////////// Checking methods ///////////////////////

  bool Boosting::isDuplicate() {

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      if ( matIntUpper.size()==1 ) return false;

      for (int j=0; j<NumAttrib; ++j)
	if ( matIntLower[matIntLower.size()-2][j] != sl[0]->a[j]
	     || matIntUpper[matIntUpper.size()-2][j] != sl[0]->b[j] )
	  return false;
      ucout << "Duplicate Solution Found!! \n" ;
      flagDuplicate = true;
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    return true;

  }


  void Boosting::checkObjValue(vector<DataXw> intData) {
    int obs;
    double wt=0.0;

    for (int i=0; i<NumObs; i++) { // for each training data
      obs = data->vecTrainData[i];
      for (int f=0; f<NumAttrib; ++f) { // for each feature
    	if ( (grma->L[f] <= intData[obs].X[f])
	     && (intData[obs].X[f] <= grma->U[f]) ) {
	  if (f==NumAttrib-1)  // if this observation is covered by this solution
	    wt+=intData[obs].w;   //dataWts[i];
	} else break; // else go to the next observation
      }  // end for each feature
    }  // end for each training observation

    cout << "grma->L: " << grma->L ;
    cout << "grma->U: " << grma->U ;
    cout << "GRMA ObjValue=" << wt << "\n";
  }


  void Boosting::checkObjValue(int k, vector<DataXw> intData) {
    int    obs;
    double wt=0.0;

    for (int i=0; i<NumObs; i++) { // for each training data
      obs = data->vecTrainData[i];
      for (int f=0; f<NumAttrib; ++f) { // for each feature
    	if ( (matIntLower[k][f] <= intData[obs].X[f])
	     && (intData[obs].X[f] <= matIntUpper[k][f]) ) {
	  if (f==NumAttrib-1)  // if this observation is covered by this solution
	    wt+=intData[obs].w;   //dataWts[i];
	} else break; // else go to the next observation
      }  // end for each feature
    }  // end for each training observation

    cout << "matIntLower[k]: " << matIntLower[k] ;
    cout << "matIntUpper[k]: " << matIntUpper[k] ;
    cout << "RMA ObjValue=" << wt << "\n";
  }


} // namespace boosting
