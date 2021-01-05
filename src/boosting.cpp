/*
 *  Author:      Ai Kagawa
 *  Description: a source file for Boosting class
 */

#include "boosting.h"


namespace boosting {

  ///////////////////////// Set-up Boosting /////////////////////////

  Boosting::Boosting(int& argc, char**& argv) {

    setup(argc, argv);     // setup all paramaters from PEBBL

    setData(argc, argv);   // set Data RMA class object from SolveRMA class

    (isPebblRMA()) ? greedyLevel=EXACT : greedyLevel=Greedy;

    if (isPebblRMA()) setupPebblRMA(argc, argv);  // setup PEBBL RMA

    resetBoosting(); // reset Boosting

    if (prma!=NULL) prma->printConfiguration(); // TODO: do not know why...

  } // end Boosting constructor


  // reset Boosting variables
  void Boosting::resetBoosting() {

    numBoxesSoFar = 0;                  // # of boxes
    numBoxesIter  = 0;                  // # of RMA solutions
    numObs        = data->numOrigObs;   // # of original observations
    numAttrib     = data->numAttrib;    // # of attributes or features

    if (isPebblRMA()) rma->incumbentValue = -getInf();  // set the pebbl RMA incumbent value to be negative infinity

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

    // vecDualVars = new double [numObs];
    // vecDualVars.resize(numObs);
    // vecIsCovered.resize(numObs);     // a vector indicate each observation is covered or not

    matIntLower.clear();    // matrix containes lower bound of box in integerized value
    matIntUpper.clear();    // matrix containes upper bound of box in integerized value

    matOrigLower.clear();   // matrix containes lower bound of box in original value
    matOrigUpper.clear();   // matrix containes upper bound of box in original value

    // if (isEvalEachIter()) { // evaluate each iteration
    matIsCvdObsByBox.clear();
    // matIsCvdObsByB    ox.resize(numObs);
    // } // end if eacluate each iteration

    // if the isSaveAllRMASols is enabled,
    // allocate vecERMAObjVal and vecGRMAObjVal
    if (isSaveAllRMASols())  resetVecRMAObjVals();

#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

  } // end reset function


  ///////////////////////// Training methods /////////////////////////

  // training process of boosting
  void Boosting::train(const unsigned int& numIter,
                       const unsigned int& greedyLevel) {

    int isStopCond = 0;

    curIter=-1;

    setBoostingParameters();  // set Boosting parameters

    try {

      // use rank 1 to solve RMP
#ifdef ACRO_HAVE_MPI
      if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

        // standadize data for L1 regularization
        if (isStandData()) {
          data->setDataStandY();  // set data->dataStandTrain
          data->setDataStandX();
        } else { // not to standerdise for debugging purpose
          data->dataStandTrain = data->dataOrigTrain;
        }

        setInitRMP();  // set the initial RMP

        solveRMP();    // solve the RMP

        // if the option for evaluating each iteration is enabled
        // evaluate the model
        if (isEvalEachIter()) {
          errTrain = evaluateEachIter(TRAIN, data->dataOrigTrain);
          errTest  = evaluateEachIter(TEST,  data->dataOrigTest);
        }

#ifdef ACRO_HAVE_MPI
      }
#endif //  ACRO_HAVE_MPI

      for (curIter=0; curIter<numIter; ++curIter) { // for each column generation iteration

        //ucout << "\nColGen Iter: " << curIter << "\n";

        setWeights();  // set data weight in DataRMA class

        solveRMA();    // solve the subproblem of RMA

        if (greedyLevel==EXACT) setPebblRMASolutions();

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

        // save the weights for the current iteration
        if (isSaveWts()) saveWeights(curIter);

        if (isSaveAllRMASols()) setVecRMAObjVals();

        if (isStoppingCondition()) isStopCond = 1;

  #ifdef ACRO_HAVE_MPI
        // If we are the root process, send our data to everyone
        for (int k = 0; k < uMPI::size; ++k)
          if (k != 0)
            MPI_Send(&isStopCond, 1, MPI_INT, k, 0, MPI_COMM_WORLD);
    } // end if (uMPI::rank==0)

    if (uMPI::rank==0) {
  #endif //  ACRO_HAVE_MPI

          if (isStopCond==1) break;

          insertColumns(); // add RMA solutions and check duplicate

          // map back from the discretized data into original
          if (delta()!=-1) setOriginalBounds();

          solveRMP();

          if (isEvalEachIter()) { // if evalute the model each iteration

            errTrain = evaluateEachIter(TRAIN, data->dataOrigTrain);
            errTest  = evaluateEachIter(TEST,  data->dataOrigTest);

            printBoostingErr();

          } // end if evalute the model each iteration

#ifdef ACRO_HAVE_MPI
  } else { // if MPI rank is not 0, receive the info about the stopping condition

          // If we are a receiver process, receive the data from the root
          MPI_Recv(&isStopCond, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

          if (isStopCond==1) break;

  } // save if MPI rank is not 0
#endif //  ACRO_HAVE_MPI

      } // end for each column generation iteration

    } catch(...) {
      ucout << "Exception during training" << "\n";
      return; // EXIT_FAILURE;
    } // end try ... catch

#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

    // if eavluating the final iteration option is enabled
    // and  eavluating the final iteration option is disabled
    if ( isEvalFinalIter() && !isEvalEachIter() ) {

      // get train and tes errors
      errTrain = evaluateAtFinal(TRAIN, data->dataOrigTrain);
      errTest  = evaluateAtFinal(TEST,  data->dataOrigTest);

      printBoostingErr();

    } // end if

    saveGERMAObjVals();   // save Greedy and PEBBL RMA solutions for each iteration

    saveModel();          // save Boosting model

#ifdef ACRO_HAVE_MPI
        }
#endif //  ACRO_HAVE_MPI

  } // end trainData function


  // call CLP to solve Master Problems
  void Boosting::solveRMP() {

    DEBUGPR(10, cout <<  "Solve Restricted Master Problem!\n");

    tc.startTime();

    model.primal();
    // model.dual();  //  invoke the dual simplex method.

    primalSol     = model.objectiveValue();        // get objective value
    vecPrimalVars = model.primalColumnSolution();  // ger primal variables
    vecDualVars   = model.dualRowSolution();       // dualColumnSolution(); 

    // if (debug>=0) printCLPsolution();

    cout << fixed << setprecision(4)
          << "Master Solution: " << primalSol << "\t";
    cout << fixed << setprecision(2)
          << " CPU Time: " << tc.getCPUTime() << "\n";

    if (debug>=0) {
      int iter;
      char clpFile[12];
      strcpy(clpFile, "clp_");
      iter = (curIter>getNumIterations()) ? -1 : curIter;
      strcat(clpFile, to_string(iter).c_str());
      strcat(clpFile, ".mps");
      model.writeMps(clpFile);
    }

    if (debug>=10) {

      unsigned int i;

	    for (i=0; i<numCols; ++i) cout << vecPrimalVars[i];
	    for (i=0; i<numRows; ++i) cout << vecDualVars[i];

    }

    DEBUGPR(1, printRMPCheckInfo());

  } // end function Boosting::solveRMP()


  // set matIsCvdObsByBox, whether or not each observation is vecCovered
  // by the lower and upper bounds, a and b of the current box k
  void Boosting::setMatIsCvdObsByBox(const unsigned int &k) {

    matIsCvdObsByBox[numBoxesSoFar+k].resize(numObs);

    for (unsigned int i=0; i< numObs; ++i) { // for each observation

      for (unsigned int j=0; j< numAttrib; ++j) { // for each attribute

        // if the curinsertPebbrent observation is covered by the current k'th box
        // for attribute k
        if ( matIntLower[numBoxesSoFar+k][j]
               <= data->dataIntTrain[i].X[j]
            && data->dataIntTrain[i].X[j]
               <= matIntUpper[numBoxesSoFar+k][j] ) {

          // if this observation is covered in all attributes
          if ( j==numAttrib-1)
            matIsCvdObsByBox[numBoxesSoFar+k][i] = true;  // set this observation is covered

        } else { // if it is not covered
          matIsCvdObsByBox[numBoxesSoFar+k][i] = false;  // set this observation is not covered
          break;
        } // end if this observation is covered in attribute j

      } // end for each attribute

    } // end for each observation

    DEBUGPR(1, cout << "matIsCvdObsByBox: " << matIsCvdObsByBox << "\n" );

  } // end setMatIsCvdObsByBox function


  // set vecIsObjValPos by box idx (k) and
  // isPosObjVal (wehther or not the current solution is positive)
  void Boosting::setVecIsObjValPos(const unsigned int &k,
                               const bool &isPosObjVal) {
    vecIsObjValPos[numBoxesSoFar+k] = isPosObjVal;
  } // enf setVecIsObjValPos function


  void Boosting::setMatIntBounds(const unsigned int &k,
                             const vector<unsigned int> &lower,
                             const vector<unsigned int> &upper) {
    matIntLower[matIntLower.size()-numBoxesIter+k] = lower;
    matIntUpper[matIntUpper.size()-numBoxesIter+k] = upper;
  } // end setVecIsObjValPos function


  // set original lower and upper bounds matrices
  void Boosting::setOriginalBounds() {

    // multimap<int, double>::iterator it;
    double tmpLower, tmpUpper;
    unsigned int boundVal;

    matOrigLower.resize(numBoxesSoFar);
    matOrigUpper.resize(numBoxesSoFar);

    // for each box inserted in the current boosting iteration
    for (unsigned int k = numBoxesSoFar-numBoxesIter; k<numBoxesSoFar; ++k) {

      matOrigLower[k].resize(numAttrib);
      matOrigUpper[k].resize(numAttrib);

      for (unsigned int j=0; j<numAttrib; ++j) { // for each attribute

        ///////////////////////////// mid point rule //////////////////////////////
        // if the lower integer bound is greater than 0
        if ( matIntLower[k][j] > 0 ) {

          // tmpLower = getUpperBound(k, j, -1, false);
          // tmpUpper = getLowerBound(k, j,  0, false);
          boundVal = matIntLower[k][j];
          tmpLower = data->vecAttribIntInfo[j].vecBins[boundVal-1].upperBound;
          tmpUpper = data->vecAttribIntInfo[j].vecBins[boundVal].lowerBound;

          matOrigLower[k][j] = (tmpLower + tmpUpper) / 2.0;  // mid point rule

          DEBUGPR(10, cout << "(k,j): (" << k << ", " << j
                  << ") matIntLower[k][j]-1: " << matIntLower[k][j]-1
                  << " LeastLower: "           << tmpLower << "\n"
                  << " matIntLower[k][j]: "    << matIntLower[k][j]
                  << " GreatestLower: "        << tmpUpper << "\n");

        // else if the lower bound in integer is 0,
        // the original lower bound is negative infinity
        } else matOrigLower[k][j] = -getInf();

        // if the upper integer bound is less than the maximum integer bound
        if ( matIntUpper[k][j] < data->vecNumDistVals[j]-1 ) {

          // tmpLower = getUpperBound(k, j, 0, true);
          // tmpUpper = getLowerBound(k, j, 1, true);
          boundVal = matIntUpper[k][j];
          tmpLower = data->vecAttribIntInfo[j].vecBins[boundVal].upperBound;
          tmpUpper = data->vecAttribIntInfo[j].vecBins[boundVal+1].lowerBound;

          matOrigUpper[k][j] = (tmpLower + tmpUpper) / 2.0;  // apply the mid point rule

          DEBUGPR(10, cout << "(k,j): (" << k << ", " << j
                  << ") matIntUpper[k][j]: " << matIntUpper[k][j]
                  << " LeastUpper: " << tmpLower << "\n"
                  << " matIntUpper[k][j]+1: " << matIntUpper[k][j]+1
                  << " GreatestUpper: " << tmpUpper << "\n");

        // else if the upper bound in integer is the maximum integer bound,
        // the original lower bound is negative infinity
        } else matOrigUpper[k][j] = getInf();

      } // end for each attribute

    } // end for each box

    DEBUGPR(10, cout << "Integerized Lower: \n" << matIntLower );
    DEBUGPR(10, cout << "Integerized Upper: \n" << matIntUpper );
    DEBUGPR(10, cout << "Original Lower: \n"    << matOrigLower );
    DEBUGPR(10, cout << "Original Upper: \n"    << matOrigUpper );

  } // end function REPR::setOriginalBounds()


  // setPebblRMASolutions
  void Boosting::setPebblRMASolutions() {

    rma->getAllSolutions(s);
    numBoxesIter = s.size();
    sl.resize(numBoxesIter);

    // for each boxes in current iteration
    for (unsigned int k=0; k<numBoxesIter; ++k)
      sl[k] = dynamic_cast<pebblRMA::rmaSolution*>(s[k]);

  } // end setPebblRMASolutions function


  // resize vecERMAObjVal and vecGRMAObjVal
  void Boosting::resetVecRMAObjVals() {

    // if PEBBL RMA is enabled
    if (isPebblRMA())
      vecERMAObjVal.resize(getNumIterations());

    // if PEBBL RMA and its initial guess options are enabled
    // or PEBBL RMA is not enabled (GreedyRMA only)
    if ( (isPebblRMA() && isInitGuess()) || !isPebblRMA() )
      vecGRMAObjVal.resize(getNumIterations());

  }  // end resetVecRMAObjVals function


  // set vecERMAObjVal and vecGRMAObjVal for current iteration
  void Boosting::setVecRMAObjVals() {

    // if PEBBL RMA is enabled
    if (isPebblRMA())
      vecERMAObjVal[curIter] = rma->workingSol.value;

    // if PEBBL RMA and its initial guess options are enabled
    // or PEBBL RMA is not enabled (GreedyRMA only)
    if ( (isPebblRMA() && isInitGuess()) || !isPebblRMA() )
      vecGRMAObjVal[curIter] = grma->getObjVal();

  } // end setVecRMAObjVals function

  /************************ Checking methods ************************/

  // TODO: combine this function for Pebbl and Greedy RMA boxes
  // check wether or not the "k"-th box in the current iteration is duplicated
  bool Boosting::checkDuplicateBoxes(vector<unsigned int> vecIntLower,
                                     vector<unsigned int> vecIntUpper) {

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

      // if the cuurent box is only one, no duplicates
      if ( matIntUpper.size()==1 )
        return false;

      // for each boxes already in model
      for (unsigned int l=0; l < numBoxesSoFar; ++l)
        for (unsigned int j=0; j<numAttrib; ++j) // for each attribute
          // if no duplicates boxes
          if ( matIntLower[l][j] != vecIntLower[j]       // sl[k]->a[j]
               || matIntUpper[l][j] != vecIntUpper[j] )  // sl[k]->b[j] )
            return false;

      ucout << "Duplicate Solution Found!! \n" ;

#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    return true;

  } // end checkDuplicateBoxes function


  // check objective value for the current lower and upper bounds
  void Boosting::checkObjValue(vector<DataXw> intData,
                               vector<unsigned int> lower,
                               vector<unsigned int> upper) {

    double totalWeights=0.0;

    for (unsigned int i=0; i<numObs; i++) { // for each training data

      for (unsigned int j=0; j<numAttrib; ++j) { // for each feature

        if ( (lower[j] <= intData[i].X[j])
             && (intData[i].X[j] <= upper[j]) ) {

          if (j==numAttrib-1)  // if this observation is covered by this solution
            totalWeights += intData[i].w;   //dataWts[i];

        } else break; // else go to the next observation

      }  // end for each feature

    }  // end for each training observation

    cout << "Lower Bound: " << lower ;
    cout << "Upper Bound: " << upper ;
    cout << "Objective Value: " << totalWeights << "\n";

  } // end checkObjValue function

  /************************ Printing functions ************************/

  // print curret iteration, testing and testing errors
  void Boosting::printBoostingErr() {
      ucout << "Iter: " << curIter
            << ":\tTest/Train Errors: " << errTest << " " << errTrain << "\n";
  } // end printBoostingErr function


  // print function for CLP solutions
  void Boosting::printCLPsolution() {

    int numberRows = model.numberRows();
    double * rowPrimal = model.primalRowSolution();
    double * rowDual = model.dualRowSolution();

    int iRow;

    for (iRow=0;iRow<numberRows;iRow++)
      printf("Row %d, primal %g, dual %g\n",iRow,
    rowPrimal[iRow],rowDual[iRow]);

    int numberColumns = model.numberColumns();
    double * columnPrimal = model.primalColumnSolution();
    double * columnDual = model.dualColumnSolution();

    int iColumn;

    for (iColumn=0;iColumn<numberColumns;iColumn++)
      printf("Column %d, primal %g, dual %g\n",iColumn,
    columnPrimal[iColumn],columnDual[iColumn]);

    ///////////////////////////////////////////////////////////
    //
    // double value;
    //
    // // Print column solution
    // int numberColumns = model.numberColumns();
    //
    // // Alternatively getColSolution()  // get Primal column solution
    // double * columnPrimal   = model.primalColumnSolution();
    // // Alternatively getReducedCost()  // get Dual row solution
    // double * columnDual     = model.dualColumnSolution();
    // // Alternatively getColLower()
    // double * columnLower    = model.columnLower();
    // // Alternatively getColUpper()
    // double * columnUpper    = model.columnUpper();
    // // Alternatively getObjCoefficients()
    // double * columnObjective = model.objective();
    //
    // int iColumn;
    //
    // cout << "               Primal          Dual         Lower         Upper          Cost"
    //           << endl;
    //
    // for (iColumn = 0; iColumn < numberColumns; iColumn++) { // for each column
    //
    //   cout << setw(6) << iColumn << " ";
    //
    //   // print primal variables
    //   value = columnPrimal[iColumn];
    //   if (fabs(value) < 1.0e5)
    //     cout << setiosflags(ios::fixed | ios::showpoint) << setw(14) << value;
    //   else
    //     cout << setiosflags(ios::scientific) << setw(14) << value;
    //
    //   // print dual variables
    //   value = columnDual[iColumn];
    //   if (fabs(value) < 1.0e5)
    //     cout << setiosflags(ios::fixed | ios::showpoint) << setw(14) << value;
    //   else
    //     cout << setiosflags(ios::scientific) << setw(14) << value;
    //
    //   // print primal lower bounds variables
    //   value = columnLower[iColumn];
    //   if (fabs(value) < 1.0e5)
    //     cout << setiosflags(ios::fixed | ios::showpoint) << setw(14) << value;
    //   else
    //     cout << setiosflags(ios::scientific) << setw(14) << value;
    //
    //   // print primal lower bounds variables
    //   value = columnUpper[iColumn];
    //   if (fabs(value) < 1.0e5)
    //     cout << setiosflags(ios::fixed | ios::showpoint) << setw(14) << value;
    //   else
    //     cout << setiosflags(ios::scientific) << setw(14) << value;
    //
    //   // print objective variables
    //   value = columnObjective[iColumn];
    //   if (fabs(value) < 1.0e5)
    //     cout << setiosflags(ios::fixed | ios::showpoint) << setw(14) << value;
    //   else
    //     cout << setiosflags(ios::scientific) << setw(14) << value;
    //
    //   cout << "\n";
    //
    // } // end each column

  } // end printCLPsolution function




  // save weights for current iteration (curIter)
  void Boosting::saveWeights(const unsigned int& curIter) {

    // TODO if this directory exists, create the directory
    //mkdir("./wts")

    // create a file name to save
    stringstream s;
    s << "wt_" << problemName << "_" << curIter << ".out";
    ofstream os(s.str().c_str());

    for (unsigned int i=0; i < numObs ; ++i) // for each observation
      os << data->dataIntTrain[i].w << ", "; // output each weight

  } // end saveWeights function







  /************************ Saving functions ************************/




  // save actual y values and predicted y values
  void Boosting::savePredictions(const bool &isTest, vector<DataXy> origData) {

    unsigned int numIdx;
    stringstream s;

    // create a file name to save
    if (isTest) s << "predictionTest" << '.' << problemName;
    else        s << "predictionTrain" << '.' << problemName;

    ofstream os(s.str().c_str());
    // appending to its existing contents
    //ofstream os(s.str().c_str(), ofstream::app);

    os << "ActY \t Boosting  \n";

    numIdx = (isTest) ? data->numTrainObs : data->numTestObs;

    for (unsigned int i=0; i < numIdx; ++i) { // for each observation

      if (isTest) os << origData[i].y  << " " << predTest[i] << "\n";
      else        os << origData[i].y << " " << predTrain[i] << "\n";

    } // end for each observation

    os.close();  // close the file

  } // for savePredictions function


  // save both greedy and PEBBL RMA solutions for all Boosting iterations
  void Boosting::saveGERMAObjVals() {

    // set the output file name
    stringstream s;
    s << "GERMA_" << problemName << ".out";
    ofstream os(s.str().c_str());

    // output Greedy RMA and PEBBL RMA solutions for each iteration
    os << "iter\tGRMA\tERMA\n";
    for (unsigned int i=0; i<getNumIterations(); ++i ) // for each iteration
      os << i << "\t" << vecGRMAObjVal[i] << "\t" << vecERMAObjVal[i] << "\n" ;

    os.close();  // close the output file

  } // end saveGERMAObjVals function


} // namespace boosting
