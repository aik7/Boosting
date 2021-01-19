/*
 *  Author:      Ai Kagawa
 *  Description: a source file for Boosting class
 */

#include "boosting.h"


namespace boosting {

  ///////////////////////// Set-up Boosting /////////////////////////

  Boosting::Boosting(int& argc, char**& argv)
#ifdef HAVE_GUROBI
  :  modelGrb(env)
#endif // HAVE_GUROBI
  {

    initMPI(argc, argv);

    isRMAonly = false;     // is not RMA only

    setup(argc, argv);     // setup all paramaters from PEBBL

    if      (rmaSolveMode().compare("exact")==0)
      _isPebblRMA = true;
    else if (rmaSolveMode().compare("greedy")==0)
      _isPebblRMA = false;
    // TODO: add hybrid here

    setDataRMA(argc, argv);   // set Data RMA class object from SolveRMA class

    initMPI(argc, argv);

    if (isPebblRMA())
      setupPebblRMA(argc, argv);  // setup PEBBL RMA

    resetBoosting();                              // reset Boosting

    (isPebblRMA()) ? greedyLevel=EXACT : greedyLevel=GREEDY;

    if (ROOTPROC) {
      cout << (isUseGurobi() ? "Gurobi" : "CLP") << " solver\n";
      createOutputDir();
    }

  } // end Boosting constructor


  void Boosting:: createOutputDir() {

    int n = outputDir().length();  // # of characters in the string
    char charOutDir[n + 1];        // declaring character array

    // copying the contents of the string to char array
    strcpy(charOutDir, outputDir().c_str());

    // creating a directory
    if (mkdir(charOutDir, 0777) != 0)
        cout << strerror(errno) << "\n";
    else
        cout << "\"" << outputDir() << "\" directory created\n";

  } // end createOutputDir function


  // reset Boosting variables
  void Boosting::resetBoosting() {

    isStopCond = 0;   // whether or not to the model hits the stopping condition
    curIter    = 0;   // current boosting iteration

    numBoxesSoFar = 0;  // # of boxes inserted by the previous boosing iteration
    numBoxesIter  = 0;  // # of boxes to insert in the current iteration

    if (ROOTPROC) { // if root process

      // matrix containes lower and upper bounds of boxes in integerized value
      matIntLower.clear();
      matIntUpper.clear();

      // matrix containes lower and upper bounds of boxes in original value
      matOrigLower.clear();
      matOrigUpper.clear();

      // matrix containes whether or not each observation is covered by each box
      matIsCvdObsByBoxTrain.clear();
      matIsCvdObsByBoxTest .clear();

      // if isSaveErrors is enabled,
      // allocate to store error for the train and test datasets
      if (isSaveErrors()) {
        vecErrTrain.resize(getNumIterations()+1);
        if (data->numTestObs!=0)
          vecErrTest .resize(getNumIterations()+1);
      }

      // if isSavePred is enabled and the last column generation iteration
      if ( isSavePred() ) {
        vecPredTrain.resize(data->numTrainObs);
        if (data->numTestObs!=0)
          vecPredTest. resize(data->numTestObs);
      }

      // if the isSaveAllRMASols is enabled,
      // allocate vecERMAObjVal and vecGRMAObjVal
      if (isSaveAllRMASols()) resetVecRMAObjVals();

    } // end root process

  } // end reset function


  ///////////////////////// Training methods /////////////////////////

  // training process of boosting
  void Boosting::train(const unsigned int& numIter,
                       const unsigned int& greedyLevel) {

    setBoostingParameters();  // set Boosting parameters

    try {

      // use only one process to solve RMP
      if (ROOTPROC) { // if root process

        setStandardizedData();  // set data->dataStandTrain, standardize data

        setInitRMP();    // set the initial RMP

        solveRMP();      // solve RMP using CLP or Gurobi

        // if the option for evaluating each iteration is enabled
        // evaluate the model
        if (isEvalEachIter()) evaluateModel();

      } // end if root process

      // for each column generation iteration
      for (curIter=1; curIter<numIter+1; ++curIter) {

        //ucout << "\nColGen Iter: " << curIter << "\n";

        setWeights();  // set data weight in DataRMA class

        solveRMA();    // solve the subproblem of RMA

        if (greedyLevel==EXACT) setPebblRMASolutions();

        if (ROOTPROC) { // if root process

          // save the weights for the current iteration
          if (isSaveWts())           saveWeights(curIter);

          // save Greedy and/or Exact RMA solutions in a file
          if (isSaveAllRMASols())    setVecRMAObjVals();

          if (isStoppingCondition()) isStopCond = 1;

        } // end if root process

        setStoppingCondition();

        if (isStopCond==1) break;

        if (ROOTPROC) { // if root process

          insertColumns();  // insert columns using RMA solutions

          solveRMP();       // solve the updated RMP

          // set matOrigLower and matOrigUpper using matIntLower and matIntUpper
          // map the integer bounds to the original bounds
          if (delta()!=-1) setOriginalBounds();

          if (isEvalEachIter()) evaluateModel();

        } // end if root process

      } // end for each column generation iteration

    } catch(...) {
      ucout << "Exception during training" << "\n";
      return; // EXIT_FAILURE;
    } // end try ... catch

    if (ROOTPROC) { // if root process

      // if eavluating the final iteration option is enabled
      // and eavluating the final iteration option is disabled
      if ( isEvalFinalIter() && !isEvalEachIter() ) evaluateModel();

      if ( isSaveErrors() ) saveErrors();

      // save predictions
      if ( isSavePred() ) {
        savePredictions(TRAIN, data->dataOrigTrain);
        if (data->numTestObs!=0)
          savePredictions(TEST,  data->dataOrigTrain);
      }

      // save Greedy and PEBBL RMA solutions for each iteration
      if ( isSaveAllRMASols() ) saveGERMAObjVals();

      if ( isSaveModel() )      saveModel();    // save Boosting model

    } // end if root process

  } // end train function


  // set dataStandTrain
  void Boosting::setStandardizedData() {

    // standadize data for L1 regularization
    if (isStandData()) {
      data->setStandardizedY();  // set data->dataStandTrain
      data->setStandardizedX();
    } else { // not to standerdise for debugging purpose
      data->dataStandTrain = data->dataOrigTrain;
    }

  } // end setDataStand function


  // solve RMP using CLP or Gurobi
  void Boosting::solveRMP() {

#ifdef HAVE_GUROBI
    if (isUseGurobi())
      solveGurobiRMP();   // solve the RMP using Gurobi
    else {
#endif // HAVE_GUROBI
      solveClpRMP();    // solve the RMP using CLP
#ifdef HAVE_GUROBI
    }
#endif // HAVE_GUROBI

    printRMPSolution();

  } // end solveRMP function


  // call CLP to solve Master Problems
  // and get objective value, primal and dual variables
  void Boosting::solveClpRMP() {

    DEBUGPR(10, cout << "Solve Restricted Master Problem!\n");

    tc.startTime();

    modelClp.primal();
    // modelClp.dual();  //  invoke the dual simplex method.

    primalObjVal  = modelClp.objectiveValue();        // get objective value
    vecPrimalVars = modelClp.primalColumnSolution();  // ger primal variables
    vecDualVars   = modelClp.dualRowSolution();       // dualColumnSolution();

    if (isPrintBoost()) {
      int iter;
      char clpFile[12];
      strcpy(clpFile, "clp_");
      iter = (curIter>getNumIterations()) ? -1 : curIter;
      strcat(clpFile, to_string(iter).c_str());
      strcat(clpFile, ".mps");
      modelClp.writeMps(clpFile);
    }

    if (debug>=1)
      printRMPCheckInfo();

  } // end function Boosting::solveRMP()


#ifdef HAVE_GUROBI

  // reset Gurobi for the next column generation iteration
  void Boosting::resetGurobi() {

    // remove constraints
    for (unsigned int i=0; i<numRows; ++i) // for each row
      modelGrb.remove(modelGrb.getConstrs()[i]);

    // remote variables
    for (unsigned int j=0; j<numCols; ++j) // for each column
      modelGrb.remove(modelGrb.getVars()[j]);

    modelGrb.reset();  // reset the Gurobi model
    modelGrb.update(); // update the Gurobi model

  } // end resetGurobi function


  // call GUROBI to solve Master Problems
  // and get objective value, primal and dual variables
  void Boosting::solveGurobiRMP() {

    // DEBUGPRX(10, data, "Solve Restricted Master Problem!\n");

    tc.startTime();

    modelGrb.optimize();  // run the Gurobi solver

    vecPrimalVars = new double[numCols];  // allocate space for primal variables
    vecDualVars   = new double[numRows];  // allocate space for dual variables

    vars   = modelGrb.getVars();     // get all variables
    constr = modelGrb.getConstrs();  // get all constraints

    // if Gurobi status is optimal, get the solutions
    if (modelGrb.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {

      primalObjVal = modelGrb.get(GRB_DoubleAttr_ObjVal);

      // get primal variables
      for (unsigned int j = 0; j < numCols; ++j)  // for each column
        vecPrimalVars[j] = vars[j].get(GRB_DoubleAttr_X);

      // get dual variables
      for (unsigned int i = 0; i < numRows; ++i)  // for each row
        vecDualVars[i]   = constr[i].get(GRB_DoubleAttr_Pi);

    } // end if Gurobi status is optimal

    if (isPrintBoost()) {
      int iter;
      char grbFile[12];
      strcpy(grbFile, "grb_");
      iter = (curIter>getNumIterations()) ? -1 : curIter;
      strcat(grbFile, to_string(iter).c_str());
      strcat(grbFile, ".mps"); // .lp, .mps, or .rew
      modelGrb.write(grbFile);
    }

    if (debug>=1)
      printRMPCheckInfo();

    // DEBUGPRX(0, data, " Master Solution: " <<primalObjVal << "\t");
    // printRMPSolution();

  } // end function Boosting::solveMaster()

#endif // HAVE_GUROBI


  void Boosting::setStoppingCondition() {

#ifdef ACRO_HAVE_MPI
    MPI_Bcast(&isStopCond, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  } // end setIsStopCond function


  // set vecIsObjValPos by box idx (k) and
  // isPosObjVal (wehther or not the current solution is positive)
  void Boosting::setVecIsObjValPos(const unsigned int &k,
                               const bool &isPosObjVal) {
    vecIsObjValPos[numBoxesSoFar+k] = isPosObjVal;
    DEBUGPR(1, cout << "vecIsObjValPos: " << vecIsObjValPos );
  } // enf setVecIsObjValPos function


  void Boosting::setMatIntBounds(const unsigned int &k,
                             const vector<unsigned int> &lower,
                             const vector<unsigned int> &upper) {
    matIntLower[numBoxesSoFar+k] = lower;
    matIntUpper[numBoxesSoFar+k] = upper;
    DEBUGPR(1, cout << "matIntLower: " << matIntLower );
    DEBUGPR(1, cout << "matIntUpper: " << matIntUpper );
  } // end setMatIntBounds function


  // TODO: merge setMatIsCvdObsByBox functions
  // set matIsCvdObsByBox, whether or not each observation is vecCovered
  // by the lower and upper bounds, a and b of the current box k
  void Boosting::setMatIsCvdObsByBox(const unsigned int &k) {

    matIsCvdObsByBoxTrain[numBoxesSoFar+k].resize(data->numTrainObs);

    for (unsigned int i=0; i< data->numTrainObs; ++i) { // for each observation

      for (unsigned int j=0; j< data->numAttrib; ++j) { // for each attribute

        // if the current observation is covered by the current k'th box
        // for attribute k
        if ( matIntLower[numBoxesSoFar+k][j]
               <= data->dataIntTrain[i].X[j]
            && data->dataIntTrain[i].X[j]
               <= matIntUpper[numBoxesSoFar+k][j] ) {

          // if this observation is covered in all attributes
          if ( j==data->numAttrib-1) // set this observation is covered
            matIsCvdObsByBoxTrain[numBoxesSoFar+k][i] = true;

        } else { // if it is not covered, set this observation is not covered
          matIsCvdObsByBoxTrain[numBoxesSoFar+k][i] = false;
          break;
        } // end if this observation is covered in attribute j

      } // end for each attribute

    } // end for each observation

    DEBUGPR(1, cout << "matIsCvdObsByBoxTrain: " << matIsCvdObsByBoxTrain << "\n" );

  } // end setMatIsCvdObsByBox function


  // set matIsCvdObsByBox, whether or not each observation is vecCovered
  // by the lower and upper bounds, a and b of the current box k
  void Boosting::setMatIsCvdObsByBox(const int &k, const bool &isTrain,
                                     const vector<DataXy> &origData,
                                     deque<deque<bool> > &matIsCvdObsByBox) {

    unsigned int numObs = isTrain ? data->numTrainObs : data->numTestObs;

    matIsCvdObsByBox[numBoxesSoFar-numBoxesIter+k].resize(numObs);

    for (unsigned int i=0; i< numObs; ++i) { // for each observation

      for (unsigned int j=0; j< data->numAttrib; ++j) { // for each attribute

        // if the current observation is covered by the current k'th box
        // for attribute k
        if ( matOrigLower[numBoxesSoFar-numBoxesIter+k][j] <= origData[i].X[j]
             && origData[i].X[j]
             <= matOrigUpper[numBoxesSoFar-numBoxesIter+k][j] ) {

          // if this observation is covered in all attributes
          if ( j==data->numAttrib-1) // set this observation is covered
            matIsCvdObsByBox[numBoxesSoFar-numBoxesIter+k][i] = true;

        } else { // if it is not covered, set this observation is not covered
          matIsCvdObsByBox[numBoxesSoFar-numBoxesIter+k][i] = false;
          break;
        } // end if this observation is covered in attribute j

      } // end for each attribute

    } // end for each observation

    DEBUGPR(1, cout << "matIsCvdObsByBox: " << matIsCvdObsByBox << "\n" );

  } // end setMatIsCvdObsByBox function


  // TODO: I am sure that there is better way to to this...
  void Boosting::setMatIsCvdObsByBoxTestPerIter() {
    matIsCvdObsByBoxTest.resize(numBoxesSoFar);
    for (unsigned int k=0; k<numBoxesIter; ++k)
      setMatIsCvdObsByBox(k, TEST, data->dataOrigTest, matIsCvdObsByBoxTest);
  }


  // set original lower and upper bounds matrices
  void Boosting::setOriginalBounds() {

    // multimap<int, double>::iterator it;
    double tmpLower, tmpUpper;
    unsigned int boundVal;

    matOrigLower.resize(numBoxesSoFar);
    matOrigUpper.resize(numBoxesSoFar);

    // for each box inserted in the current boosting iteration
    for (unsigned int k = numBoxesSoFar-numBoxesIter; k<numBoxesSoFar; ++k) {

      matOrigLower[k].resize(data->numAttrib);
      matOrigUpper[k].resize(data->numAttrib);

      for (unsigned int j=0; j<data->numAttrib; ++j) { // for each attribute

        //////////////////////////// mid point rule ///////////////////////////
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

    // PEBBL branching::getAllSolutions
    // Do you need whichProcessor?
    rma->getAllSolutions(vecPebblSols);

    // set # of boxes inserting in this iteration
    numBoxesIter = vecPebblSols.size();
    vecPebblRMASols.resize(numBoxesIter);

    // for each boxes in current iteration,
    // dynamically downcast from a PEBBL solution object
    // to a RMA solution object
    for (unsigned int k=0; k<numBoxesIter; ++k)
      vecPebblRMASols[k] = dynamic_cast<pebblRMA::rmaSolution*>(vecPebblSols[k]);

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

    if (ROOTPROC) { // if root process

      // if the cuurent box is only one, no duplicates
      if ( matIntUpper.size()==1 )
        return false;

      // for each boxes already in model
      for (unsigned int l=0; l < numBoxesSoFar; ++l)
        for (unsigned int j=0; j<data->numAttrib; ++j) // for each attribute
          // if no duplicates boxes
          if ( matIntLower[l][j] != vecIntLower[j]
               || matIntUpper[l][j] != vecIntUpper[j] )
            return false;

      ucout << "Duplicate Solution Found!! \n" ;

    } // end if root process

    return true;

  } // end checkDuplicateBoxes function


  // evalute the model each iteration
  void Boosting::evaluateModel() {

    errTrain = evaluate(TRAIN, data->dataOrigTrain, matIsCvdObsByBoxTrain);

    if (data->numTestObs!=0) {
      setMatIsCvdObsByBoxTestPerIter();
      errTest  = evaluate(TEST,  data->dataOrigTest, matIsCvdObsByBoxTest);
    }

    printBoostingErr();

  } // end evaluateModel function

  /************************ Printing functions ************************/

  // print RMP objectiva value and CPU run time
  void Boosting::printRMPSolution() {
    cout << fixed << setprecision(4)
        << "Master Solution: " << primalObjVal << "\t";
    cout << fixed << setprecision(2)
        << " CPU Time: " << tc.getCPUTime() << "\n";
  } // end printRMPSolution function


  // print curret iteration, testing and testing errors
  void Boosting::printBoostingErr() {
    ucout << "Iter: " << curIter;
    ucout << "\tTrainMSE: " << errTrain;
    if(data->numTestObs!=0) ucout << "\tTestMSE: "  << errTest;
    ucout << "\n";
  } // end printBoostingErr function

  /************************ Saving functions ************************/

  // save actual y values and predicted y values
  void Boosting::saveErrors() {

    stringstream s;

    // create a file name to save
    s << outputDir() << "/" << "error_" << problemName << ".out";

    ofstream os(s.str().c_str());
    // appending to its existing contents
    //ofstream os(s.str().c_str(), ofstream::app);

    // header
    os << "Iteration\tTrain_MSE";
    if (data->numTestObs!=0) os << "\tTest_MSE";
    os << "\n";

    // for each boosting iteration
    for (unsigned int i=0; i < getNumIterations()+1; ++i) {
      os << i << "\t" << vecErrTrain[i];
      if (data->numTestObs!=0) os << "\t" << vecErrTest[i];
      os << "\n";
    }

    os.close();  // close the file

  } // end saveErrors function


  // save actual y values and predicted y values
  void Boosting::savePredictions(const bool &isTrain, vector<DataXy> origData) {

    stringstream s;

    // create a file name to save
    if (isTrain)
      s << outputDir() << "/" << "predictionTrain_" << problemName << ".out";
    else
      s << outputDir() << "/" << "predictionTest_"  << problemName << ".out";

    ofstream os(s.str().c_str());
    // appending to its existing contents
    //ofstream os(s.str().c_str(), ofstream::app);

    unsigned int numObs = (isTrain ? data->numTrainObs : data->numTestObs );

    os << "ActY\tBoostingPredictedY\n";

    for (unsigned int i=0; i < numObs; ++i) { // for each observation

      if (isTrain) os << origData[i].y << "\t" << vecPredTrain[i] << "\n";
      else         os << origData[i].y << "\t" << vecPredTest[i]  << "\n";

    } // end for each observation

    os.close();  // close the file

  } // for savePredictions function


  // save both greedy and PEBBL RMA solutions for all Boosting iterations
  void Boosting::saveGERMAObjVals() {

    // set the output file name
    stringstream s;

    s << outputDir() << "/" << "RMAObjVals_" << problemName << ".out";

    ofstream os(s.str().c_str());

    os << "Iteration\tGreedy_RMA\tPEBBL_RMA\n";

    // for each boosting iteration
    for (unsigned int i=0; i<getNumIterations(); ++i )
      os << i << "\t" << vecGRMAObjVal[i] << "\t" << vecERMAObjVal[i] << "\n" ;

    os.close();  // close the output file

  } // end saveGERMAObjVals function


  // save weights for current iteration (curIter)
  void Boosting::saveWeights(const unsigned int& curIter) {

    // create a file name to save
    stringstream s;
    s << outputDir() << "/" << "wt_" << problemName << "_" << curIter << ".out";
    ofstream os(s.str().c_str());

    for (unsigned int i=0; i < data->numTrainObs ; ++i) // for each observation
      os << data->dataIntTrain[i].w << ", "; // output each weight

  } // end saveWeights function


} // namespace boosting
