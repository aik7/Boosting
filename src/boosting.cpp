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

    setup(argc, argv);     // setup all paramaters from PEBBL

    setData(argc, argv);   // set Data RMA class object from SolveRMA class

    (isPebblRMA()) ? greedyLevel=EXACT : greedyLevel=Greedy;

    if (isPebblRMA()) setupPebblRMA(argc, argv);  // setup PEBBL RMA

    resetBoosting();                              // reset Boosting

    cout << (isUseGurobi() ? "Gurobi" : "CLP") << " solver\n";

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

    matIntLower.clear();    // matrix containes lower bound of box in integerized value
    matIntUpper.clear();    // matrix containes upper bound of box in integerized value

    matOrigLower.clear();   // matrix containes lower bound of box in original value
    matOrigUpper.clear();   // matrix containes upper bound of box in original value

    matIsCvdObsByBox.clear();  // matrix containes whether or not each observation is covered by each box

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

    isStopCond = 0;
    curIter    = -1;

    setBoostingParameters();  // set Boosting parameters

    try {

      // use rank 1 to solve RMP
#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

      setDataStand();  // set data->dataStandTrain, standardize data

      setInitRMP();    // set the initial RMP

      solveRMP();      // solve RMP using CLP or Gurobi

      // if the option for evaluating each iteration is enabled
      // evaluate the model
      if (isEvalEachIter()) evaluateModel();

#ifdef ACRO_HAVE_MPI
  } // end if (uMPI::rank==0)
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
        if (isSaveWts())           saveWeights(curIter);

        // save Greedy and/or Exact RMA solutions in a file
        if (isSaveAllRMASols())    setVecRMAObjVals();

        if (isStoppingCondition()) isStopCond = 1;

#ifdef ACRO_HAVE_MPI
  } // end if (uMPI::rank==0)
#endif //  ACRO_HAVE_MPI

        setStoppingCondition();

        if (isStopCond==1) break;

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

      insertColumns();  // insert columns using RMA solutions

      solveRMP();       // solve the updated RMP

      // map back from the discretized data into original
      if (delta()!=-1) setOriginalBounds();

      if (isEvalEachIter()) evaluateModel();

#ifdef ACRO_HAVE_MPI
  } // end if (uMPI::rank==0)
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
    if ( isEvalFinalIter() && !isEvalEachIter() ) evaluateModel();

    saveGERMAObjVals();   // save Greedy and PEBBL RMA solutions for each iteration

    saveModel();          // save Boosting model

#ifdef ACRO_HAVE_MPI
        }
#endif //  ACRO_HAVE_MPI

  } // end trainData function


  // set dataStandTrain
  void Boosting::setDataStand() {

    // standadize data for L1 regularization
    if (isStandData()) {
      data->setDataStandY();  // set data->dataStandTrain
      data->setDataStandX();
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

    DEBUGPR(10, cout <<  "Solve Restricted Master Problem!\n");

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

    if (uMPI::rank==0) {

      // If we are the root process, send our data to everyone
      for (int k = 0; k < uMPI::size; ++k)
        if (k != 0)
          MPI_Send(&isStopCond, 1, MPI_INT, k, 0, MPI_COMM_WORLD);

    } else { // if MPI rank is not 0, receive the info about the stopping condition

      // If we are a receiver process, receive the data from the root
      MPI_Recv(&isStopCond, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);

    } // save if MPI rank is not 0

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


    // evalute the model each iteration
    void Boosting::evaluateModel() {
      errTrain = evaluateEachIter(TRAIN, data->dataOrigTrain);
      errTest  = evaluateEachIter(TEST,  data->dataOrigTest);
      printBoostingErr();
    } // end evaluateModel function

  /************************ Printing functions ************************/

  // print RMP objectiva value and CPU run time
  void Boosting::printRMPSolution() {
    cout << fixed << setprecision(4)
        << "Master Solution: " <<primalObjVal << "\t";
    cout << fixed << setprecision(2)
        << " CPU Time: " << tc.getCPUTime() << "\n";
  } // end printRMPSolution function


  // print curret iteration, testing and testing errors
  void Boosting::printBoostingErr() {
      ucout << "Iter: " << curIter
            << ":\tTest/Train Errors: " << errTest << " " << errTrain << "\n";
  } // end printBoostingErr function


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

      if (isTest) os << origData[i].y << " " << predTest[i] << "\n";
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
