/*
 *  Author:      Ai Kagawa
 *  Description: a source file for REPR class
 */

#include "repr.h"

namespace boosting {


  // set REPR parameters
  void REPR::setBoostingParameters() {
    P = 1;
    C = getCoefficientC();
    D = 0; //getCoefficientD();
    E = getCoefficientE();
    F = 0; //getCoefficientF();
  } // end setBoostingParameters function


  /****************** set initial RMP in CLP (begin) ***********************/

  // set the initial restricted master problem (RMP)
  void REPR::setInitRMP() {

    DEBUGPR(10, cout << "Set Initial Restricted Master Problem!" << "\n");

    setInitRMPVariables();   // initialize variables for RMP

    setClpParameters();      // set CLP parameters

    setInitRMPObjective();   // set Objective

    setInitRMPColumnBound(); // set column bounds

    setInitRMPRowBound();    // set row bounds

    setConstraintsLHS();     // set the left hand side of the constraints

    setInitRMPClpModel();    // set the CLP model

  }  // end function REPR::setInitialMaster()


  // set initial RMP variables
  void REPR::setInitRMPVariables() {

    // # of columns for RMP
    numCols = 1 + 2 * numAttrib + numObs;      // vecPrimalVars.size();

    // # of rows
    numRows = 2 * numObs ; // //NumVar+1; // +1 for constant term

    DEBUGPR(1, cout << "numAttrib: " << numAttrib << "\n");
    DEBUGPR(1, cout << "numObs: "    << numObs    << "\n");
    DEBUGPR(1, cout << "numCols: "   << numCols   << "\n");

  } // end setInitRMPVariables function


  // set CLP model
  void REPR::setClpParameters() {

    colIndex    = new int[numRows];  // row index for this colum

    for (unsigned int i=0; i<numRows; ++i) // for each row
      colIndex[i] = i;

    // This is fully dense - but would not normally be so
    numElements = numRows * numCols;

    elements = new double [numElements];  // elements in the constraints
    rows     = new int [numElements];
    starts   = new CoinBigIndex [numCols+1]; // the first index of this column
    lengths  = new int [numCols];            // # of rows in each column

    // columns to insert
    columnInsert = new double[numRows];

    model.setOptimizationDirection(1); // maximization

    // to turn off some output, 0 gives nothing and each increase
    // in value switches on more messages.
    model.setLogLevel(0);
    // matrix.setDimensions(numRows, numCols); // setDimensions (int numrows, int numcols)

    // set the dimension of the model
    model.resize(numRows, numCols);

  } // end resetCLPModel function


  // set Initial RMP objective
  void REPR::setInitRMPObjective() {

    objective   = model.objective();

    for (unsigned int k = 0; k < numCols; ++k) { // end for each column
      if (k==0)                 objective[k] = 0.0;  // for the constnt term
      else if (k<1+2*numAttrib) objective[k] = C;    // for the linear variables
      else                      objective[k] = 1;    // for the observation variables
    } // end for each column

  } // end setInitRMPObjective


  // set column bounds for RMP
  void REPR::setInitRMPColumnBound() {

    columnLower = model.columnLower();
    columnUpper = model.columnUpper();

    columnLower[0] = -COIN_DBL_MAX;

    // set lower and upper bounds for each variables
    // for (unsigned int k = 0; k < numCols; ++k) { // for each colum
    //   columnLower[k] = (k==0) ? -COIN_DBL_MAX : 0.0;
    //   columnUpper[k] = COIN_DBL_MAX;
    // } // end each column

    if (debug>=5)  {
      cout << "column lower: ";
      for (unsigned int i = 0; i<numCols; ++i)
        if (columnLower[i] == -COIN_DBL_MAX) cout << "-inf, ";
        else                                 cout << columnLower[i] << ", ";
      cout << "\ncolumn upper: ";
      for (unsigned int i = 0; i<numCols; ++i)
        if (columnUpper[i] == COIN_DBL_MAX) cout << "inf, ";
        else                                cout << columnUpper[i] << ", ";
      cout << "\n";
    }

  } // end setInitRMPColumnBound function


  // set row bounds for RMP
  void REPR::setInitRMPRowBound() {

    rowLower    = model.rowLower();
    rowUpper    = model.rowUpper();

    for (unsigned int k = 0; k < numRows; k++) { // for each row
      if (k < numObs) {
        rowLower[k] = -COIN_DBL_MAX; //-inf;
        rowUpper[k] = data->dataStandTrain[k].y;
      } else {
        rowLower[k] = -COIN_DBL_MAX; //-inf;
        rowUpper[k] = -data->dataStandTrain[k-numObs].y;
      } // end if
    } // end each row

    if (debug>=5) {
      cout << "\nrow lower: ";
      for (unsigned int i = 0; i<numRows; ++i)
        if (rowLower[i] == -COIN_DBL_MAX) cout << "-inf, ";
        else                              cout << rowLower[i] << ", ";
      cout << "\nrow: upper";
      for (unsigned int i = 0; i<numRows; ++i)
        if (rowUpper[i] == COIN_DBL_MAX) cout << "inf, ";
        else                             cout << rowUpper[i] << ", ";
      cout << "\n";
    }

  } // end setInitRMPRowBound function


  // set LHS (left-hand side) constraints (elements),
  // and set starts, lengths, and rows
  void REPR::setConstraintsLHS() {

    unsigned int idx;
    CoinBigIndex idxClp = 0;

    for (unsigned int j = 0; j < numCols; j++) { // for each column

      starts[j]  = idxClp;    // set each column index starts
      lengths[j] = numRows;   // set # of rows in each column

      for (unsigned int i = 0; i < numRows; ++i) { // for each row

        idx = (i <numObs) ? i : i-numObs;

        rows[idxClp] = i;     // set each element's row index

        if (j==0) // for the constant terms
          elements[idxClp] = (i < numObs) ? 1.0 : -1.0;
        else if (j<1+numAttrib) {    // for positive linear variables
          if (i < numObs)
            elements[idxClp] = data->dataStandTrain[idx].X[j-1];  // -1 for constant term
          else
            elements[idxClp] = -data->dataStandTrain[idx].X[j-1];
        } else if (j<1+2*numAttrib) { // for negative linear variables
          if (i < numObs)
            elements[idxClp] = -data->dataStandTrain[idx].X[j-1-numAttrib];
          else
            elements[idxClp] = data->dataStandTrain[idx].X[j-1-numAttrib];
        } else { // for oservation error variables, episilon_i
          if (j-1-2*numAttrib==idx)
            elements[idxClp] = (j < numObs) ? 1.0 : -1.0;
          else
            elements[idxClp] = 0;
        }

        ++idxClp; // increment the CLP index

      } // end for each row

    } // end for each column

    starts[numCols] = idxClp;  // last column's index starts ...

    if (debug>=10) printClpElements();  // print the element matrix

  } // end setConstraintsLHS function


  // set CLP model
  void REPR::setInitRMPClpModel() {

    // assign to matrix
    matrix = new CoinPackedMatrix(true, 0.0, 0.0);
    matrix->assignMatrix(true, numRows, numCols, numElements,
                         elements, rows, starts, lengths);

    clpMatrix = new ClpPackedMatrix(matrix);

    // replace CLP matrix (current is deleted by deleteCurrent=true)
    model.replaceMatrix(clpMatrix, true);

    // matrix.assignMatrix(true, numRows, numCols, numElements,
    //                       elements, rows, starts, lengths);
    // // clpMatrix = ClpPackedMatrix(matrix);
    // model.loadProblem(matrix,
    //                   columnLower, columnUpper, objective,
    //                   rowLower, rowUpper);

  } // end setInitRMPModel function

  /****************** set initial RMP in CLP (end) ***********************/

  // set a weight of each observation for RMA
  // if rank 0, send the weights to the other ranks
  // if not rank 0, receive the weights from the rank 0
  void REPR::setWeights() {

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

      // assign a weight for each observation ($/mu-/nu$, dual variables)
      for (unsigned int i=0; i < numObs ; ++i) // for each observation
        data->dataIntTrain[i].w = vecDualVars[i] - vecDualVars[numObs+i];

#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

#ifdef ACRO_HAVE_MPI

    for (unsigned int i = 0; i < numObs; ++i) { // for each observation

      if ((uMPI::rank==0)) { // if rank 0

        // If we are the root process, send the observation weights to everyone
        for (int k = 0; k < uMPI::size; ++k)
          if (k != 0)
            MPI_Send(&data->dataIntTrain[i].w,
                     1, MPI_DOUBLE, k, 0, MPI_COMM_WORLD);

      } else { // else (if not rank 0)

        // If we are a receiver process, receive the data from the root
        MPI_Recv(&data->dataIntTrain[i].w,
                 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      } // end if rank 0 else ...

    } // end for each observation

#endif //  ACRO_HAVE_MPI

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

    if(debug>=1) {
      ucout << "Weight: ";
      for (unsigned int i=0; i < numObs ; ++i)
            ucout << data->dataIntTrain[i].w << ", ";
      ucout << "\n";
    } // end debug

#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

  } // end setWeights function


  // check wether or not to spop the REPR iteration
  bool REPR::isStoppingCondition() {

    if (greedyLevel==EXACT) {  // if PEBBL RMA
      // if the incumbent is less tha E + threthold
      if (rma->incumbentValue <= E ) {
        ucout << "PEBBL Stopping Condition!\n";
        return true;
      } // end if the stopping condition
    } // end if PEBBL RMA

    if (greedyLevel==Greedy) { // if greedy RMA
      // if the current iteration is greater than 0,
      // and the current lower and upper bounads are the same
      // as the previous iteration's
      if (curIter>0
          && grma->getLowerBounds() == matIntLower[curIter-1]
          && grma->getUpperBounds() == matIntUpper[curIter-1] ) {
        ucout << "Greedy Stopping Condition!\n";
        return true;
      } // end if the stopping condition
    } // end if greedy RMA

    return false;

  } // end isStoppingCondition function



  // insert columns in each column iteration
  void REPR::insertPebblColumns() { //const int& GreedyLevel) {

    if ( curIter!=0 ) { // if not the first iteration
      // check whether or not the current iterations has any duplicate boxes
      if (debug>=1) {
        for (unsigned int k=0; k<numBoxesIter; ++k)
          checkDuplicateBoxes(sl[k]->a, sl[k]->b);
      } // end debug
    } // end if not the first iteration

    vecIsObjValPos.  resize(numBoxesSoFar+numBoxesIter);
    matIntUpper.     resize(numBoxesSoFar+numBoxesIter);
    matIntLower.     resize(numBoxesSoFar+numBoxesIter);
    matIsCvdObsByBox.resize(numBoxesSoFar+numBoxesIter);

    for (unsigned int k=0; k<numBoxesIter; ++k) { // for each solution

      //if (!checkDuplicateBoxes(k)) { // if this rule is not duplicates

      setVecIsObjValPos(k, sl[k]->isPosIncumb);

      setMatIntBounds(k, sl[k]->a, sl[k]->b);

      setMatIsCvdObsByBox(k);

      // set vecIsCovered, wether or not each observation is covered
      // by lower and upper bound (a, b)
      // setVecIsCovered(sl[k]->a, sl[k]->b);

      insertColumnClpModel(k);

    } // end for each solution, k

    numBoxesSoFar  += numBoxesIter;
    numCols        += numBoxesIter;

    // if (s.size()!=numBoxesIter) { // remove extra space due to duplicates
    //   matIntUpper.resize(matIntUpper.size()-(s.size()-numBoxesIter));
    //   matIntLower.resize(matIntLower.size()-(s.size()-numBoxesIter));
    //   vecIsObjValPos.resize(vecIsObjValPos.size()-(s.size()-numBoxesIter));
    // }

    for (unsigned int k=0; k<s.size(); ++k)
      sl[k]->dispose();

  } // end insertPebblColumns function


  // insert columns in each column iteration
  void REPR::insertGreedyColumns() { //const int& GreedyLevel) {

    numBoxesIter = 1;

    if ( curIter!=0 ) { // if not the first iteration
      // check whether or not the current iterations has any duplicate boxes
      if (debug>=1) {
        for (unsigned int k=0; k<numBoxesIter; ++k)
          checkDuplicateBoxes(grma->getLowerBounds(), grma->getUpperBounds());
      } // end debug
    } // end if not the first iteration

    vecIsObjValPos.resize(vecIsObjValPos.size()+numBoxesIter);
    matIntUpper.resize(matIntUpper.size()+numBoxesIter);
    matIntLower.resize(matIntLower.size()+numBoxesIter);

    setVecIsObjValPos(0, grma->isPostObjVal());

    setMatIntBounds(0, grma->getLowerBounds(), grma->getUpperBounds());

    setMatIsCvdObsByBox(0);

    insertColumnClpModel(0);

    // TODO: for now, add one observation per iteration, but can be fixed that later
    ++numBoxesSoFar;
    ++numCols;

  } // end insertGreedyColumns function


  // insert a colum in CLP model
  void REPR::insertColumnClpModel(const unsigned int &k) {

    matIsCvdObsByBox[numBoxesSoFar+k].resize(numObs);

    for (unsigned int i = 0; i < numObs; ++i) { // for each observation

      if (matIsCvdObsByBox[numBoxesSoFar+k][i]) { // if this observatoin is covered

        if (vecIsObjValPos[k]) { // if positive solution
      	  columnInsert[i]        = 1;
      	  columnInsert[numObs+i] = -1;
        } else {  // if not a positive solution
      	  columnInsert[i]        = -1;
      	  columnInsert[numObs+i] = 1;
      	} // end if positive or negative solution

      } else { // if this observation is not covered by k-th box
        columnInsert[i]        = 0;
        columnInsert[numObs+i] = 0;
      } // end if covered observation or not

    } // end for each observation

    // insert column
    //(numRows: # of rows, colIndex: column index,
    // columnInsert: column values to insert,
    // lower and upper bounds of this column variable = {0, COIN_DBL_MAX},
    // objective coefficient = {E})
    model.addColumn(numRows, colIndex, columnInsert, 0.0, COIN_DBL_MAX, E);

  } // end insertColumnClpModel function


  //////////////////////// Evaluating methods //////////////////////////////

  // evaluate error rate in each iteration
  double REPR::evaluateEachIter(const bool &isTest, vector<DataXy> origData) {

    double err, err2, actY, expY, mse=0.0;
    unsigned int obs, numIdx;

    // set the size of training or testing observations
    if (isTest) numIdx = data->vecTestObsIdx.size();
    else        numIdx = data->vecTrainObsIdx.size();

    for (unsigned int i=0; i<numIdx; ++i) { // for each obsercation

      // f(X) = \beta_0
      expY = vecPrimalVars[0];    // for constant terms
      DEBUGPR(20, cout << "constant expY: " << expY << "\n");

      if (isTest) obs = data->vecTestObsIdx[i];
      else        obs = data->vecTrainObsIdx[i];

      // f(X) += \sum_{j=1}^n ( \beta_j^+ - \beta_j^-)
      for (unsigned int j=0; j<numAttrib; ++j) { // for each attribute
        expY += ( origData[obs].X[j] - data->vecAvgX[j] ) / data->vecSdX[j]
                * ( vecPrimalVars[j+1] - vecPrimalVars[numAttrib+j+1] );
        DEBUGPR(20, cout << "linear expY: " << expY << "\n");
      } // end for each attribute

      DEBUGPR(20, cout << "obs: " << obs
                       << " linearReg expY: " << expY
                       <<  " features: " << origData[obs].X );

      // for each
      for (unsigned int k=0; k<matOrigLower.size(); ++k) { // for each box solution

        // //if (vecPrimalVars[numVar+2*k]!=0) {
        if (!(vecPrimalVars[data->numTrainObs+2*numAttrib+k+1] ==0) ) {
          if ( matIsCvdObsByBox[k][obs] ) { // if this observation is covered
            if (vecIsObjValPos[k]) // if positive box variable
              expY +=  vecPrimalVars[data->numTrainObs + 2*data->numAttrib +k+1] ;
            else
              expY += -vecPrimalVars[data->numTrainObs +2*data->numAttrib +k+1] ;
            DEBUGPR(20, cout << "kth box: " << k
                             << " box exp: " << expY << "\n");
          }
        } // end if for the coefficient of box not 0
      } // end for each box

      DEBUGPR(20, cout << "before normalied expY: " << expY
                       << ", avgY: "                <<  data->avgY
                       << ", sdY: "                 <<  data->sdY << "\n") ;

      expY = data->avgY + expY * data->sdY;
      actY = origData[obs].y;	// actual y value

      // if isSavePred is enabled and the last column generation iteration
      if ( isSavePred() && (curIter==getNumIterations()) ) {
        //predictions.resize(data->numOrigObs);
        //predictions[obs] = expY;
        savePredictions(TEST,  data->dataOrigTest);
        savePredictions(TRAIN, data->dataOrigTrain);
      }

      err = expY - actY;	// difference between expacted and actual y values
      err2 = pow(err, 2);

#ifdef ACRO_HAVE_MPI
      if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
        DEBUGPR(10, cout << "actY-expY " << actY << " - " << expY
                         << " = " << err << " err^2: " << err2 << "\n" ) ;
#ifdef ACRO_HAVE_MPI
      }
#endif //  ACRO_HAVE_MPI

      mse += err2;

    } // end for each observation

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      DEBUGPR(20, cout << "MSE: " <<  mse / (double) numIdx << "\n");
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    return mse / (double) numIdx;

  }	// end evaluateEachIter function


  // evaluate error rate in the end of iterations
  double REPR::evaluateAtFinal(const bool &isTest, vector<DataXy> origData) {

    double err, err2, actY, expY, mse=0.0;
    unsigned int i, j, k, obs, numIdx;

    if (isTest) numIdx = data->vecTestObsIdx.size();
    else        numIdx = data->vecTrainObsIdx.size();

    for (i=0; i<numIdx; ++i) { // for each obsercation

      expY = vecPrimalVars[0];    // for constant terms
      DEBUGPR(20, cout << "constant expY: " << expY << "\n");

      if (isTest) obs = data->vecTestObsIdx[i];
      else        obs = data->vecTrainObsIdx[i];

      for (j=0; j<numAttrib; ++j) {
        expY +=  ( origData[obs].X[j]-data->vecAvgX[j] ) / data->vecSdX[j]
          * ( vecPrimalVars[j+1] - vecPrimalVars[numAttrib+j+1] );
        DEBUGPR(20, cout << "linear expY: " << expY << "\n");
      }

      DEBUGPR(20, cout << "obs: " << obs << " linearReg expY: " << expY
              <<  " features: " << origData[obs].X );

      for (k=0; k<matOrigLower.size(); ++k) { // for each box solution

        // if the coefficients (gamma^+-gamma^-)=0 for the box[k] is not 0
        // if the coefficients for the box[k] is not 0
        if (!(vecPrimalVars[data->numTrainObs+2*numAttrib+k+1] ==0)) {//if (vecPrimalVars[numVar+2*k]!=0) {

          for (j=0; j<numAttrib; ++j) { // for each attribute
            if (matOrigLower[k][j] <= origData[obs].X[j] &&
        origData[obs].X[j] <= matOrigUpper[k][j] ) {
              if ( j==numAttrib-1) { // all features are covered by the box
        if (vecIsObjValPos[k])
          expY +=  vecPrimalVars[data->numTrainObs+2*numAttrib+k+1] ;
        else
          expY += -vecPrimalVars[data->numTrainObs+2*numAttrib+k+1] ;
        DEBUGPR(20, cout << "kth box: " << k	<< " box exp: " << expY << "\n");
              }
            } else break; // this observation is not covered
          } // end for each attribute

        } // end if for the coefficient of box not 0

      } // end for each box

      DEBUGPR(20, cout << "before normalied expY: " << expY
              << ",  avgY: " <<  data->avgY << ", sdY: " <<  data->sdY << "\n") ;

      expY = data->avgY + expY * data->sdY;
      actY = origData[obs].y;	// actual y value

      // if isSavePred is enabled and the last column generation iteration
      if ( isSavePred() && (curIter==getNumIterations()) ) {
        //predictions.resize(data->numOrigObs);
        //predictions[obs] = expY;
        savePredictions(TEST,  data->dataOrigTest);
        savePredictions(TRAIN, data->dataOrigTrain);
      }

      err = expY - actY;	// difference between expacted and actual y values
      err2 = pow(err, 2);

#ifdef ACRO_HAVE_MPI
      if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
        DEBUGPR(10, cout << "actY-expY " << actY << " - " << expY
        << " = " << err << " err^2: " << err2 << "\n" ) ;
#ifdef ACRO_HAVE_MPI
      }
#endif //  ACRO_HAVE_MPI

      mse += err2;

    } // end for each observation

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      DEBUGPR(20, cout << "mse: " <<  mse / (double) numIdx << "\n");
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    return mse / (double) numIdx;

  } // end evaluateAtFinal function


  //////////////////////// Printing methods //////////////////////////////

  // print solution for the restricted master problem
  void REPR::printRMPCheckInfo() {

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

      unsigned int i, j;
      double sumPrimal = 0;

      // for linear variables
      if (C != 0) { // if C is not 0
        for (j = 1; j < 1+numAttrib; ++j)
          sumPrimal += C * vecPrimalVars[j];  // + \beta^+ * X
        for (j = 1+numAttrib; j < 1+2*numAttrib; ++j)
          sumPrimal -= C * vecPrimalVars[j];  // - \beta^- * X
      }

      // for observation variable, episilon_i
      for (j = 1+2*numAttrib; j < 1+2*numAttrib+numObs; ++j) {
        if (P==1)      sumPrimal += vecPrimalVars[j];
        else if (P==2) sumPrimal += vecPrimalVars[j]*vecPrimalVars[j];
      }

      // if (D != 0) {
      //   for (j = 1; j < numAttrib+1; ++j)	// for linear square coefficients
      //     sumPrimal += D*vecPrimalVars[j]*vecPrimalVars[j];
      //   for (j = 1+numAttrib; j < 2*numAttrib+1; ++j)
      //     sumPrimal -= D*vecPrimalVars[j]*vecPrimalVars[j];
      // }

      cout << "vecPrimalVars: ";
      for (i=0; i<numCols; ++i) cout << vecPrimalVars[i] << " ";
      cout << "\n";

      ////////////////////////////////////////////////////////////////////

      double sumDual=0, sumDualCheck=0;

      for (i=0; i<numObs; i++)  { // for each observation

        sumDual += data->dataStandTrain[i].y
                   * ( vecDualVars[i] - vecDualVars[numObs+i] );
        // if (P==2)
        //   sumDual -= pow( ( vecDualVars[i] - vecDualVars[numObs+i] ), 2 ) / 4.0;

      }

      cout << "vecDualVars: ";
      for (i=0; i<numRows; ++i) cout << vecDualVars[i] << " ";
      cout << "\n";

      cout << "Check PrimalObj: " << sumPrimal << " = DualObj:" << sumDual << "\n";

      ////////////////////////////////////////////////////////////////////

      // for (i=0; i<numObs; i++)  // for each observation
      //   cout << "Check mu+nu=eps; mu: " << vecDualVars[i]
      //      << ", nu:" << vecDualVars[numObs+i]
      //      << ", eps: " << vecPrimalVars[2*numAttrib+1+i] << "\n" ;

      ////////////////////////////////////////////////////////////////////

      for (i=0; i<numObs; i++) // for each observation
        sumDualCheck += ( vecDualVars[i] - vecDualVars[numObs+i] );

      cout << "sumDualCheck: "  << sumDualCheck  << " (This should be 0.)\n";

      ////////////////////////////////////////////////////////////////////

      vector<double> vecSumConstCheck(numAttrib);
      fill(vecSumConstCheck.begin(), vecSumConstCheck.end(), 0);

      for (j=0; j<numAttrib; ++j)  // for each attribute
        for (i=0; i<numObs; ++i)   // for each observation
          vecSumConstCheck[j] += ( vecDualVars[i] - vecDualVars[numObs+i] )
                                 * data->dataStandTrain[i].X[j] ;

      cout << "vecSumConstCheck: "  << vecSumConstCheck
           << " (Each element should be >= -C.)\n";

#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

} // end printRMPCheckInfo function


  // print RMA information
  void REPR::printRMAInfo() {

  #ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
  #endif //  ACRO_HAVE_MPI
      DEBUGPR(20, cout <<  "E: " << E <<
              ", incumb: " << rma->incumbentValue<< "\n");
  #ifdef ACRO_HAVE_MPI
    }
  #endif //  ACRO_HAVE_MPI

  } // end printRMAInfo function

  // print the LHS constrraint elements
  void REPR::printClpElements() {

    CoinBigIndex idxClp = 0;

    cout << "element:\n";

    for (unsigned int j = 0; j < numCols; j++) { // for each column
      for (unsigned int  i = 0; i < numRows; ++i) { // for each row
        cout << elements[idxClp] << " ";  // print out the element
        idxClp++;
      } // end for each row
      cout << "\n";
    } // end for each column
    cout << "end element:\n";

  } // end printClpElements function


// save trained REPR model
void REPR::saveModel() {

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

  unsigned int i;

  // set the output file name
  stringstream s;
  s << problemName << "_model_" << getDateTime() << ".out";
  ofstream os(s.str().c_str());

  // save # of attributes and boxes
  os << "#_of_attributes: " << data->numAttrib << "\n";
  os << "#_of_boxes:      " << numBoxesSoFar << "\n";

  // output the constant term
  os << vecPrimalVars[0] << " ";

  // output the coefficients for the linear variables
  for (i=0; i<data->numAttrib; ++i)  // for each attribute
    os << vecPrimalVars[1+i] - vecPrimalVars[1+data->numAttrib+i] << " ";

  // output the cofficeitns for the box variables
  for (i=0; i<numBoxesSoFar; ++i) // for each box
    os << vecPrimalVars[1+2*data->numAttrib+numObs+i] << " ";

  os << "\n" ;  // go to the next line

  // output each box's lower and upper bounds in original values
  for (unsigned int k=0; k<curIter; ++k ) { // for each Boosting iteration

    if (matOrigLower.size()!=0) { // if integerized
      os << "Box " << k << "_a: "<< matOrigLower[k] << "\n" ;
      os << "Box " << k << "_b: "<< matOrigUpper[k] << "\n" ;
    } else {
      os << "Box_" << k << "_a: "<< matIntLower[k] << "\n" ;
      os << "Box_" << k << "_b: "<< matIntUpper[k] << "\n" ;
    } // end if

  } // end for each Boosting iteration

  // TODO: we do not have to save this info
  // output integerization info
  // for (unsigned int j=0; j<data->numAttrib; ++j) { // for each attribute
  //
  //   os << "Attrib_" << j << "_a: " ; // output lower bounds of bins
  //   for (unsigned int k=0; k<data->vecNumDistVals[j]; ++k)  // for each bin or value
  //     os << vecAttribIntInfo[j].vecBins[k].lowerBound << " ";
  //
  //   os << "\nAttrib_" << j << "_b: " ; // output upper bounds of bins
  //   for (unsigned int k=0; k<data->vecNumDistVals[j]; ++k)  // for each bin or value
  //     os << vecAttribIntInfo[j].vecBins[k].upperBound << " ";
  //
  // os << "\n" ;   // go to the next line
  //
  // } // end for each attribute
  //
  // os.close();

#ifdef ACRO_HAVE_MPI
  }
#endif //  ACRO_HAVE_MPI

  } // end saveModel function


} // namespace boosting
