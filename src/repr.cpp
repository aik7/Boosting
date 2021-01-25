/*
 *  Author:      Ai Kagawa
 *  Description: a source file for REPR class
 */

#include "repr.h"

namespace boosting {


  // set REPR parameters
  void REPR::setBoostingParameters() {
    P = isUseGurobi() ? getExponentP() : 1;
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

#ifdef HAVE_GUROBI
  if ( isUseGurobi() )
    setGurobiRMP();
  else {
#endif // HAVE_GUROBI

    /************************* using CLP **************************/

    setClpParameters();      // set CLP parameters

    setInitRMPObjective();   // set Objective

    setInitRMPColumnBound(); // set column bounds

    setInitRMPRowBound();    // set row bounds

    setConstraintsLHS();     // set the left hand side of the constraints

    setInitRMPClpModel();    // set the RMP using CLP

#if HAVE_GUROBI
  } // end Gurobi RMP
#endif // HAVE_GUROBI

  }  // end function REPR::setInitialMaster()


#ifdef HAVE_GUROBI

  void REPR::setGurobiRMP() {

    unsigned int i, j;
    columnLower = new double[numCols];
    columnUpper = new double[numCols];
    char* vtype = NULL;

    // DEBUGPRX(10, data, "Setup Initial Restricted Master Problem!" << "\n");

    modelGrb.getEnv().set(GRB_IntParam_Method, 0);

    columnLower[0] = -getInf(); // beta is free variable
    for (i = 1; i < numCols; ++i) columnLower[i] = 0;
    for (i = 0; i < numCols; ++i) columnUpper[i] = getInf();

    // Add variables to the model
    vars = modelGrb.addVars(columnLower, columnUpper,
                            NULL, vtype, NULL, numCols);

    // set constraits
    for (i = 0; i < data->numTrainObs; ++i) {
      lhs = vars[0];                       // beta_0
      for (j = 0; j < data->numAttrib; ++j)      // for positive linear variable, beta^+_j
        if (data->dataStandTrain[i].X[j] != 0)
          lhs += data->dataStandTrain[i].X[j]*vars[1+j];
      for (j = 0; j < data->numAttrib; ++j)      // for negative linear variable, beta^-_j
        if (data->dataStandTrain[i].X[j] != 0)
          lhs -= data->dataStandTrain[i].X[j]*vars[1+data->numAttrib+j];
      for (j = 0; j < data->numTrainObs; ++j)        // error for each observation, episilon
        if (i==j)
          lhs -= vars[1+2*data->numAttrib+j];

      modelGrb.addConstr(lhs, GRB_LESS_EQUAL, data->dataStandTrain[i].y);
    }

    for (i = 0; i < data->numTrainObs; ++i) {
      lhs = -vars[0];                      // bias, beta_0
      for (j = 0; j < data->numAttrib; ++j)      // for positive linear variable, beta^+_j
        if (data->dataStandTrain[i].X[j] != 0)
          lhs -= data->dataStandTrain[i].X[j]*vars[1+j];
      for (j = 0; j < data->numAttrib; ++j)      // for negative linear variable, beta^-_j
        if (data->dataStandTrain[i].X[j] != 0)
          lhs += data->dataStandTrain[i].X[j]*vars[1+data->numAttrib+j];
      for (j = 0; j < data->numTrainObs; ++j)         // error for each observation, episilon
        if (i==j)
          lhs -= vars[1+2*data->numAttrib+j];

      modelGrb.addConstr(lhs, GRB_LESS_EQUAL, -data->dataStandTrain[i].y);
    }

    // set cobjectives
    obj = 0;

    if (C != 0) { // if C is not 0
      for (j = 1; j < 2*data->numAttrib+1; ++j)
        obj += C*vars[j];
    } // end if C is not 0

    if (D != 0) { // if D is not 0
      for (j = 1; j < 2*data->numAttrib+1; ++j) // for linear square coefficients
        obj += D*vars[j]*vars[j];
    } // end if D is not 0

    for (j = 2*data->numAttrib+1; j < numCols; ++j) {
      if (P==1)      obj += vars[j];
      else if (P==2) obj += vars[j]*vars[j];
    }

    modelGrb.setObjective(obj); // optimization sense = None, minimization
    modelGrb.update();
    // modelGrb.write("master.lp");
    modelGrb.getEnv().set(GRB_IntParam_OutputFlag, 0);  // not to print out GUROBI

  } // end setGruobiRMP function

#endif // HAVE_GUROBI


  // set initial RMP variables
  void REPR::setInitRMPVariables() {

    // # of columns for RMP
    numCols = 1 + 2 * data->numAttrib + data->numTrainObs;      // vecPrimalVars.size();

    // # of rows
    numRows = 2 * data->numTrainObs ; // //NumVar+1; // +1 for constant term

    DEBUGPR(1, cout << "numAttrib:   "  << data->numAttrib << "\n");
    DEBUGPR(1, cout << "numTrainObs: "  << data->numTrainObs    << "\n");
    DEBUGPR(1, cout << "numCols:     "  << numCols   << "\n");

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

    modelClp.setOptimizationDirection(1); // 1:minimize, -1:maximize

    // to turn off some output, 0 gives nothing and each increase
    // in value switches on more messages.
    modelClp.setLogLevel(0);
    // matrix.setDimensions(numRows, numCols); // setDimensions (int numrows, int numcols)

    // set the dimension of the model
    modelClp.resize(numRows, numCols);

  } // end resetCLPModel function


  // set Initial RMP objective
  void REPR::setInitRMPObjective() {

    objective   = modelClp.objective();

    for (unsigned int k = 0; k < numCols; ++k) { // end for each column
      if (k==0)                 objective[k] = 0.0;  // for the constnt term
      else if (k<1+2*data->numAttrib) objective[k] = C;    // for the linear variables
      else                      objective[k] = 1;    // for the observation variables
    } // end for each column

  } // end setInitRMPObjective


  // set column bounds for RMP
  void REPR::setInitRMPColumnBound() {

    columnLower = modelClp.columnLower();
    columnUpper = modelClp.columnUpper();

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

    rowLower    = modelClp.rowLower();
    rowUpper    = modelClp.rowUpper();

    for (unsigned int k = 0; k < numRows; k++) { // for each row
      if (k < data->numTrainObs) {
        rowLower[k] = -COIN_DBL_MAX; //-inf;
        rowUpper[k] = data->dataStandTrain[k].y;
      } else {
        rowLower[k] = -COIN_DBL_MAX; //-inf;
        rowUpper[k] = -data->dataStandTrain[k-data->numTrainObs].y;
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

        idx = (i <data->numTrainObs) ? i : i-data->numTrainObs;

        rows[idxClp] = i;     // set each element's row index

        if (j==0) // for the constant terms
          elements[idxClp] = (i < data->numTrainObs) ? 1.0 : -1.0;

        else if (j<1+data->numAttrib) {    // for positive linear variables
          if (i < data->numTrainObs)
            elements[idxClp] = data->dataStandTrain[idx].X[j-1];  // -1 for constant term
          else
            elements[idxClp] = -data->dataStandTrain[idx].X[j-1];

        } else if (j<1+2*data->numAttrib) { // for negative linear variables
          if (i < data->numTrainObs)
            elements[idxClp] = -data->dataStandTrain[idx].X[j-1-data->numAttrib];
          else
            elements[idxClp] = data->dataStandTrain[idx].X[j-1-data->numAttrib];

        } else { // for observation error variables, episilon_i
          if (j-1-2*data->numAttrib==idx)
            elements[idxClp] = -1;
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
    modelClp.replaceMatrix(clpMatrix, true);

    // matrix.assignMatrix(true, numRows, numCols, numElements,
    //                       elements, rows, starts, lengths);
    // // clpMatrix = ClpPackedMatrix(matrix);
    // modelClp.loadProblem(matrix,
    //                   columnLower, columnUpper, objective,
    //                   rowLower, rowUpper);

  } // end setInitRMPModel function

  /****************** set initial RMP in CLP (end) ***********************/

  // set a weight of each observation for RMA
  // if rank 0, send the weights to the other ranks
  // if not rank 0, receive the weights from the rank 0
  void REPR::setWeights() {

    if (ROOTPROC) { // if root process

      // assign a weight for each observation ($/mu-/nu$, dual variables)
      for (unsigned int i=0; i < data->numTrainObs ; ++i) // for each observation
        data->dataIntTrain[i].w = vecDualVars[i] - vecDualVars[data->numTrainObs+i];

    } // end if root process

#ifdef ACRO_HAVE_MPI

    // broadcast weights
    for (unsigned int i = 0; i < data->numTrainObs; ++i) // for each observation
      MPI_Bcast(&data->dataIntTrain[i].w, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#endif //  ACRO_HAVE_MPI

    if (ROOTPROC) { // if root process

      if(debug>=1) {
        ucout << "Weight: "; // print out weights
        for (unsigned int i=0; i < data->numTrainObs ; ++i)
          ucout << data->dataIntTrain[i].w << ", ";
        ucout << "\n";
      } // end debug

    } // if root process

  } // end setWeights function


  // check wether or not to spop the REPR iteration
  bool REPR::isStoppingCondition() {

    if (greedyLevel==EXACT) {  // if PEBBL RMA
      // if the incumbent is less tha E + threthold
      if ( rma->incumbentValue <= E + getTolStopCond() ) {
        ucout << "PEBBL Stopping Condition!\n";
        return true;
      } // end if the stopping condition
    } // end if PEBBL RMA

    if (greedyLevel==GREEDY) { // if greedy RMA
      // if the current iteration is greater than 0,
      // and the current lower and upper bounads are the same
      // as the previous iteration's
      if (curIter>1
          && grma->getLowerBounds() == matIntLower[curIter-1]
          && grma->getUpperBounds() == matIntUpper[curIter-1] ) {
        ucout << "Greedy Stopping Condition!\n";
        return true;
      } // end if the stopping condition
    } // end if greedy RMA

    return false;

  } // end isStoppingCondition function


  // insert columns in each column iteration
  void REPR::insertColumns() {

    if (greedyLevel==GREEDY) numBoxesIter = 1;
    // numBoxesIter using PEBBL is already set

    vecIsObjValPos.       resize(numBoxesSoFar+numBoxesIter);
    matIntUpper.          resize(numBoxesSoFar+numBoxesIter);
    matIntLower.          resize(numBoxesSoFar+numBoxesIter);
    matIsCvdObsByBoxTrain.resize(numBoxesSoFar+numBoxesIter);

    for (unsigned int k=0; k<numBoxesIter; ++k) { // for each solution

      // 1st argument: k-th solutin in this iteration
      // 2nd argument: whether or not it is positive box variable
      // 3rd and 4th arguments: lower and upper bounds in integer value
      if (greedyLevel==EXACT) { // for PEEBL solution

        insertEachColumn(k, vecPebblRMASols[k]->isPosIncumb,
                          vecPebblRMASols[k]->a, vecPebblRMASols[k]->b);

        if (debug>=1)
          checkObjValue(data->dataIntTrain,
                        vecPebblRMASols[k]->a, vecPebblRMASols[k]->b);

        // insertEachColumn(k, rma->globalSol.isPosSol,
        //                  rma->globalSol.a, rma->globalSol.b);

      } else                    // for Greedy solution

        insertEachColumn(k, grma->isPostObjVal(),
                         grma->getLowerBounds(), grma->getUpperBounds());

    } // end for each solution, k

    numBoxesSoFar  += numBoxesIter;
    numCols        += numBoxesIter;

    if (greedyLevel==EXACT) // if PEEBL, dispose the solutions
      for (unsigned int k=0; k<vecPebblRMASols.size(); ++k)
        vecPebblRMASols[k]->dispose();

  } // end insertPebblColumns function


  // insert each column for current iteration k
  void REPR::insertEachColumn(const int & k, const bool &isPosObjVal,
                              const vector<unsigned int> &vecLower,
                              const vector<unsigned int> &vecUpper) {

    if (curIter!=0 && debug>=1) // if is not the first iteration and the debug mode
      checkDuplicateBoxes(vecLower, vecUpper);
      // check this box is duplicate of not compare to the already inserted boxes

    setVecIsObjValPos(k, isPosObjVal);      // set this box is positivie or not

    setMatIntBounds(k, vecLower, vecUpper); // set this box's bounds in the list

    setMatIsCvdObsByBox(k); // set wheather or not this box covered each observation

#ifdef HAVE_GUROBI
    if (isUseGurobi())
      insertColumnGurobiModel(k);  // insert k-th column of this iterations using Gurobi
    else
#endif // HAVE_GUROBI
      insertColumnClpModel(k);     // insert k-th column of this iterations using CLP

  } // end insertEachColumn function


  // insert a colum in CLP model
  void REPR::insertColumnClpModel(const unsigned int &k) {

    // matIsCvdObsByBox[numBoxesSoFar+k].resize(data->numTrainObs);

    for (unsigned int i = 0; i < data->numTrainObs; ++i) { // for each observation

      if (matIsCvdObsByBoxTrain[numBoxesSoFar+k][i]) { // if this observatoin is covered

        if (vecIsObjValPos[numBoxesSoFar+k]) { // if it's a positive box variable
          columnInsert[i]        = 1;
          columnInsert[data->numTrainObs+i] = -1;
        } else {  // if not a box variable
          columnInsert[i]        = -1;
          columnInsert[data->numTrainObs+i] = 1;
        } // end if positive or negative box variable

      } else { // if this observation is not covered by k-th box
        columnInsert[i]        = 0;
        columnInsert[data->numTrainObs+i] = 0;
      } // end if covered observation or not

    } // end for each observation

    // insert column
    //(numRows: # of rows, (2*data->numTrainObs)
    // colIndex: column index [0, 1, ... , 2*data->numTrainObs-1],
    // columnInsert: column values to insert,
    // lower and upper bounds of this column variable = {0, COIN_DBL_MAX},
    // objective coefficient = {E})
    modelClp.addColumn(numRows, colIndex, columnInsert, 0.0, COIN_DBL_MAX, E);

  } // end insertColumnClpModel function


#ifdef HAVE_GUROBI

  void REPR::insertColumnGurobiModel(const unsigned int &k) {

    // add columns using GUROBI
    col.clear();
    constr = modelGrb.getConstrs();

    for (unsigned int i = 0; i < data->numTrainObs; ++i) { // for each observation

      if (matIsCvdObsByBoxTrain[numBoxesSoFar+k][i]) { // if this observatoin is covered

        if (vecIsObjValPos[numBoxesSoFar+k]) { // if it's positive box variable
          col.addTerm( 1, constr[i]);
          col.addTerm(-1, constr[i+data->numTrainObs]);
        } else {
          col.addTerm(-1, constr[i]);
          col.addTerm( 1, constr[i+data->numTrainObs]);
        } // end if covered observation or not

      } // end if this observation is covered

    } // end for each observation

    // insert column
    // lower and upper bounds of this column variable = {0.0, GRB_INFINITY}
    // coefficient of the objective = {E}
    // inserting the column in the constraint = {col}
    modelGrb.addVar(0.0, GRB_INFINITY, E, GRB_CONTINUOUS, col);

  } // end insertColumnGurobiModel function

#endif // HAVE_GUROBI

  //////////////////////// Evaluating methods //////////////////////////////

  // evaluate error rate in each iteration
  double REPR::evaluate(const bool &isTrain, const vector<DataXy> &origData,
                        const deque<deque<bool> > &matIsCvdObsByBox) {

    double err, err2, actY, expY, mse=0.0;

    // set the size of training or testing observations
    unsigned int numObs = ( isTrain ? data->numTrainObs : data->numTestObs );

    for (unsigned int i=0; i<numObs; ++i) { // for each obsercation

      // f(X) = \beta_0
      expY = vecPrimalVars[0];    // for constant terms
      DEBUGPR(20, cout << "constant expY: " << expY << "\n");

      // f(X) += \sum_{j=1}^n ( \beta_j^+ - \beta_j^-)
      for (unsigned int j=0; j<data->numAttrib; ++j) { // for each attribute
        expY += ( origData[i].X[j] - data->vecAvgX[j] ) / data->vecSdX[j]
                * ( vecPrimalVars[j+1] - vecPrimalVars[data->numAttrib+j+1] );
        DEBUGPR(20, cout << "linear expY: " << expY << "\n");
      } // end for each attribute

      DEBUGPR(20, cout << "obs: "              << i
                       << ", linearReg expY: " << expY
                       << ", features: "       << origData[i].X );

      // for each
      for (unsigned int k=0; k<matOrigLower.size(); ++k) { // for each box solution

        // //if (vecPrimalVars[numVar+2*k]!=0) {
        // if the primal variable for this box variable is not 0
        if (!(vecPrimalVars[data->numTrainObs + 2*data->numAttrib +k+1] ==0) ) {

          if ( matIsCvdObsByBox[k][i] ) { // if this observation is covered

            if (vecIsObjValPos[k])          // if positive box variable
              expY +=  vecPrimalVars[data->numTrainObs + 2*data->numAttrib +k+1];
            else
              expY += -vecPrimalVars[data->numTrainObs + 2*data->numAttrib +k+1];

            DEBUGPR(20, cout << "kth box: "  << k
                             << " box exp: " << expY << "\n");

          } // end if this observation is covered

        } // end if for the coefficient of box not 0

      } // end for each box

      DEBUGPR(20, cout << "before normalied expY: " << expY
                       << ", avgY: "                << data->avgY
                       << ", sdY: "                 << data->sdY << "\n") ;

      // map back to the expected y value in its original value
      expY = data->avgY + expY * data->sdY;
      actY = origData[i].y;                  // actual y value

      // if isSavePred is enabled and
      //    ( the last column generation iteration
      //      or after hitting the stopping condition)
      // save prediction for each observation
      if ( isSavePred() && ( (curIter==getNumIterations()) || isStopCond ) ) {
        if (isTrain) vecPredTrain[i] = expY;
        else         vecPredTest[i]  = expY;
      }

      err  = expY - actY;  // difference between expacted and actual y values
      err2 = pow(err, 2);

      if (ROOTPROC) {// if root process
        DEBUGPR(10, cout << "actY-expY " << actY << " - " << expY
                         << " = " << err << " err^2: " << err2 << "\n" ) ;
      } // end if root process

      mse += err2;

    } // end for each observation

    mse /= (double) numObs;

    if (ROOTPROC) { DEBUGPR(20, cout << "MSE: " <<  mse << "\n"); }

    if ( isSaveErrors() ) { // store MSEs in a vector to save
      if (isTrain) vecErrTrain[curIter] = mse;
      else         vecErrTest [curIter] = mse;
    }

    return mse;

  }	// end evaluateEachIter function


  //////////////////////// Printing methods //////////////////////////////

  // print solution for the restricted master problem
  void REPR::printRMPCheckInfo() {

    unsigned int i, j;

    ////////////// print primal variables ////////////////////
    if(debug>=2) {

      cout << "/************** vecPrimalVars *************/";

      // output the constant term
      cout << "\nbias: " << std::fixed << std::setprecision(6) << vecPrimalVars[0];

      cout << "\n\npositive coefficients_for_linear_variables:\n";
      // output the coefficients for the linear variables
      for (i=0; i<data->numAttrib; ++i)  // for each attribute
        cout << vecPrimalVars[1+i] << " ";

      cout << "\n\nnegative coefficients_for_linear_variables:\n";
      // output the coefficients for the linear variables
      for (i=0; i<data->numAttrib; ++i)  // for each attribute
        cout << vecPrimalVars[1+data->numAttrib+i] << " ";

      cout << "\n\nerror terms:\n";
      // output the coefficients for the linear variables
      for (i=0; i<data->numTrainObs; ++i)  // for each attribute
        cout << vecPrimalVars[1+2*data->numAttrib+i] << " ";

      cout << "\n\ncoefficients_for_box_variables:\n";
      // output the cofficeitns for the box variables
      for (i=0; i<numBoxesSoFar; ++i) // for each box
        if (vecIsObjValPos[i])
          cout <<  vecPrimalVars[1+2*data->numAttrib+data->numTrainObs+i] << " ";
        else
          cout << -vecPrimalVars[1+2*data->numAttrib+data->numTrainObs+i] << " ";

      cout << "\n\n";

    }  // end if debug

    ////////////// print dual variables ////////////////////

    if (debug>=2) {

      cout << "/************** vecDualVars *************/";

      cout << "\nvecDualVars: \n";
      for (i=0; i<data->numTrainObs; ++i) cout << vecDualVars[i] << " ";
      cout << "\n";
      for (i=data->numTrainObs; i<2*data->numTrainObs; ++i) cout << vecDualVars[i] << " ";
      cout << "\n\n";

    } // end if debug

    ////////////// check primal solution = dual solution //////////////
    double sumPrimal = 0;

    // for linear variables
    if (C != 0) { // if C is not 0
      for (j = 1; j < 1+2*data->numAttrib; ++j)
        sumPrimal += C * vecPrimalVars[j];  // + \beta^+ * X
    }

    // for observation variable, episilon_i
    for (j = 1+2*data->numAttrib; j < 1+2*data->numAttrib+data->numTrainObs; ++j) {
      if (P==1)      sumPrimal += vecPrimalVars[j];
      else if (P==2) sumPrimal += vecPrimalVars[j]*vecPrimalVars[j];
    }

    for (i=0; i<numBoxesSoFar; ++i) // for each observation
        sumPrimal += E * vecPrimalVars[1+2*data->numAttrib+data->numTrainObs+i];

    // if (D != 0) {
    //   for (j = 1; j < data->numAttrib+1; ++j)	// for linear square coefficients
    //     sumPrimal += D*vecPrimalVars[j]*vecPrimalVars[j];
    //   for (j = 1+data->numAttrib; j < 2*data->numAttrib+1; ++j)
    //     sumPrimal -= D*vecPrimalVars[j]*vecPrimalVars[j];
    // }

    double sumDual = 0;

    for (i=0; i<data->numTrainObs; i++)  { // for each observation

      sumDual += data->dataStandTrain[i].y
                 * ( vecDualVars[i] - vecDualVars[data->numTrainObs+i] );
      if (P==2)
        sumDual -= pow( ( vecDualVars[i] - vecDualVars[data->numTrainObs+i] ), 2 ) / 4.0;

    } // end for each observation

    cout << "\nCheck PrimalObj: " << sumPrimal
         << " = DualObj:" << sumDual << "\n";

    ////////////////////////////////////////////////////////////////////

    // for (i=0; i<data->numTrainObs; i++)  // for each observation
    //   cout << "Check mu+nu=eps; mu: " << vecDualVars[i]
    //      << ", nu:" << vecDualVars[data->numTrainObs+i]
    //      << ", eps: " << vecPrimalVars[2*data->numAttrib+1+i] << "\n" ;

    ////////////////////////////////////////////////////////////////////

    double sumDualCheck=0;

    for (i=0; i<data->numTrainObs; i++) // for each observation
      sumDualCheck += ( vecDualVars[i] - vecDualVars[data->numTrainObs+i] );

    cout << "sumDualCheck: "  << sumDualCheck  << " (This should be 0.)\n";

    ////////////////////////////////////////////////////////////////////

    vector<double> vecSumConstCheck(data->numAttrib);
    fill(vecSumConstCheck.begin(), vecSumConstCheck.end(), 0);

    for (j=0; j<data->numAttrib; ++j)  // for each attribute
      for (i=0; i<data->numTrainObs; ++i)   // for each observation
        vecSumConstCheck[j] += ( vecDualVars[i] - vecDualVars[data->numTrainObs+i] )
                               * data->dataStandTrain[i].X[j] ;

    cout << "vecSumConstCheck: "  << vecSumConstCheck
         << " (Each element should be >= -C.)\n";

  } // end printRMPCheckInfo function


  // print RMA information
  void REPR::printRMAInfo() {

      DEBUGPR(20, cout <<  "E: " << E <<
              ", incumb: " << rma->incumbentValue<< "\n");

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

    unsigned int i;

    // set the output file name
    stringstream s;

    s << outputDir() << "/" << "model_" << problemName << ".out"; // << getDateTime()
    ofstream os(s.str().c_str());

    // save # of attributes and boxes
    os << "#_of_attributes: " << data->numAttrib << "\n";
    os << "#_of_boxes:      " << numBoxesSoFar << "\n";

    // output the constant term
    os << "\nbias: " << vecPrimalVars[0];

    os << "\n\ncoefficients_for_linear_variables:\n";
    // output the coefficients for the linear variables
    for (i=0; i<data->numAttrib; ++i)  // for each attribute
      os << vecPrimalVars[1+i] - vecPrimalVars[1+data->numAttrib+i] << " ";

    os << "\n\ncoefficients_for_box_variables:\n";
    // output the cofficeitns for the box variables
    for (i=0; i<numBoxesSoFar; ++i) // for each box
      if (vecIsObjValPos[i])
        os <<   vecPrimalVars[1+2*data->numAttrib+data->numTrainObs+i] << " ";
      else
        os << - vecPrimalVars[1+2*data->numAttrib+data->numTrainObs+i] << " ";

    os << "\n\nthe_average_value_of_y_value: ";
    os << data->avgY;

    os << "\n\nthe_standard_deviation_of_y_value: ";
    os << data->sdY;

    os << "\n\nthe_average_value_of_each_attribute:\n";
    for (unsigned int j=0; j<data->numAttrib; ++j)
      os << data->vecAvgX[j] << " ";

    os << "\n\nthe_standard_deviation_of_each_attribute:\n";
    for (unsigned int j=0; j<data->numAttrib; ++j)
      os << data->vecSdX[j] << " ";

    os << "\n\n" ;  // go to the next line

    // output each box's lower and upper bounds in original values
    for (unsigned int k=0; k<curIter-1; ++k ) { // for each Boosting iteration

      if (matOrigLower.size()!=0) { // if integerized

        os << "\nBox_" << k << "_a: " ;
        for (unsigned int j=0; j<data->numAttrib; ++j)
          os << matOrigLower[k][j] << " " ;

        os << "\nBox_" << k << "_b: " ;
        for (unsigned int j=0; j<data->numAttrib; ++j)
          os << matOrigUpper[k][j] << " " ;

      } else { // if data is not integerized for RMA

        os << "Box_" << k << "_a: " ;
        for (unsigned int j=0; j<data->numAttrib; ++j) { // for each attribute
          // if lower bound is 0, set -inf
          if (matIntLower[k][j]==0)
            os << -getInf() << " ";
          else
            os << matIntLower[k][j] << " ";
        } // end for each attribute

        os << "\nBox_" << k << "_b: " ;
        for (unsigned int j=0; j<data->numAttrib; ++j) {  // for each attribute
          // if lower bound is the maximum value, set inf
          if (matIntUpper[k][j]==data->vecNumDistVals[j]-1)
            os << getInf() << " ";
          else
            os << matIntUpper[k][j] << " ";
        } // end for each attribute

        os << "\n" ;

      } // end if

    } // end for each Boosting iteration

    os.close();

  } // end saveModel function

} // namespace boosting
