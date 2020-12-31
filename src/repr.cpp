/*
 *  File name:   repr.cpp
 *  Author:      Ai Kagawa
 *  Description: a source file for REPR class
 */

#include "repr.h"

namespace boosting {


  // set REPR parameters
  void REPR::setBoostingParameters() {
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

    colIndex    = new int[numRows];  // column index
    // rowIndex    = new int[numCols];  // row index

    for (unsigned int i=0; i<numRows; ++i) // for each row
      colIndex[i] = i;

    // for (i=0; i<numCols; ++i) // for each column
    //   rowIndex[i] = i;

    //matrix.setDimensions(0, numCols);

    // This is fully dense - but would not normally be so
    numElements = numRows * numCols;

    elements = new double [numElements];  // elements in the constraints
    starts   = new CoinBigIndex [numCols+1];
    rows     = new int [numElements];;
    lengths  = new int [numCols];

    model.setOptimizationDirection(1);               // maximization

    // to turn off some output, 0 gives nothing and each increase
    // in value switches on more messages.
    model.setLogLevel(0);
    //matrix.setDimensions(numRows, numCols); // setDimensions (int numrows, int numcols)

    model.resize(numRows, numCols);

    // TODO: what are these?  Check CLP manual
    columnLower = model.columnLower();
    columnUpper = model.columnUpper();
    objective   = model.objective();
    rowLower    = model.rowLower();
    rowUpper    = model.rowUpper();

  } // end setInitRMPVariables function


  // set Initial RMP objective
  void REPR::setInitRMPObjective() {
    // set the objectives
    for (unsigned int k = 0; k < numCols; k++) { // end for each column
      if (k==0)                 objective[k] = 0.0;  // for the constnt term
      else if (k<1+2*numAttrib) objective[k] = C;    // for the linear variables
      else                      objective[k] = 1;    // for the observation variables
    } // end for each column

  } // end setInitRMPObjective


  // set LHS (left-hand side) constraints (elements),
  // and set starts, lengths, and rows
  void REPR::setConstraintsLHS() {

    unsigned int idx;
    CoinBigIndex idxClp = 0;

    for (unsigned int j = 0; j < numCols; j++) { // for each column

      starts[j]  = idxClp;    // set each column index starts
      lengths[j] = numRows;   // set each column

      for (unsigned int i = 0; i < numRows; ++i) { // for each row

	idx = (i <numObs) ? i : i-numObs;

	rows[idxClp] = i;     // set each element's row index

	if (j==0) // for the constant terms
	  elements[idxClp] = (i < numObs) ? 1.0 : -1.0;
	else if (j<1+numAttrib) {    // for positive linear variables
          if (i < numObs)
            elements[idxClp] = data->dataStandTrain[idxTrain(idx)].X[j-1];  // -1 for constant term
          else
	    elements[idxClp] = -data->dataStandTrain[idxTrain(idx)].X[j-1];
	} else if (j<1+2*numAttrib) { // for negative linear variables
          if (i < numObs)
            elements[idxClp] = -data->dataStandTrain[idxTrain(idx)].X[j-1-numAttrib];
          else
            elements[idxClp] = data->dataStandTrain[idxTrain(idx)].X[j-1-numAttrib];
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


  // set column bounds for RMP
  void REPR::setInitRMPColumnBound() {

    // set lower and upper bounds for each variables
    for (unsigned int k = 0; k < numCols; k++) { // for each colum
      columnLower[k] = (k==0) ? -COIN_DBL_MAX : 0.0;
      columnUpper[k] = COIN_DBL_MAX; //upperColumn[k];
    } // end each column

  } // end setInitRMPColumnBound function


  // set row bounds for RMP
  void REPR::setInitRMPRowBound() {
    for (unsigned int k = 0; k < numRows; k++) { // for each row
      if (k < numObs) {
	rowLower[k] = -COIN_DBL_MAX; //-inf;
	rowUpper[k] = data->dataStandTrain[idxTrain(k)].y;
      } else {
	rowLower[k] = -COIN_DBL_MAX; //-inf;
	rowUpper[k] = -data->dataStandTrain[idxTrain(k-numObs)].y;
      } // end if
    } // end each row
  } // end setInitRMPRowBound function


  // set CLP model
  void REPR::setInitRMPClpModel() {

    // assign to matrix
    matrix = new CoinPackedMatrix(true, 0.0, 0.0);
    matrix->assignMatrix(true, numRows, numCols, numElements,
			 elements, rows, starts, lengths);

    ClpPackedMatrix *clpMatrix = new ClpPackedMatrix(matrix);
    model.replaceMatrix(clpMatrix, true);

  } // end setInitRMPModel function

  /****************** set initial RMP in CLP (end) ***********************/

  // set a weight of each observation for RMA
  void REPR::setDataWts() {

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      // assign a weight for each observation ($/mu-/nu$, dual variables)
      for (unsigned int i=0; i < numObs ; ++i) // for each observation
      	SolveRMA::data->dataIntTrain[idxTrain(i)].w
          = vecDualVars[i]-vecDualVars[numObs+i];
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

#ifdef ACRO_HAVE_MPI

    for (unsigned int i = 0; i < numObs; ++i) { // for each observation

      if ((uMPI::rank==0)) { // if rank 0

	// If we are the root process, send the observation weights to everyone
        for (int k = 0; k < uMPI::size; ++k)
	  if (k != 0)
	    MPI_Send(&SolveRMA::data->dataIntTrain[idxTrain(i)].w,
                     1, MPI_DOUBLE, k, 0, MPI_COMM_WORLD);

      } else { // else (if not rank 0)

    	// If we are a receiver process, receive the data from the root
    	MPI_Recv(&SolveRMA::data->dataIntTrain[idxTrain(i)].w,
                 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
		 MPI_STATUS_IGNORE);

      } // end if

    } // end for each observation

#endif //  ACRO_HAVE_MPI

    DEBUGPR(1, ucout << "wt: ");
    DEBUGPR(1,
	    for (unsigned int i=0; i < numObs ; ++i) {
	      ucout << data->dataIntTrain[idxTrain(i)].w << ", ";
	    });
    DEBUGPR(1, ucout << "\n");

  } // end setDataWts function


  // check wether or not to spop the REPR iteration
  bool REPR::isStoppingCondition() {

    if (greedyLevel==EXACT) {  // if PEBBL RMA
      // if the incumbent is less tha E + threthold
      if (rma->incumbentValue <= E + .00001) {
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
  void REPR::insertExactColumns() { //const int& GreedyLevel) {

    setPebblRMASolutions();

    if ( curIter!=0 ) { // if not the first iteration
      // check whether or not the current iterations has any duplicate boxes
      DEBUGPR(1, isDuplicate());
    }

    vecIsObjValPos.resize(numBoxesSoFar+numBoxesIter);
    matIntUpper.resize(numBoxesSoFar+numBoxesIter);
    matIntLower.resize(numBoxesSoFar+numBoxesIter);

    for (unsigned int k=0; k<s.size(); ++k) { // for each solution

      //if (!isDuplicate(k)) { // if this rule is not duplicates

      // set vecIsCovered, wether or not each observation is covered
      // by lower and upper bound (a, b)
      setVecIsCovered(sl[k]->a, sl[k]->b);

      setVecIsObjValPos(k, sl[k]->isPosIncumb);

      setMatIntBounds(k, sl[k]->a, sl[k]->b);

      insertColumnClpModel(sl[k]->isPosIncumb);

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

  } // end insertExactColumns function


  // insert columns in each column iteration
  void REPR::insertGreedyColumns() { //const int& GreedyLevel) {

    numBoxesIter = 1;

    vecIsObjValPos.resize(vecIsObjValPos.size()+1);
    matIntUpper.resize(matIntUpper.size()+1);
    matIntLower.resize(matIntLower.size()+1);

    setVecIsCovered(grma->getLowerBounds(), grma->getUpperBounds());

    setVecIsObjValPos(0, grma->isPostObjVal());

    setMatIntBounds(0, grma->getLowerBounds(), grma->getUpperBounds());

    insertColumnClpModel(grma->isPostObjVal());

    // TODO: for now, add one observation per iteration, but can be fixed that later
    ++numBoxesSoFar;
    ++numCols;

  } // end insertGreedyColumns function


  // set vecIsCovered, whether or not each observation is vecCovered
  // by the lower and upper bounds, a and b
  void REPR::setVecIsCovered(const vector<unsigned int> &a,
                             const vector<unsigned int> &b) {

    for (unsigned int i=0; i< numObs; ++i) { // for each observation

      for (unsigned int j=0; j< numAttrib; ++j) { // for each attribute

        // if the current observation is covered by the current k'th box
        // for attribute k
        if ( a[j] <=  data->dataIntTrain[idxTrain(i)].X[j] &&
	       data->dataIntTrain[idxTrain(i)].X[j] <= b[j] ) {

          // if this observation is covered in all attributes
          if ( j==numAttrib-1)
	    vecIsCovered[i]= true;  // set this observation is covered

        } else { // if it is not covered
	  vecIsCovered[i]= false;  // set this observation is not covered
	  break;
	} // end if this observation is covered in attribute j

      } // end for each attribute

    } // end for each observation

    DEBUGPR(1, cout << "vecIsCovered: " << vecIsCovered << "\n" );

  } // end setVecIsCovered function


  // insert a colum in CLP model
  void REPR::insertColumnClpModel(const bool &isPosIncumb) {

    double *columnValue = new double[numRows];

    for (unsigned int i = 0; i < numRows; ++i)
      columnValue[i] = 0;

    for (unsigned int i = 0; i < numObs; ++i) { // for each observation

      if (vecIsCovered[i]==true) { // if this observatoin is covered

        if (isPosIncumb) { // if positive solution
      	  columnValue[i]        = 1;
      	  columnValue[numObs+i] = -1;
        } else {  // if not a positive solution
      	  columnValue[i]        = -1;
      	  columnValue[numObs+i] = 1;
      	} // end if positive or negative solution

      } // end if covered observation

    } // end for each observation

    //colIndex = colIndex + numBoxesSoFar;

    //numCols+numBoxesSoFar+k
    model.addColumn(numRows, colIndex, columnValue, 0.0, COIN_DBL_MAX, E);
    //model.addVar(0.0, GRB_INFINITY, E, GRB_CONTINUOUS, col);

  } // end insertColumnClpModel function


  void REPR::setVecIsObjValPos(const unsigned int &k, const bool &isPosObjVal) {
    if (isPosObjVal) { // if this solution is positive
      DEBUGPR(1, cout << "Positive Box\n");
      vecIsObjValPos[numBoxesSoFar+k] = true;
    } else { // if this solution is not positive
      DEBUGPR(1, cout << "Negative Box\n");
      vecIsObjValPos[numBoxesSoFar+k] = false;
    } // end if
  } // enf setVecIsObjValPos function


  void REPR::setMatIntBounds(const unsigned int &k,
        const vector<unsigned int> &lower, const vector<unsigned int> &upper) {
    matIntLower[matIntLower.size()-numBoxesIter+k] = lower;
    matIntUpper[matIntUpper.size()-numBoxesIter+k] = upper;
  } // end setVecIsObjValPos function


  // setPebblRMASolutions
  void REPR::setPebblRMASolutions() {

    rma->getAllSolutions(s);
    numBoxesIter = s.size();
    sl.resize(numBoxesIter);

    // for each boxes in current iteration
    for (unsigned int k=0; k<numBoxesIter; ++k)
      sl[k] = dynamic_cast<pebblRMA::rmaSolution*>(s[k]);

  } // end setPebblRMASolutions function


  //////////////////////// Evaluating methods //////////////////////////////

  // evaluate error rate in each iteration
  double REPR::evaluateEachIter(const bool &isTest, vector<DataXy> origData) {

    double err, err2, actY, expY, mse=0.0;
    unsigned int obs, size;

    if (isTest) size = data->vecTestObsIdx.size();
    else        size = data->vecTrainObsIdx.size();

    for (unsigned int i=0; i<size; ++i) { // for each obsercation

      expY = vecPrimalVars[0];    // for constant terms
      DEBUGPR(20, cout << "constant expY: " << expY << "\n");

      if (isTest) obs = data->vecTestObsIdx[i];
      else        obs = data->vecTrainObsIdx[i];

      for (unsigned int j=0; j<numAttrib; ++j) {
	expY +=  ( origData[obs].X[j]-data->vecAvgX[j] ) / data->vecSdX[j]
	  * ( vecPrimalVars[j+1] - vecPrimalVars[numAttrib+j+1] );
	DEBUGPR(20, cout << "linear expY: " << expY << "\n");
      }

      DEBUGPR(20, cout << "obs: " << obs << " linearReg expY: " << expY
	      <<  " features: " << origData[obs].X );

      for (unsigned int k=0; k<matOrigLower.size(); ++k) { // for each box solution
	if (!(vecPrimalVars[data->numTrainObs+2*numAttrib+k+1] ==0) ) {	//if (vecPrimalVars[numVar+2*k]!=0) {
	  if ( vecCoveredObsByBox[obs][k] ) { // if this observation is covered
	    if (vecIsObjValPos[k])
	      expY +=  vecPrimalVars[data->numTrainObs+2*data->numAttrib+k+1] ;
	    else
	      expY += -vecPrimalVars[data->numTrainObs+2*data->numAttrib+k+1] ;
	    DEBUGPR(20, cout << "kth box: " << k	<< " box exp: " << expY << "\n");
	  }
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
	savePredictions(TEST, data->dataOrigTest);
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
      DEBUGPR(20, cout << "mse: " <<  mse/(double)size << "\n");
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    return mse/(double)size;

  }	// evaluateEachIter function


  // evaluate error rate in the end of iterations
  double REPR::evaluateAtFinal(const bool &isTest, vector<DataXy> origData) {

    double err, err2, actY, expY, mse=0.0;
    unsigned int i, j, k, obs, size;

    if (isTest) size = data->vecTestObsIdx.size();
    else        size = data->vecTrainObsIdx.size();

    for (i=0; i<size; ++i) { // for each obsercation

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
	savePredictions(TEST, data->dataOrigTest);
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
      DEBUGPR(20, cout << "mse: " <<  mse/(double)size << "\n");
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    return mse/(double)size;

  } // end evaluateAtFinal function


  //////////////////////// Printing methods //////////////////////////////

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


  // print solution for the restricted master problem
  void REPR::printRMPSolution() {

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

      double sumPrimal=0.0;
      unsigned int i, j;

      if (C != 0) {
      	for (j = 1; j < numAttrib+1; ++j) sumPrimal += C*vecPrimalVars[j];
      }
      for (j = numAttrib+1; j < numCols; ++j) {
      	if (P==1)   	 sumPrimal += vecPrimalVars[j];
      	else if (P==2) sumPrimal += vecPrimalVars[j]*vecPrimalVars[j];
      }

      if (D != 0) {
      	for (j = 1; j < numAttrib+1; ++j)	// for linear square coefficients
      	  sumPrimal += D*vecPrimalVars[j]*vecPrimalVars[j];
      }

      double sumDual=0, sumCheck=0;
      vector<double> checkConst(numAttrib);

      for (i=0; i<numObs; i++)  { // for each observation
      	sumDual += data->dataStandTrain[idxTrain(i)].y * ( vecDualVars[i] - vecDualVars[numObs+i] );
      	if (P==2)  sumDual -= pow( ( vecDualVars[i] - vecDualVars[numObs+i] ), 2 ) / 4.0 ;
      	sumCheck += ( vecDualVars[i] - vecDualVars[numObs+i] );

      	DEBUGPR(1, cout << "mu: " << vecDualVars[i]
                        << ", nu:" << vecDualVars[numObs+i]
      		        << ", eps: " << vecPrimalVars[2*numAttrib+i+1] << "\n" );
      }

      for (j=0; j<numAttrib; ++j) { // for each attribute
      	for (i=0; i<numObs; ++i)  { // for each observation
      	  checkConst[j] += ( vecDualVars[i] - vecDualVars[numObs+i] )
	    * data->dataStandTrain[idxTrain(i)].X[j] ;
      	}
      }

      DEBUGPR(1, cout << "vecPrimalVars: " << vecPrimalVars
	      << " PrimalObj:" << sumPrimal << "\n" );
      DEBUGPR(1, for (i=0; i<numCols; ++i) cout << vecPrimalVars[i] << " "; );
      DEBUGPR(1, cout << "\nvecDualVarss: " << vecDualVars
	      << " DualObj:" << sumDual << "\n" );
      DEBUGPR(1, for (i=0; i<numRows; ++i) cout << vecDualVars[i] << " "; );
      DEBUGPR(1, cout << "\nsumCheck: " << sumCheck << "\n" );  // sum has to be 1
      DEBUGPR(1, cout << "checkCons: " << checkConst << "\n" );

#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

} // end print RMPSolution


// save trained REPR model
void REPR::saveModel() {

  unsigned int i;

  // set the output file name
  stringstream s;
  s << problemName << "_model_" << getDateTime() << ".out";
  ofstream os(s.str().c_str());

  DEBUGPR(1, cout << numCols << "\n");
  DEBUGPR(1, for (i=0; i<numCols; ++i)
               { cout << vecPrimalVars[i] << ", ";} );

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
  for (unsigned int k=0; k<getNumIterations(); ++k ) { // for each Boosting iteration
    // os << "Box " << k << "_a: "<< matOrigLower[k] << "\n" ;
    // os << "Box " << k << "_b: "<< matOrigUpper[k] << "\n" ;
    os << "Box_" << k << "_a: "<< matIntLower[k] << "\n" ;
    os << "Box_" << k << "_b: "<< matIntUpper[k] << "\n" ;
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

} // end saveModel function

} // namespace boosting
