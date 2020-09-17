/*
 *  File name:   repr.cpp
 *  Author:      Ai Kagawa
 *  Description: a source file for REPR class
 */

#include "repr.h"

namespace boosting {

  ///////////////////////// Training methods /////////////////////////

  void REPR::setBoostingParameters() {
    C = getCoefficientC();
    D = 0; //getCoefficientD();
    E = getCoefficientE();
    F = 0; //getCoefficientF();
  }


  bool REPR::isStoppingCondition() {
    if (greedyLevel==EXACT) {
      if (rma->incumbentValue <= E + .00001) {return true;}
    }
    if (greedyLevel==Greedy) {
      if (curIter>0 && grma->L == matIntLower[curIter-1] &&
	  grma->U == matIntUpper[curIter-1] ) {
	ucout << "Stopping Condition!\n";
	return true;
      }
    }
    return false;
  }


  // set up for the initial master problem
  void REPR::setInitRMP() {

    DEBUGPR(10, cout << "Setup Initial Restricted Master Problem!" << "\n");

    int i, j, k, obs;
    numCols = 1+2*NumAttrib+NumObs; // vecPrimal.size();
    numRows = isLPBoost() ? NumObs+1 : 2*NumObs ; // //NumVar+1;	// +1 for constant term

    objValue    = new double[numCols];
    lowerColumn = new double[numCols];
    upperColumn = new double[numCols];
    lowerRow    = new double[numRows];
    upperRow    = new double[numRows];
    dataWts     = new double[NumObs];

    colIndex    = new int[numRows];
    rowIndex    = new int[numCols];
    for (i=0; i<numRows; ++i)
      colIndex[i] = i;
    for (i=0; i<numCols; ++i)
      rowIndex[i] = i;

    //matrix.setDimensions(0, numCols);

    model.setOptimizationDirection(1);               // maximization
    model.setLogLevel(0); // to turn off some output, 0 gives nothing and each increase in value switches on more messages.
    //matrix.setDimensions(numRows, numCols); // setDimensions (int numrows, int numcols)

    // Create space for 3 columns and 10000 rows
    //int numberRows = 10000;
    //int numberColumns = 3;
    // This is fully dense - but would not normally be so
    unsigned long long int numberElements = numRows * numCols;
    // Arrays will be set to default values
    model.resize(numRows, numCols);

    double * elements      = new double [numberElements];
    CoinBigIndex * starts  = new CoinBigIndex [numCols+1];
    int * rows             = new int [numberElements];;
    int * lengths          = new int[numCols];
    // Now fill in - totally unsafe but ....
    // no need as defaults to 0.0 double * columnLower = model2.columnLower();
    double * columnLower = model.columnLower();
    double * columnUpper = model.columnUpper();
    double * objective   = model.objective();
    double * rowLower    = model.rowLower();
    double * rowUpper    = model.rowUpper();

    // Columns - objective was packed
    for (k = 0; k < numCols; k++) {
      if (k==0)                 objective[k] = 0.0;
      else if (k<1+2*NumAttrib) objective[k] = C;
      else                      objective[k] = 1.0;
    }

    for (k = 0; k < numCols; k++) {
      columnLower[k] = (k==0) ? -COIN_DBL_MAX : 0.0;
      columnUpper[k] = COIN_DBL_MAX; //upperColumn[k];
    }

    // Rows
    for (k = 0; k < numRows; k++) {
      if (k < NumObs) {
	obs = data->vecTrainData[k];
	rowLower[k] = -COIN_DBL_MAX; //-inf;
	rowUpper[k] = data->standTrainData[obs].y;
      } else {
	obs = data->vecTrainData[k-NumObs];
	rowLower[k] = -COIN_DBL_MAX; //-inf;
	rowUpper[k] = -data->standTrainData[obs].y;
      }
    }

    //double rowValue[] = {1.0, -5.0, 1.0};
    CoinBigIndex put = 0;
    for (j = 0; j < numCols; j++) { // for each column
      starts[j]  = put;
      lengths[j] = numRows;
      for (int i = 0; i < numRows; ++i) { // for each row
	int index = (i <NumObs) ? i : i-NumObs;
	obs = data->vecTrainData[index];
	rows[put] = i;
	if (j==0)
	  elements[put] = (i < NumObs) ? 1.0 : -1.0;
	else if (j<1+NumAttrib)
	  elements[put] = (i < NumObs) ? data->standTrainData[obs].X[j-1]
	    : -data->standTrainData[obs].X[j-1];
	else if (j<1+2*NumAttrib)
	  elements[put] = (i < NumObs) ? -data->standTrainData[obs].X[j-1-NumAttrib]
	    : data->standTrainData[obs].X[j-1-NumAttrib];
	else // episilon (error variable)
	  if (j-1-2*NumAttrib==index)  elements[put] = (j < NumObs) ? 1.0 : -1.0;
	  else                         elements[put] = 0;
	put++;
      } // end for each row
    } // end for each column
    starts[numCols] = put;

    if (debug>=100) {
      cout << "element:\n";
      put = 0;
      for (j = 0; j < numCols; j++) {
	for (int i = 0; i < numRows; ++i) {
	  cout << elements[put] << " ";
	  put++;
	}
	cout << endl;
      }
      cout << "end element:\n";
    }

    // assign to matrix
    matrix = new CoinPackedMatrix(true, 0.0, 0.0);
    matrix->assignMatrix(true, numRows, numCols, numberElements,
			 elements, rows, starts, lengths);
    //CoinPackedMatrix * matrix = new CoinPackedMatrix(true, 0.0, 0.0);
    //matrix->assignMatrix(true, numRows, numCols, numberElements,
    //                    elements, rows, starts, lengths);
    ClpPackedMatrix *clpMatrix = new ClpPackedMatrix(matrix);
    model.replaceMatrix(clpMatrix, true);
    //printf("Time for 10000 addRow using hand written code is %g\n", CoinCpuTime() - time1);
    // If matrix is really big could switch off creation of row copy
    // model2.setSpecialOptions(256);

  }  // end function REPR::setInitialMaster()


  void REPR::setDataWts() {

    int obs;

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      for (int i=0; i < NumObs ; ++i) {
      	obs = data->vecTrainData[i];
      	data->intTrainData[obs].w = vecDual[i]-vecDual[NumObs+i];
      }
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

#ifdef ACRO_HAVE_MPI
    int i, k;
    for (i = 0; i < data->numTrainObs; ++i) {

      if ((uMPI::rank==0)) {

	// If we are the root process, send our data to everyone
	for (k = 0; k < uMPI::size; ++k)
	  if (k != 0)
	    MPI_Send(&data->intTrainData[i].w, 1, MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
    } else {

    	// If we are a receiver process, receive the data from the root
    	MPI_Recv(&data->intTrainData[i].w, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
		           MPI_STATUS_IGNORE);
    }
  }
#endif //  ACRO_H

    DEBUGPR(1, ucout << "wt: ");
    DEBUGPR(1,
	    for (int i=0; i < NumObs ; ++i) {
	      obs = data->vecTrainData[i];
	      ucout << data->intTrainData[obs].w << ", ";
	    });
    DEBUGPR(1, ucout << "\n");

  }


  // insert columns in each column iteration
  void REPR::insertExactColumns() { //const int& GreedyLevel) {

    int obs;
    numRMASols=0;

    ///////////////////////////// constraints /////////////////////////////
    rma->getAllSolutions(s);
    sl.resize(s.size());

    for (int k=0; k<s.size(); ++k)
      sl[k] = dynamic_cast<pebblRMA::rmaSolution*>(s[k]);

    if ( curIter!=0 )
      for (int k=0; k<s.size(); ++k)
	if ( vecCoveredSign[numBox-1]==sl[k]->isPosIncumb
	     && matIntLower[matIntLower.size()-1] == sl[k]->a
	     && matIntUpper[matIntUpper.size()-1] == sl[k]->b ) {
	  flagDuplicate=true;
#ifdef ACRO_HAVE_MPI
	  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
	    ucout << "insertColumns::Duplicated!\n";
#ifdef ACRO_HAVE_MPI
	  }
#endif //  ACRO_HAVE_MPI
	  return;
	}

    vecCoveredSign.resize(vecCoveredSign.size()+s.size());
    matIntUpper.resize(matIntUpper.size()+s.size());
    matIntLower.resize(matIntLower.size()+s.size());

    for (int k=0; k<s.size(); ++k) { // for each solution

      //if (k==0 && !isDuplicate()) { // if this rule is not duplicates

      for (int i=0; i< NumObs; ++i) { // for each observation
      	obs = data->vecTrainData[i];
      	for (int j=0; j< NumAttrib; ++j) { // for each attribute
      	  if ( sl[k]->a[j] <=  data->intTrainData[obs].X[j] &&
      	       data->intTrainData[obs].X[j] <= sl[k]->b[j] ) {
      	    if ( j==NumAttrib-1)
      	      vecIsCovered[i]= true;
      	  } else {
      	    vecIsCovered[i]= false;
      	    break;
      	  }
      	} // end for each attribute, j
      } // end for each observation, i

      if (sl[k]->isPosIncumb) {
      	DEBUGPR(1, cout << "Positive Box\n");
      	vecCoveredSign[numBox+numRMASols] = true;
      } else {
      	DEBUGPR(1, cout << "Negative Box\n");
      	vecCoveredSign[numBox+numRMASols] = false;
      }

      matIntLower[matIntLower.size()-s.size()+k] = sl[k]->a;
      matIntUpper[matIntUpper.size()-s.size()+k] = sl[k]->b;
      ++numRMASols;  // number of RMA solutions

      DEBUGPR(1, cout << "vecIsCovered: " << vecIsCovered << "\n" );

      // add columns using CLP

      double *columnValue = new double[numRows];
      for (int i = 0; i < numRows; ++i) columnValue[i] = 0;

      for (int i = 0; i < NumObs; ++i) {
      	obs = data->vecTrainData[i];
      	if (vecIsCovered[i]==true) {
      	  if (sl[k]->isPosIncumb) {
      	    columnValue[i]        = 1;
      	    columnValue[NumObs+i] = -1;
      	  } else {
      	    columnValue[i]        = -1;
      	    columnValue[NumObs+i] = 1;
      	  }
      	}
      }

      //colIndex = colIndex + numBox;

      //numCols+numBox+k
      model.addColumn(numRows, colIndex, columnValue, 0.0, COIN_DBL_MAX, E);
      //model.addVar(0.0, GRB_INFINITY, E, GRB_CONTINUOUS, col);

      //} // end if duplicate rules

    } // end for each solution, k

    numBox  += numRMASols;
    numCols += numRMASols;

    if (s.size()!=numRMASols) { // remove extra space due to duplicates
      matIntUpper.resize(matIntUpper.size()-(s.size()-numRMASols));
      matIntLower.resize(matIntLower.size()-(s.size()-numRMASols));
      vecCoveredSign.resize(vecCoveredSign.size()-(s.size()-numRMASols));
    }

    for (int k=0; k<s.size(); ++k)
      sl[k]->dispose();

  }


  // insert columns in each column iteration
  void REPR::insertGreedyColumns() { //const int& GreedyLevel) {

    int obs;
    numRMASols=0;

    ///////////////////////////// constraints /////////////////////////////
    vecCoveredSign.resize(vecCoveredSign.size()+1);
    matIntUpper.resize(matIntUpper.size()+1);
    matIntLower.resize(matIntLower.size()+1);

    for (int i=0; i< NumObs; ++i) { // for each observation
      obs = data->vecTrainData[i];
      for (int j=0; j< NumAttrib; ++j) { // for each attribute
	if ( grma->L[j] <=  data->intTrainData[obs].X[j] &&
	     data->intTrainData[obs].X[j] <= grma->U[j] ) {
	  if ( j==NumAttrib-1)
	    vecIsCovered[i]= true;
	} else {
	  vecIsCovered[i]= false;
	  break;
	}
      } // end for each attribute, j
    } // end for each observation, i

    if (grma->isPosIncumb) {
      DEBUGPR(1, cout << "Positive Box\n");
      vecCoveredSign[numBox] = true;
    } else {
      DEBUGPR(1, cout << "Negative Box\n");
      vecCoveredSign[numBox] = false;
    }

    matIntLower[matIntLower.size()-1] = grma->L;
    matIntUpper[matIntUpper.size()-1] = grma->U;

    DEBUGPR(1, cout << "vecIsCovered: " << vecIsCovered );

    /*
    // add columns using GUROBI
    //col.clear();
    //constr = model.getConstrs();
    for (int i = 0; i < NumObs; ++i) {
      obs = data->vecTrainData[i];
      if (vecIsCovered[i]==true) {
	if (grma->isPosIncumb) {
	  //col.addTerm(1, constr[i]);
	  //col.addTerm(-1, constr[i+NumObs]);
	} else {
	  //col.addTerm(-1, constr[i]);
	  //col.addTerm(1, constr[i+NumObs]);
	}
      }
    }
    */
    //model.addVar(0.0, GRB_INFINITY, E, GRB_CONTINUOUS, col);

    double *columnValue = new double[numRows];
    for (int i = 0; i < numRows; ++i) columnValue[i] = 0;

    for (int i = 0; i < NumObs; ++i) {
    	obs = data->vecTrainData[i];
    	if (vecIsCovered[i]==true) {
    	  if (grma->isPosIncumb) {
    	    columnValue[i]        = 1;
    	    columnValue[NumObs+i] = -1;
    	  } else {
    	    columnValue[i]        = -1;
    	    columnValue[NumObs+i] = 1;
    	  }
    	}
    }

    model.addColumn(numRows, colIndex, columnValue, 0.0, COIN_DBL_MAX, E);

    ++numBox;
    ++numCols;

  }


  //////////////////////// Evaluating methods //////////////////////////////

  // evaluate error rate in each iteration
  double REPR::evaluateEachIter(const int & isTest, vector<DataXy> origData) {

    double err, err2, actY, expY, mse=0.0;
    int obs, size;

    if (isTest) size = data->vecTestData.size();
    else        size = data->vecTrainData.size();

    for (int i=0; i<size; ++i) { // for each obsercation

      expY = vecPrimal[0];    // for constant terms
      DEBUGPR(20, cout << "constant expY: " << expY << "\n");

      if (isTest) obs = data->vecTestData[i];
      else        obs = data->vecTrainData[i];

      for (int j=0; j<NumAttrib; ++j) {
	expY +=  ( origData[obs].X[j]-data->avgX[j] ) / data->sdX[j]
	  * ( vecPrimal[j+1] - vecPrimal[NumAttrib+j+1] );
	DEBUGPR(20, cout << "linear expY: " << expY << "\n");
      }

      DEBUGPR(20, cout << "obs: " << obs << " linearReg expY: " << expY
	      <<  " features: " << origData[obs].X );

      for (int k=0; k<matOrigLower.size(); ++k) { // for each box solution
	if (!(vecPrimal[data->numTrainObs+2*NumAttrib+k+1] ==0) ) {	//if (vecPrimal[numVar+2*k]!=0) {
	  if ( vecCoveredObsByBox[obs][k] ) { // if this observation is covered
	    if (vecCoveredSign[k])
	      expY +=  vecPrimal[data->numTrainObs+2*data->numAttrib+k+1] ;
	    else
	      expY += -vecPrimal[data->numTrainObs+2*data->numAttrib+k+1] ;
	    DEBUGPR(20, cout << "kth box: " << k	<< " box exp: " << expY << "\n");
	  }
	} // end if for the coefficient of box not 0
      } // end for each box

      DEBUGPR(20, cout << "before normalied expY: " << expY
	      << ",  avgY: " <<  data->avgY << ", sdY: " <<  data->sdY << "\n") ;

      expY = data->avgY + expY * data->sdY;
      actY = origData[obs].y;	// actual y value

      // if writePred is enabled and the last column generation iteration
      if ( writePred() && (curIter==NumIter) ) {
	//predictions.resize(data->numOrigObs);
	//predictions[obs] = expY;
	writePredictions(TEST, data->origTestData);
	writePredictions(TRAIN, data->origTrainData);
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
  double REPR::evaluateAtFinal(const int & isTest, vector<DataXy> origData) {

    double err, err2, actY, expY, mse=0.0;
    int obs, size;

    if (isTest) size = data->vecTestData.size();
    else        size = data->vecTrainData.size();

    for (int i=0; i<size; ++i) { // for each obsercation

      expY = vecPrimal[0];    // for constant terms
      DEBUGPR(20, cout << "constant expY: " << expY << "\n");

      if (isTest) obs = data->vecTestData[i];
      else        obs = data->vecTrainData[i];

      for (int j=0; j<NumAttrib; ++j) {
	expY +=  ( origData[obs].X[j]-data->avgX[j] ) / data->sdX[j]
	  * ( vecPrimal[j+1] - vecPrimal[NumAttrib+j+1] );
	DEBUGPR(20, cout << "linear expY: " << expY << "\n");
      }

      DEBUGPR(20, cout << "obs: " << obs << " linearReg expY: " << expY
	      <<  " features: " << origData[obs].X );

      for (int k=0; k<matOrigLower.size(); ++k) { // for each box solution

	// if the coefficients (gamma^+-gamma^-)=0 for the box[k] is not 0
	// if the coefficients for the box[k] is not 0
	if (!(vecPrimal[data->numTrainObs+2*NumAttrib+k+1] ==0)) {//if (vecPrimal[numVar+2*k]!=0) {

	  for (int j=0; j<NumAttrib; ++j) { // for each attribute
	    if (matOrigLower[k][j] <= origData[obs].X[j] &&
		origData[obs].X[j] <= matOrigUpper[k][j] ) {
	      if ( j==NumAttrib-1) { // all features are covered by the box
		if (vecCoveredSign[k])
		  expY +=  vecPrimal[data->numTrainObs+2*NumAttrib+k+1] ;
		else
		  expY += -vecPrimal[data->numTrainObs+2*NumAttrib+k+1] ;
		DEBUGPR(20, cout << "kth box: " << k	<< " box exp: " << expY << "\n");
	      }
	    } else break; // this observation is not covered
	  } // end for each attribute, j

	}	// end if for the coefficient of box not 0

      } // end for each box

      DEBUGPR(20, cout << "before normalied expY: " << expY
	      << ",  avgY: " <<  data->avgY << ", sdY: " <<  data->sdY << "\n") ;

      expY = data->avgY + expY * data->sdY;
      actY = origData[obs].y;	// actual y value

      // if writePred is enabled and the last column generation iteration
      if ( writePred() && (curIter==NumIter) ) {
	//predictions.resize(data->numOrigObs);
	//predictions[obs] = expY;
	writePredictions(TEST, data->origTestData);
	writePredictions(TRAIN, data->origTrainData);
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

  }	// evaluateAtFinal function


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

  }


  void REPR::printRMPSolution() {

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

      double sumPrimal=0.0;
      int i,j, obs;

      if (C != 0) {
      	for (j = 1; j < NumAttrib+1; ++j) sumPrimal += C*vecPrimal[j];
      }
      for (j = NumAttrib+1; j < numCols; ++j) {
      	if (P==1)   	 sumPrimal += vecPrimal[j];
      	else if (P==2) sumPrimal += vecPrimal[j]*vecPrimal[j];
      }

      if (D != 0) {
      	for (j = 1; j < NumAttrib+1; ++j)	// for linear square coefficients
      	  sumPrimal += D*vecPrimal[j]*vecPrimal[j];
      }

      double sumDual=0, sumCheck=0;
      vector<double> checkConst(NumAttrib);
      for (i=0; i<NumObs; i++)  {
      	obs = data->vecTrainData[i];
      	sumDual += data->standTrainData[obs].y * ( vecDual[i] - vecDual[NumObs+i] );
      	if (P==2)  sumDual -= pow( ( vecDual[i] - vecDual[NumObs+i] ), 2 ) / 4.0 ;
      	sumCheck += ( vecDual[i] - vecDual[NumObs+i] );

      	DEBUGPR(1, cout << "mu: " << vecDual[i] << " nu:" << vecDual[NumObs+i]
      		<< " eps: " << vecPrimal[2*NumAttrib+i+1] << "\n" );
      }

      for (j=0; j<NumAttrib; ++j) {
      	for (i=0; i<NumObs; ++i)  {
      	  obs = data->vecTrainData[i];
      	  checkConst[j] += ( vecDual[i] - vecDual[NumObs+i] )
	    * data->standTrainData[obs].X[j] ;
      	}
      }

      DEBUGPR(1, cout << "vecPrimal: " << vecPrimal
	      << " PrimalObj:" << sumPrimal << "\n" );
      DEBUGPR(1, for (i=0; i<numCols; ++i) cout << vecPrimal[i] << " "; );
      DEBUGPR(1, cout << "\nvecDual: " << vecDual
	      << " DualObj:" << sumDual << "\n" );
      DEBUGPR(1, for (i=0; i<numRows; ++i) cout << vecDual[i] << " "; );
      DEBUGPR(1, cout << "\nsumCheck: " << sumCheck << "\n" );  // sum has to be 1
      DEBUGPR(1, cout << "checkCons: " << checkConst << "\n" );
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI
  }


} // namespace boosting



// Add variables to the model
//vars = model.addVars(LB, UB, NULL, vtype, NULL, numCols);
/*
// set constraits
for (i = 0; i < NumObs; ++i) {
lhs = vars[0];										   // beta_0
obs = data->vecTrainData[i];
for (j = 0; j < NumAttrib; ++j)      // beta^+_j
if (data->standData[obs].X[j] != 0)
lhs += data->standData[obs].X[j]*vars[1+j];
for (j = 0; j < NumAttrib; ++j)      // beta^-_j
if (data->standData[obs].X[j] != 0)
lhs -= data->standData[obs].X[j]*vars[1+NumAttrib+j];
for (j = 0; j < NumObs; ++j) 				// episilon
if (i==j)
lhs -= vars[1+2*NumAttrib+j];
// model.addConstr(lhs, GRB_LESS_EQUAL, data->standData[obs].y);
}

for (i = 0; i < NumObs; ++i) {
lhs = -vars[0];										   // beta_0
obs = data->vecTrainData[i];
for (j = 0; j < NumAttrib; ++j)      // beta^+_j
if (data->standData[obs].X[j] != 0)
lhs -= data->standData[obs].X[j]*vars[1+j];
for (j = 0; j < NumAttrib; ++j)      // beta^-_j
if (data->standData[obs].X[j] != 0)
lhs += data->standData[obs].X[j]*vars[1+NumAttrib+j];
for (j = 0; j < NumObs; ++j) 				// episilon
if (i==j)
lhs -= vars[1+2*NumAttrib+j];
// model.addConstr(lhs, GRB_LESS_EQUAL, -data->standData[obs].y);
}

// set cobjectives
obj = 0;

if (C != 0) {
for (j = 1; j < 2*NumAttrib+1; ++j)
obj += C*vars[j];
}
if (D != 0) {
for (j = 1; j < 2*NumAttrib+1; ++j)	// for linear square coefficients
obj += D*vars[j]*vars[j];
}
for (j = 2*NumAttrib+1; j < numCols; ++j) {
if (P==1)  		 obj += vars[j];
else if (P==2) obj += vars[j]*vars[j];
}
*/
/*
  model.setObjective(obj);
  model.update();
  //model.write("master.lp");
  model.getEnv().set(GRB_IntParam_OutputFlag, 0);  // not to print out GUROBI
*/

/*
// solve RMA
void REPR::solveRMA() { //const int& GreedyLevel) {

rma->reset();

#ifdef ACRO_HAVE_MPI
if (parallel) {
prma->reset();
//prma->printConfiguration();
//CommonIO::begin_tagging();
}
#endif //  ACRO_HAVE_MPI

rma->workingSol.value=-inf;
rma->mmapCachedCutPts.clear();
rma->numDistObs = NumObs;				// only use training data
rma->setSortObsNum(data->vecTrainData);	// only use training data
setDataWts();

if (data->getInitialGuess()) {
grma = new GreedyRMA(data);
grma->runGreedyRangeSearch();
}

rma->resetTimers();
InitializeTiming();
rma->solve();

} // end function REPR::solveRMA()
*/


/*
// train data using REPRoost
void REPR::train(const bool& isOuter, const int& NumIter, const int& greedyLevel) {

int i, j;
curIter=-1;

setREPRdata();
flagDuplicate=false;

try {

data->setStandData(data->origTrainData, data->standTrainData);					// standadize data for L1 regularization
setInitialMaster();
solveMaster();  //solveInitialMaster();

data->integerizeData(data->origTrainData, data->intTrainData); 	// integerize features
if (BaseRMA::exactRMA()) rma->setData(data);

for (curIter=0; curIter<NumIter; ++curIter) {

//ucout << "\nColGen Iter: " << curIter << "\n";
setDataWts();

solveRMA();


if ( !BaseRMA::exactRMA() || greedyLevel==base::Greedy) {	// Greedy RMA

grma = new greedyRMA::GreedyRMA(static_cast<BaseRMA *>(this), data);
grma->runGreedyRangeSearch();

if ( isOuter && grma->maxObjValue <= E + .00001 && BaseRMA::exactRMA()) {

solveRMA();	// solve RMA for each iteration

#ifdef ACRO_HAVE_MPI
if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
ucout << " RMA Solution:  " << rma->incumbentValue
<< "\tCPU time: " << rma->searchTime << "\n";
#ifdef ACRO_HAVE_MPI
}
#endif //  ACRO_HAVE_MPI
//
if ( rma->incumbentValue <= E + .00001 ) {
#ifdef ACRO_HAVE_MPI
if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
ucout << "Stopping Condition! RMA <=  E!!" << "\n";
#ifdef ACRO_HAVE_MPI
}
#endif //  ACRO_HAVE_MPI
//
rma->incumbentValue = inf;
break;
} else {
insertColumns(); // add RMA solutions
}

if (flagDuplicate) {
#ifdef ACRO_HAVE_MPI
if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
ucout << "Stop due to duplicates!" << "\n";
#ifdef ACRO_HAVE_MPI
}
#endif //  ACRO_HAVE_MPI
rma->incumbentValue = inf;
break;
}

// if the current rule is the same as the previous rule
} else if ( curIter>0 &&
grma->L == matIntLower[curIter-1] &&
grma->U == matIntUpper[curIter-1] )  {
#ifdef ACRO_HAVE_MPI
if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
ucout << "Stopping Condition! Greedy RMA" << "\n";
#ifdef ACRO_HAVE_MPI
}
#endif //  ACRO_HAVE_MPI
break;
} else {
insertGreedyColumns(); // add Greedy RMA solutions
}

} else if (BaseRMA::exactRMA()) {	// Exact RMA

#ifdef ACRO_HAVE_MPI
if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
if (isOuter) ucout << "Outer Iter: " << curIter+1 << " ";
else 				 ucout << "Inner Iter: " << curIter+1 << " ";
#ifdef ACRO_HAVE_MPI
}
#endif //  ACRO_HAVE_MPI

solveRMA();	// solve RMA for each iteration

if ( rma->incumbentValue <= E + .00001 )  {
#ifdef ACRO_HAVE_MPI
if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
ucout << "Stopping Condition! RMA <=  E" << "\n";
#ifdef ACRO_HAVE_MPI
}
#endif //  ACRO_HAVE_MPI
rma->incumbentValue = inf;
break;
}

insertColumns(); // add RMA solutions

if (flagDuplicate) {
#ifdef ACRO_HAVE_MPI
if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
ucout << "EREPR: Stop due to duplicates!" << "\n";
#ifdef ACRO_HAVE_MPI
}
#endif //  ACRO_HAVE_MPI
rma->incumbentValue = inf;
break;
}
} // end GreedyRMA or ExactRMA

// map back from the discretized data into original
setOriginalBounds();

#ifdef ACRO_HAVE_MPI
if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
if (isOuter) ucout << "Outer Iter: " << curIter+1;
else 				 ucout << "Inner Iter: " << curIter+1;
#ifdef ACRO_HAVE_MPI
}
#endif //  ACRO_HAVE_MPI

solveMaster();

} // end for each column generation iteration

#ifdef ACRO_HAVE_MPI
if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
if (isOuter) ucout << "OutREPR ";
else         ucout << "InnREPR ";
ucout << curIter << ":\tTest/Train Errors: " << errTest << " " << errTrain << "\n";
#ifdef ACRO_HAVE_MPI
}
#endif //  ACRO_HAVE_MPI

if ( evalFinalIter() && !(evalEachIter()) ) evaluateFinal();

// clean up GUROBI for the next crossvalidation set
//resetGurobi();

} catch(...) {
ucout << "Exception during optimization" << "\n";
return; // EXIT_FAILURE;
} // end try ... catch

} // trainData function
*/
