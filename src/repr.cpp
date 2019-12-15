/*
 *  File name:   repr.cpp
 *  Author:      Ai Kagawa
 *  Description: a source file for REPR class
 */

#include "repr.h"

namespace boosting {


///////////////////////// functions for LPB class /////////////////////////

/*
REPR::REPR(int argc, char** argv) { // }, Data* d) {

  data = d;

  NumIter = data->getIterations();

	P = data->getExponentP();

  if (data->exactRMA()) {

#ifdef ACRO_HAVE_MPI
    //uMPI::init(&argc, &argv, MPI_COMM_WORLD);
    int nprocessors = uMPI::size;
    /// Do parallel optimization if MPI indicates that we're using more than one processor
    if (parallel_exec_test<parallelBranching>(argc,argv,nprocessors)) {
      /// Manage parallel I/O explicitly with the utilib::CommonIO tools
      CommonIO::begin();
      CommonIO::setIOFlush(1);
      parallel = true;
      prma = new parRMA;
      rma  = prma;
      prma->setParameter(data, data->debug);
    } else {
#endif // ACRO_HAVE_MPI
      rma = new RMA;
#ifdef ACRO_HAVE_MPI
    }
#endif // ACRO_HAVE_MPI
    rma->setParameters(data, data->debug);

  } // end if exactRMA is enabled

}
*/

void REPR::initBoostingData() {

  numBox     = 0;
	numRMASols = 0;
	NumObs     = data->numTrainObs;
	NumAttrib  = data->numAttrib;

  if (!innerCV()) {
  	C 				= getCoefficientC();
  	D 				= getCoefficientD();
  	E 				= getCoefficientE();
  	F 				= getCoefficientF();
  }
  D = 0;
	F = 0;

	vecDual.resize(NumObs);
	vecIsCovered.resize(NumObs);
	if (BaseRMA::exactRMA())	rma->incumbentValue = inf;

	matIntLower.clear();
	matIntUpper.clear();

  matOrigLower.clear();
  matOrigUpper.clear();

	if (evalEachIter()) {
		vecCoveredObsByBox.clear();
		vecCoveredObsByBox.resize(data->numOrigObs);
	}

}


// train data using REPRoost
void REPR::trainData(const bool& isOuter, const int& NumIter, const int& greedyLevel) {

	int i, j;
  curIter=-1;

  initBoostingData();
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

			if ( !BaseRMA::exactRMA() || greedyLevel==Greedy) {	// Greedy RMA

				grma = new GreedyRMA(static_cast<BaseRMA *>(this), data);
        grma->runGreedyRangeSearch();

        if ( isOuter && grma->maxObjValue <= E + .00001 && BaseRMA::exactRMA()) {

          solveRMA();	// solve RMA for each iteration
/*
          #ifdef ACRO_HAVE_MPI
            if (uMPI::rank==0) {
          #endif //  ACRO_HAVE_MPI
            ucout << " RMA Solution:  " << rma->incumbentValue
              << "\tCPU time: " << rma->searchTime << "\n";
          #ifdef ACRO_HAVE_MPI
           }
          #endif //  ACRO_HAVE_MPI
*/
          if ( rma->incumbentValue <= E + .00001 ) {
            #ifdef ACRO_HAVE_MPI
              if (uMPI::rank==0) {
            #endif //  ACRO_HAVE_MPI
              ucout << "Stopping Condition! RMA <=  E!!" << "\n";
            #ifdef ACRO_HAVE_MPI
              }
            #endif //  ACRO_HAVE_MPI

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

		if ( evalFinalIter() && !(evalEachIter()) )
			evaluateFinal();

		// clean up GUROBI for the next crossvalidation set
		resetGurobi();
/*
	} catch(GRBException e) {
		ucout << "Error during GUROBI " << e.getErrorCode() << "\n";
		ucout << e.getMessage() << "\n";
		return; // EXIT_FAILURE;
*/
  } catch(...) {
		ucout << "Exception during optimization" << "\n";
		return; // EXIT_FAILURE;
  } // end try ... catch

} // trainData function


// set up for the initial master problem
void REPR::setInitialMaster() {

  int i, j, obs;
	numRows = 2*NumObs;
	numCols = 1+2*NumAttrib+NumObs; //NumVar+1;	// +1 for constant term
	//double* LB = new double[numCols];
	//double* UB = new double[numCols];
	//char* vtype = NULL;

	DEBUGPR(10, cout << "Setup Initial Restricted Master Problem!" << "\n");

  int sizeCol = vecPrimal.size();
  int sizeRow = isLPBoost() ? NumObs+1 : 2*NumObs ;

  objective   = new double[sizeCol];
  lowerColumn = new double[sizeCol];
  upperColumn = new double[sizeCol];
  lowerRow    = new double[sizeRow];
  upperRow    = new double[sizeRow];
  dataWts     = new double[NumObs];

  columnIndex = new int[NumRow];
  for (i=0; i<NumRow; ++i) columnIndex[i] = i;
  //element     = new double [2*sizeCol];
  //start       = new CoinBigIndex[sizeCol+1];
  //row         = new int[sizeRow];
  //start[sizeCol] = 2 * sizeCol;

  model.setOptimizationDirection(1);               // maximization
  model.setLogLevel(0); // to turn off some output, 0 gives nothing and each increase in value switches on more messages.
  matrix.setDimensions(sizeRow, sizeCol); // setDimensions (int numrows, int numcols)

	// model.getEnv().set(GRB_IntParam_Method, 0);

	for (i=0; i<numCols; ++i) {
    lowerColumn[i] = 0;
    upperColumn[i] = inf;
    objective[i]   = 1.0;
    row.insert(i, 1.0); // insert( int index, double element )
  }

  lowerColumn[0] = -inf; // beta_0 is free variable

  // set constraits
	for (i = 0; i < NumObs; ++i) {
    row.insert(0, 1.0);
		obs = data->vecTrainData[i];
		for (j = 0; j < NumAttrib; ++j)      // beta^+_j
      row.insert(1+j, data->standData[obs].X[j]);
		for (j = 0; j < NumAttrib; ++j)      // beta^-_j
      row.insert(1+NumAttrib+j, -data->standData[obs].X[j);
		for (j = 0; j < NumObs; ++j) 				// episilon
			if (i==j)
				row.insert(1+2*NumAttrib+j, 1.0);

    matrix.appendRow(row);
    lowerRow[i] = -inf;
    upperRow[i] = data->standData[obs].y;
	}

	for (i = 0; i < NumObs; ++i) {
		row.insert(0, -1.0);									   // beta_0
		obs = data->vecTrainData[i];
		for (j = 0; j < NumAttrib; ++j)      // beta^+_j
			row.insert(1+j, -data->standData[obs].X[j]);
		for (j = 0; j < NumAttrib; ++j)      // beta^-_j
      row.insert(1+NumAttrib+j, data->standData[obs].X[j);
		for (j = 0; j < NumObs; ++j) 				// episilon
			if (i==j)
        row.insert(1+2*NumAttrib+j, -1.0);

    matrix.appendRow(row);
    lowerRow[NumObs+i] = -inf;
    upperRow[NumObs+i] = -data->standData[obs].y;
	}

  matrix.loadProblem(matrix, lowerColumn, upperColumn, objective,
                     lowerRow, upperRow);

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

}  // end function REPR::setInitialMaster()


void REPR::setDataWts() {

	int obs;

  DEBUGPR(1, cout << "wt: ");
	for (int i=0; i < NumObs ; ++i) {
		obs = data->vecTrainData[i];
		data->intTrainData[obs].w = (vecDual[i]-vecDual[NumObs+i]);
    DEBUGPR(1, cout << data->intTrainData[obs].w << ", ");
	}
  DEBUGPR(1, cout << "\n");

}


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


// insert columns in each column iteration
void REPR::insertColumns() { //const int& GreedyLevel) {

	int obs;
	numRMASols=0;

	///////////////////////////// constraints /////////////////////////////
	rma->getAllSolutions(s);
	sl.resize(s.size());

	for (int k=0; k<s.size(); ++k)
		sl[k] = dynamic_cast<rmaSolution*>(s[k]);

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

    // add columns using GUROBI/CLP
    //col.clear();
    //constr = model.getConstrs();

    double *columnValue = new double[NumRow];
    for (int i = 0; i < NumObs; ++i) {
      obs = data->vecTrainData[i];
      if (vecIsCovered[i]==true) {
        if (sl[k]->isPosIncumb) {
          columnValue[i]        = 1;
          columnValue[NumObs+i] = -1;
          //col.addTerm(1, constr[i]);
          //col.addTerm(-1, constr[i+NumObs]);
        } else {
          columnValue[i]        = -1;
          columnValue[NumObs+i] = 1;
          //col.addTerm(-1, constr[i]);
          //col.addTerm(1, constr[i+NumObs]);
        }
      }
    }
    model.addColumn(1, columnIndex, columnValue, 0.0, COIN_DBL_MAX, E);
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
	numRMASols=1;

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
	//model.addVar(0.0, GRB_INFINITY, E, GRB_CONTINUOUS, col);

	++numBox;
	++numCols;

}


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
		for (j = 1; j < NumAttrib+1; ++j)
			sumPrimal += C*vecPrimal[j];
	}
	for (j = NumAttrib+1; j < numCols; ++j) {
		if (P==1)
			sumPrimal += vecPrimal[j];
		else if (P==2)
			sumPrimal += vecPrimal[j]*vecPrimal[j];
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

		DEBUGPR(10, cout << "mu: " << vecDual[i] << " nu:" << vecDual[NumObs+i]
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
	DEBUGPR(1, cout << "vecDual: " << vecDual
								 << " DualObj:" << sumDual << "\n" );
	DEBUGPR(1, cout << "sumCheck: " << sumCheck << "\n" );  // sum has to be 1
	DEBUGPR(1, cout << "checkCons: " << checkConst << "\n" );
#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI
}

////////////////////// Evaluating methods ///////////////////////////////////////

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


} // namespace boosting
