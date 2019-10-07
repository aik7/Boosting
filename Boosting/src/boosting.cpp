/*
 *  File name:   boosting.cpp
 *  Author:      Ai Kagawa
 *  Description: a source file for Boosting class
 */

#include "boosting.h"


namespace boosting {

///////////////////////// functions for LPB class /////////////////////////

// solve RMA
void Boosting::solveRMA() { //const int& GreedyLevel) {

#ifdef ACRO_HAVE_MPI
	if (parallel) {
		prma->reset();
		prma->printConfiguration();
		CommonIO::begin_tagging();
	} else {
#endif //  ACRO_HAVE_MPI
		rma->reset();
#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI

	rma->workingSol.value=-inf;
	rma->mmapCachedCutPts.clear();
	rma->numDistObs = NumObs;				// only use training data
	rma->setSortObsNum(data->vecTrainData);	// only use training data
	//setDataWts();

	rma->resetTimers();
	InitializeTiming();
	if (data->printBoost()) rma->solve();  // print out B&B details
	else                    rma->search();

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
	rma->printSolutionTime();
#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI

} // end function LPB::solveRMA()


// call GUROBI to solve Master Problems
void Boosting::solveMaster() {

  int i;

	DEBUGPRX(10, data, "Solve Restricted Master Problem!\n");

	tc.startTime();

	model.optimize();

	if (data->printBoost()) model.write("master.lp");

	vars = model.getVars();
	constr = model.getConstrs();

	vecPrimal.resize(numCols);
	vecDual.resize(numRows);

	if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
		primalVal = model.get(GRB_DoubleAttr_ObjVal);
		for (i = 0; i < numCols; ++i)
			vecPrimal[i] = vars[i].get(GRB_DoubleAttr_X);
		for (i = 0; i < numRows; ++i)
			vecDual[i] = constr[i].get(GRB_DoubleAttr_Pi);
	}

#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
	DEBUGPRX(0, data, " Master Solution: " << primalVal << "\t");
	printRMPSolution();
  tc.endCPUTime();
  DEBUGPRX(2, data, tc.endWallTime()); DEBUGPRX(2, data, "\n");
#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI

	if (data->evaluateEachIter()) {
		for (i=numBox-numRMASols; i<numBox; ++i) {
			setCoveredTrainObs();
			setCoveredTestObs();
		}
		//ucout << "Iter: " << curIter+1 << "\t";
		evaluateEach();
	}

} // end function Boosting::solveMaster()


// set original lower and upper bounds matrices
void Boosting::setOriginalBounds() {

  multimap<int, double>::iterator it;
  double lower, upper, tmpLower, tmpUpper;
  matOrigLower.resize(matIntLower.size());
  matOrigUpper.resize(matIntUpper.size());

  for (int k = matIntLower.size()-numRMASols; k<matIntLower.size(); ++k) {

    matOrigLower[k].resize(NumAttrib);
    matOrigUpper[k].resize(NumAttrib);

    for (int j=0; j<NumAttrib; ++j) { // for each attribute

      ///////////////////////////// mid point rule //////////////////////////////
      if ( matIntLower[k][j] > 0 ) { // lowerBound

        tmpLower =  getUpperBound(k, j, -1, false);
        tmpUpper =  getLowerBound(k, j, 0, false);
        lower =  (tmpLower + tmpUpper) / 2.0;

        DEBUGPRX(10, data, "(k,j): (" << k << ", " << j
          << ") matIntLower[k][j]-1: " << matIntLower[k][j]-1
          << " LeastLower: " << tmpLower << "\n"
          << " matIntLower[k][j]: " << matIntLower[k][j]
          << " GreatestLower: " << tmpUpper << "\n");

      } else lower=-inf; // if matIntLower[k][j] < 0 and matIntLower[k][j] != rma->distFeat[j]

      if ( matIntUpper[k][j] < data->distFeat[j] ) { // upperBound

        tmpLower = getUpperBound(k, j, 0, true);
        tmpUpper =  getLowerBound(k, j, 1, true);
        upper = (tmpLower + tmpUpper) / 2.0;

        DEBUGPRX(10, data, "(k,j): (" << k << ", " << j
          << ") matIntUpper[k][j]: " << matIntUpper[k][j]
          << " LeastUpper: " << tmpLower << "\n"
          << " matIntUpper[k][j]+1: " << matIntUpper[k][j]+1
          << " GreatestUpper: " << tmpUpper << "\n");

      } else upper=inf; // if matIntUpper[k][j] < rma->distFeat[j] and matIntUpper[k][j] != 0

      // store values
      matOrigLower[k][j]=lower;
      matOrigUpper[k][j]=upper;

    } // end for each attribute, j

  }

  DEBUGPRX(10, data, "Integerized Lower: \n" << matIntLower );
  DEBUGPRX(10, data, "Integerized Upper: \n" << matIntUpper );
  DEBUGPRX(10, data, "Real Lower: \n" << matOrigLower );
  DEBUGPRX(10, data, "Real Upper: \n" << matOrigUpper );

} // end function REPR::setOriginalBounds()


double Boosting::getLowerBound(int k, int j, int value, bool isUpper) {
  int boundVal; double min = inf;
  if (isUpper) boundVal = matIntUpper[k][j];
  else         boundVal = matIntLower[k][j];
  return data->vecFeature[j].vecIntMinMax[boundVal+value].minOrigVal;
}


double Boosting::getUpperBound(int k, int j, int value, bool isUpper)  {
  int boundVal; double max = -inf;
  if (isUpper) boundVal = matIntUpper[k][j];
  else         boundVal = matIntLower[k][j];
  return data->vecFeature[j].vecIntMinMax[boundVal+value].maxOrigVal;
}


// reset Gurobi for the next column generation iteration
void Boosting::resetGurobi() {

  int sizeCol = vecPrimal.size();
  int sizeRow = data->isLPBoost() ? NumObs+1 : 2*NumObs ;

  for (int i=0; i<sizeRow; ++i)
    model.remove(model.getConstrs()[i]);
  //ucout << "Num var: " << sizeCol << endl;

  for (int j=0; j<sizeCol; ++j)
    model.remove(model.getVars()[j]);

  model.reset();
  model.update();

}


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

/*
bool Boosting::isDuplicate() {

#ifdef ACRO_HAVE_MPI
  if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
  if ( matIntUpper.size()==1 ) return false;

  for (int j=0; j<NumAttrib; ++j)
    if ( matIntLower[matIntLower.size()-2][j] != grma->L[j]
      || matIntUpper[matIntUpper.size()-2][j] != grma->U[j] )
        return false;
  ucout << "Duplicate Solution Found!! \n" ;
  flagDuplicate = true;
#ifdef ACRO_HAVE_MPI
  }
#endif //  ACRO_HAVE_MPI
  return true;

}
*/

void Boosting::evaluateEach() {

  errTrain = evaluateEachIter(TRAIN);
  errTest = evaluateEachIter(TEST);
	if (data->isLPBoost() && data->printBoost()) printEachIterAllErrs();

	#ifdef ACRO_HAVE_MPI
		if (uMPI::rank==0) {
	#endif //  ACRO_HAVE_MPI
		(isOuter) ? ucout << "Outer " : ucout << "Inner ";
		ucout << "Iter: " << curIter+1 << " ";
		ucout << "Test/Train Errors: " << errTest << " " << errTrain ;
		if (data->isLPBoost()) ucout << " " << "rho: "<< vecPrimal[NumObs] << "\n";
		else           ucout << "\n";
	#ifdef ACRO_HAVE_MPI
		}
	#endif //  ACRO_HAVE_MPI

}


void Boosting::evaluateFinal() {

  errTrain = evaluateAtFinal(TRAIN);
  errTest = evaluateAtFinal(TEST);

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
	if (data->printBoost()) ucout << "vecCoveredObsByBox Train:\n";
	for (int i=0; i<data->vecTrainData.size(); ++i) {
		obs = data->vecTrainData[i];
		vecCoveredObsByBox[obs].resize(numBox);
		vecCoveredObsByBox[obs][numBox-1] = vecIsCovered[i];
		if (data->printBoost()) {
			for (int j=0; j<numBox; ++j)
				ucout << vecCoveredObsByBox[obs][j] << " ";
	    ucout << "\n";
		}
	}

}


// set covered test observations
void Boosting::setCoveredTestObs() {

	int obs;
	if (data->printBoost()) ucout << "\nvecCoveredObsByBox Test:\n";
	for (int i=0; i<data->vecTestData.size(); ++i) { // for each test dataset
		obs = data->vecTestData[i];
		vecCoveredObsByBox[obs].resize(numBox);

		for (int j=0; j<NumAttrib; ++j) { // for each attribute

			if (matOrigLower[numBox-1][j] <= data->origData[obs].X[j] &&
					data->origData[obs].X[j] <= matOrigUpper[numBox-1][j]  ) {
				if ( j==NumAttrib-1) { // all features are covered by the box
					vecCoveredObsByBox[obs][numBox-1] = true;
				}
			} else {
				vecCoveredObsByBox[obs][numBox-1] = false;
				break; // this observation is not covered
			}

		} // end for each attribute, j

		if (data->printBoost()) {
			for (int j=0; j<numBox; ++j)
				ucout << vecCoveredObsByBox[obs][j] << " ";
	    ucout << "\n";
		}

	} // for each test dataset

} // setCoveredTestObs function


void Boosting::writePredictions(const int& isTest) {

  stringstream s;
  int obs, size;

  if (isTest) s << "predictionTest" << '.' << data->problemName;
  else        s << "predictionTrain" << '.' << data->problemName;

  ofstream os(s.str().c_str());
  // appending to its existing contents
  //ofstream os(s.str().c_str(), ofstream::app);

  os << "ActY \t Boosting  \n";

  (isTest) ? obs = data->vecTestData.size() : obs = data->vecTrainData.size();
  for (int i=0; i < size; ++i) {
    (isTest) ? obs = data->vecTestData[i] : obs = data->vecTrainData[i];
    if (isTest) os << data->origData[obs].y << " " << predTest[i] << "\n";
		else        os << data->origData[obs].y << " " << predTrain[i] << "\n";
  }

  os.close();
}


void Boosting::checkObjValue() {
	int obs;
	double wt=0.0;

	for (int i=0; i<NumObs; i++) { // for each training data
  	obs = data->vecTrainData[i];
    for (int f=0; f<NumAttrib; ++f) { // for each feature
    	if ( (grma->L[f] <= data->intData[obs].X[f])
    			&& (data->intData[obs].X[f] <= grma->U[f]) ) {
        if (f==NumAttrib-1)  // if this observation is covered by this solution
          wt+=data->intData[obs].w;   //dataWts[i];
      } else break; // else go to the next observation
  	}  // end for each feature
  }  // end for each training observation

  cout << "grma->L: " << grma->L ;
  cout << "grma->U: " << grma->U ;
	cout << "GRMA ObjValue=" << wt << "\n";
}


void Boosting::checkObjValue(int k) {
	int obs;
	double wt=0.0;

	for (int i=0; i<NumObs; i++) { // for each training data
  	obs = data->vecTrainData[i];
    for (int f=0; f<NumAttrib; ++f) { // for each feature
    	if ( (matIntLower[k][f] <= data->intData[obs].X[f])
    			&& (data->intData[obs].X[f] <= matIntUpper[k][f]) ) {
        if (f==NumAttrib-1)  // if this observation is covered by this solution
          wt+=data->intData[obs].w;   //dataWts[i];
      } else break; // else go to the next observation
  	}  // end for each feature
  }  // end for each training observation

  cout << "matIntLower[k]: " << matIntLower[k] ;
  cout << "matIntUpper[k]: " << matIntUpper[k] ;
	cout << "RMA ObjValue=" << wt << "\n";
}


} // namespace boosting
