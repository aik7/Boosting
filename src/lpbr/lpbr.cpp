/*
 *  File name:   lpbr.cpp
 *  Author:      Ai Kagawa
 *  Description: a source file for LPBR class
 */

#include "lpbr.h"


namespace boosting {

  /////////////////////// functions for LPBR class ///////////////////////
  
  LPBR::LPBR(int argc, char** argv, Data* d) {
    
    data = d;
    
    NumIter = data->getIterations();
    D = data->getCoefficientD() ;
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
	prma->setParameters(data, data->debug);
      } else {
#endif // ACRO_HAVE_MPI
	rma = new RMA;
#ifdef ACRO_HAVE_MPI
      }
#endif // ACRO_HAVE_MPI

    } // end if exactRMA is enabled

  }


  void LPBR::initBoostingData() {

    numBox = 0;
    numRMASols=0;
    NumObs = data->numTrainObs;
    NumAttrib = data->numAttrib;

    if (!data->innerCV())
      if (D==.5)
	D = 1 / ( data->numTrainObs * data->getNu() ) ;
    ucout << " Parameter D: " << D << "\n";

    vecDual.resize(NumObs+1);
    vecIsCovered.resize(NumObs);
    if (data->exactRMA())	rma->incumbentValue = inf;

    matIntLower.clear();
    matIntUpper.clear();

    matOrigLower.clear();
    matOrigUpper.clear();

    if (data->evaluateEachIter()) {
      vecCoveredObsByBox.clear();
      vecCoveredObsByBox.resize(data->numOrigObs);
    }
  }


  // train data using LPBR
  void LPBR::trainData(const bool& isOuter, const int& NumIter, const int& greedyLevel) {

    int i, j;
    int numVar;
    curIter=-1;
    alpha=0;

    initBoostingData();
    flagDuplicate=false;

    try {

      data->setPosNegObs();
      data->integerizeData(); 	// integerize features

      setInitialMaster();
      if (data->initRules() || data->init1DRules()) {
	solveMaster();
	alpha = vecDual[NumObs];
      }
      if (data->exactRMA()) rma->setData(data);

      numVar=NumObs+2;

      // for each column generation iteration
      for (curIter=0; curIter<NumIter; ++curIter) {

	if ( data->greedyRMA() || greedyLevel==Greedy) {	// Greedy RMA

	  setDataWts();
	  grma = new GreedyRMA(data);
	  grma->runGreedyRangeSearch();

	  ucout << "alpha: " << alpha << " GRMA: " << grma->maxObjValue <<"\n";

	  if ( abs(grma->maxObjValue) < -alpha +.000001 && data->exactRMA()) {

	    solveRMA();	// solve RMA for each iteration
	    insertColumns(); // add RMA solutions

	    ucout << "alpha: " << alpha << " RMA:  " << rma->incumbentValue <<"\n";

	    if ( rma->incumbentValue < -alpha +.000001 )  {
	      ucout << "Stopping Condition! RMA <=  -alpha" << "\n";
	      rma->incumbentValue = inf;
	      break;
	    }

	    if (flagDuplicate) {
	      ucout << "Stop due to duplicates!" << "\n";
	      rma->incumbentValue = inf;
	      break;
	    }

	    // if the current rule is the same as the previous rule
	  } else if ( curIter>0 &&
		      grma->L == matIntLower[curIter-1] &&
		      grma->U == matIntUpper[curIter-1] )  {
	    ucout << "Stopping Condition! Greedy RMA" << "\n";

	    if (data->exactRMA()) {

	      solveRMA();	// solve RMA for each iteration
	      insertColumns(); // add RMA solutions

	      ucout << "alpha: " << alpha << " RMA:  " << rma->incumbentValue <<"\n";

	      if ( rma->incumbentValue <= -alpha +.00001 )  {
		ucout << "Stopping Condition! RMA <=  -alpha" << "\n";
		rma->incumbentValue = inf;
		break;
	      }

	      if (flagDuplicate) {
		ucout << "Stop due to duplicates!" << "\n";
		rma->incumbentValue = inf;
		break;
	      }

	    } else break;

	  } else {
	    insertGreedyColumns(); // add Greedy RMA solutions
	  }

	} else if (data->exactRMA()) {	// Exact RMA

	  solveRMA();	// solve RMA for each iteration

	  ucout << "alpha: " << alpha << " RMA: " << rma->incumbentValue <<"\n";

	  if ( rma->incumbentValue < -alpha +.000001 )  {
	    ucout << "Stopping Condition! RMA <=  -alpha" << "\n";
	    rma->incumbentValue = inf;
	    break;
	  }

	  insertColumns(); // add RMA solutions

	  if (flagDuplicate) {
	    ucout << "Stop due to duplicates!" << "\n";
	    rma->incumbentValue = inf;
	    break;
	  }

	}	// end GreedyRMA or ExactRMA

	if (data->printBoost()) {

	  if (data->initRules() || data->init1DRules()) {
	    ucout << "beta: ";
	    for (int j=0; j<NumAttrib; ++j) ucout << vecPrimal[NumObs+2+j] << " ";
	    ucout << "\n";
	    numVar += NumAttrib;
	  }

	  if (curIter!=0) {
	    ucout << "gamma: ";
	    for (int k=0; k<matIntLower.size(); ++k)
	      ucout << vecPrimal[numVar+k] << " ";
	    ucout << "\n";
	  }

	  if (data->initGuess()) checkObjValue();
	  checkObjValue(numBox-1);

	}

	// map back from the discretized data into original
	setOriginalBounds();

	if (isOuter) ucout << "Outer Iter: " << curIter+1;
	else 				 ucout << "Inner Iter: " << curIter+1;

	solveMaster();
	alpha = vecDual[NumObs];

      } // end for each column generation iteration

      if (isOuter) ucout << "Outer Final Iter: ";
      else         ucout << "Inner Final Iter: ";
      ucout << curIter << " Test/Train Errors: " << errTest << " " << errTrain
	    << " " << "rho: "<< vecPrimal[NumObs] << "\n";

      if ( data->evaluateFinalIter() && !(data->evaluateEachIter()) )
	evaluateFinal();

      // clean up GUROBI for the next crossvalidation set
      resetGurobi();

    } catch(GRBException e) {
      ucout << "Error during GUROBI " << e.getErrorCode() << "\n";
      ucout << e.getMessage() << "\n";
      return; // EXIT_FAILURE;
    } catch(...) {
      ucout << "Exception during optimization" << "\n";
      return; // EXIT_FAILURE;
    } // end try ... catch

  } // trainData function


  // set up for the initial master problem
  void LPBR::setInitialMaster() {

    //* LPBR soft margin
    int i, j, obs;
    numRows = NumObs+1; // m inequality and 1 linear combination combination
    numCols = NumObs+1+1; // episilons and rho
    if (data->initRules()) {
      setSimpleRules();
      numCols+=NumAttrib;
    }
    if (data->init1DRules()) {
      set1DRules();
      numCols+=NumAttrib;
    }
    double* LB = new double[numCols];
    double* UB = new double[numCols];
    char* vtype = NULL;

    DEBUGPRX(10, data, "Setup Initial Master Problem!" << "\n");

    for ( i = 0; i < NumObs; ++i) { // \episilon >=0 0
      LB[i] = 0;
      UB[i] = inf;
    }

    // rho
    LB[NumObs] = data->getLowerRho(); // rho is not constrained by default
    UB[NumObs] = data->getUpperRho();

    // gamma_0
    LB[NumObs+1] = -inf;
    UB[NumObs+1] = inf;

    if (data->initRules() || data->init1DRules())
      for ( j=0; j<NumAttrib; ++j) { // for simple rules
	LB[NumObs+2+j] = 0;
	UB[NumObs+2+j] = inf;
      }

    // Add variables to the model
    vars = model.addVars(LB, UB, NULL, vtype, NULL, numCols);

    // Add constraints to the model
    for (i=0; i<NumObs; ++i) {

      lhs = 0;

      for (j=0; j<NumObs; ++j)
	if (i==j) lhs +=  vars[j];	 // + episilon_i

      lhs -= vars[NumObs]; // - rho

      obs = data->vecTrainData[i];

      lhs += data->origData[obs].y * vars[NumObs+1]; // gamma_0

      if (data->initRules())
	for (j=0; j<NumAttrib; ++j)
	  if (vecCoveredObsBySimpleRule[obs][j])
	    lhs += data->origData[obs].y * vars[NumObs+2+j];

      if (data->init1DRules())
	for (j=0; j<NumAttrib; ++j)
	  if (vecCoveredObsBySimpleRule[obs][j])
	    if (vecSMIsPositive[j])	lhs += data->origData[obs].y * vars[NumObs+2+j];
	    else                    lhs -= data->origData[obs].y * vars[NumObs+2+j];

      model.addConstr(lhs, GRB_GREATER_EQUAL, 0); // inequality constraint
    }

    lhs = 0; //vars[NumObs+1]; // gamma_0
    if (data->initRules() || data->init1DRules())
      for (j=0; j<NumAttrib; ++j)
	lhs += vars[NumObs+2+j];
    model.addConstr(lhs, GRB_EQUAL, 1); // equality constraint

    // set objective function
    obj = 0;
    for (i = 0; i < NumObs; ++i)
      if (P==1)  		 obj += D*vars[i];
      else if (P==2) obj += D*vars[i]*vars[i];

    obj -= vars[NumObs];  // -rho

    model.setObjective(obj);
    model.update();
    if (data->printBoost()) model.write("master.lp");
    model.getEnv().set(GRB_IntParam_OutputFlag, 0);  // not to print out GUROBI

  }  // end function LPB::setInitialMaster()


  void LPBR::setDataWts() {

    int obs;

    for (int i=0; i < NumObs ; ++i) {
      obs = data->vecTrainData[i];
      if (numBox==0 && !data->initRules() && !data->init1DRules() )
	data->intData[obs].w = (1.0/NumObs) * data->origData[obs].y;
      else
	data->intData[obs].w = vecDual[i] * data->origData[obs].y;
    }

    if (data->printBoost()) {
      ucout << "y: ";
      for (int i=0; i<NumObs; ++i)
	ucout << data->origData[data->vecTrainData[i]].y << "\t";
      ucout << "\nw: ";
      for (int i=0; i<NumObs; ++i)
	ucout << data->intData[data->vecTrainData[i]].w << "\t";
      ucout << "\n";
    }

  }


  // insert columns in each column iteration
  void LPBR::insertColumns() { //const int& GreedyLevel) {

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
	  ucout << "insertColumns::Duplicated!\n";
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
	  if ( sl[k]->a[j] <=  data->intData[obs].X[j] &&
	       data->intData[obs].X[j] <= sl[k]->b[j] ) {
	    if ( j==NumAttrib-1)
	      vecIsCovered[i]= true;
	  } else {
	    vecIsCovered[i]= false;
	    break;
	  }
	} // end for each attribute, j
      } // end for each observation, i

      if (sl[k]->isPosIncumb) {
	DEBUGPRX(1, data, "Positive Box\n");
	vecCoveredSign[numBox+numRMASols] = true;
      } else {
	DEBUGPRX(1, data, "Negative Box\n");
	vecCoveredSign[numBox+numRMASols] = false;
      }

      matIntLower[matIntLower.size()-s.size()+k] = sl[k]->a;
      matIntUpper[matIntUpper.size()-s.size()+k] = sl[k]->b;
      ++numRMASols;  // number of RMA solutions

      DEBUGPRX(1, data, "vecIsCovered: " << vecIsCovered << "\n" );

      // add columns using GUROBI
      col.clear();
      constr = model.getConstrs();
      for (int i = 0; i < NumObs; ++i) {
	obs = data->vecTrainData[i];
	if (vecIsCovered[i]==true)
	  if (sl[k]->isPosIncumb)
	    col.addTerm(data->origData[obs].y, constr[i]);
	  else
	    col.addTerm(-data->origData[obs].y, constr[i]);
      }
      col.addTerm(1, constr[NumObs]);
      model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, col);

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
  void LPBR::insertGreedyColumns() { //const int& GreedyLevel) {

    int obs;
    numRMASols=1;

    ///////////////////////////// constraints /////////////////////////////
    vecCoveredSign.resize(vecCoveredSign.size()+1);
    matIntUpper.resize(matIntUpper.size()+1);
    matIntLower.resize(matIntLower.size()+1);

    for (int i=0; i< NumObs; ++i) { // for each observation
      obs = data->vecTrainData[i];
      for (int j=0; j< NumAttrib; ++j) { // for each attribute
	if ( grma->L[j] <=  data->intData[obs].X[j] &&
	     data->intData[obs].X[j] <= grma->U[j] ) {
	  if ( j==NumAttrib-1)
	    vecIsCovered[i]= true;
	} else {
	  vecIsCovered[i]= false;
	  break;
	}
      } // end for each attribute, j
    } // end for each observation, i

    if (grma->isPosIncumb) {
      DEBUGPRX(1, data, "Positive Box\n");
      vecCoveredSign[numBox] = true;
    } else {
      DEBUGPRX(1, data, "Negative Box\n");
      vecCoveredSign[numBox] = false;
    }

    matIntLower[matIntLower.size()-1] = grma->L;
    matIntUpper[matIntUpper.size()-1] = grma->U;

    DEBUGPRX(1, data, "vecIsCovered: " << vecIsCovered );

    // add columns using GUROBI
    col.clear();
    constr = model.getConstrs();
    for (int i = 0; i < NumObs; ++i) {
      obs = data->vecTrainData[i];
      if (vecIsCovered[i]==true)
	if (grma->isPosIncumb)
	  col.addTerm(data->origData[obs].y, constr[i]);
	else
	  col.addTerm(-data->origData[obs].y, constr[i]);
    }
    col.addTerm(1, constr[NumObs]);
    model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, col);

    ++numBox;
    ++numCols;

  }


  void LPBR::set1DRules() {

    int obs, k;

    vec1DRuleLower.resize(NumAttrib);
    vec1DRuleUpper.resize(NumAttrib);

    vecSMIsPositive.clear();
    vecSMIsPositive.resize(NumAttrib);

    vecCoveredObsBySimpleRule.clear();
    vecCoveredObsBySimpleRule.resize(data->numOrigObs);
    for (int i=0; i<data->numOrigObs; ++i)
      vecCoveredObsBySimpleRule[i].resize(NumAttrib);

    grma = new GreedyRMA(data);
    grma->setInit1DRules();

    for (int j=0; j<NumAttrib; ++j) {
      grma->set1DOptRange(j);
      vecSMIsPositive[j] = grma->isPosIncumb;
      vec1DRuleLower[j] = grma->optLower;
      vec1DRuleUpper[j] = grma->optUpper;

      for (int i=0; i<NumObs; ++i) {
	obs = data->vecTrainData[i];
	if ( vec1DRuleLower[j] <= data->intData[obs].X[j]
	     && data->intData[obs].X[j] <= vec1DRuleUpper[j] )
	  vecCoveredObsBySimpleRule[obs][j] = true;
	else
	  vecCoveredObsBySimpleRule[obs][j] = false;
      }

      setOriginal1DRule(j, grma->optLower, grma->optUpper);

      // set covered test observations by simple rules
      for (int i=0; i<data->vecTestData.size(); ++i) { // for each test dataset
	obs = data->vecTestData[i];
	if (vec1DRuleLower[j] <= data->origData[obs].X[j]
	    && data->origData[obs].X[j] <= vec1DRuleUpper[j])
	  vecCoveredObsBySimpleRule[obs][j] = true;
	else
	  vecCoveredObsBySimpleRule[obs][j] = false;
      } // for each test dataset

    }

    if (data->printBoost()) {
      ucout << "vecSMIsPositive: " << vecSMIsPositive << "\n";
      ucout << "vecCoveredObsBySimpleRule\n";
      for (int i=0; i<data->numOrigObs; ++i) { // for each test dataset
	for (int j=0; j<NumAttrib; ++j)
	  ucout << vecCoveredObsBySimpleRule[i][j] << " ";
	ucout << "\n";
      } // for each test dataset
      ucout << "\n";
    }

  } // end set1DRules


  void LPBR::setOriginal1DRule(const int& j, const int& l, const int& u) {

    double tmpLower, tmpUpper, lower, upper;

    if ( l > 0 ) { // lowerBound

      tmpLower =  getUpperBound1D(j, -1, false);
      tmpUpper =  getLowerBound1D(j, 0, false);
      lower =  (tmpLower + tmpUpper) / 2.0;

      DEBUGPRX(10, data, "j: " << j
	       << " matIntLower[j]-1: " << l -1
	       << " LeastLower: " << tmpLower << "\n"
	       << " matIntLower[j]: " << u
	       << " GreatestLower: " << tmpUpper << "\n");

    } else lower=-inf; // if matIntLower[k][j] < 0 and matIntLower[k][j] != rma->distFeat[j]

    if ( u < data->distFeat[j] ) { // upperBound

      tmpLower = getUpperBound1D(j, 0, true);
      tmpUpper =  getLowerBound1D(j, 1, true);
      upper = (tmpLower + tmpUpper) / 2.0;

      DEBUGPRX(10, data, "j: " << j
	       << " matIntUpper[j]: " << u
	       << " LeastUpper: " << tmpLower << "\n"
	       << " matIntUpper[j]+1: " << u+1
	       << " GreatestUpper: " << tmpUpper << "\n");

    } else upper=inf; // if matIntUpper[k][j] < rma->distFeat[j] and matIntUpper[k][j] != 0

    // store values
    vec1DRuleLower[j]=lower;
    vec1DRuleUpper[j]=upper;

    DEBUGPRX(10, data, "j: " << j << " orig lower: " << lower
	     << " orig upper: " << upper << "\n");

  }


  double LPBR::getLowerBound1D(int j, int value, bool isUpper) {
    int boundVal; double min = inf;
    if (isUpper) boundVal = vec1DRuleUpper[j];
    else         boundVal = vec1DRuleLower[j];
    return data->vecFeature[j].vecIntMinMax[boundVal+value].minOrigVal;
  }


  double LPBR::getUpperBound1D(int j, int value, bool isUpper)  {
    int boundVal; double max = -inf;
    if (isUpper) boundVal = vec1DRuleUpper[j];
    else         boundVal = vec1DRuleLower[j];
    return data->vecFeature[j].vecIntMinMax[boundVal+value].maxOrigVal;
  }


  void LPBR::setSimpleRules() {

    bool isPrevPos;
    int obs, k;
    vector<int> numObsEachVal(data->maxL);
    vector<int> vecSumEachVal(data->maxL);
    vecSimpleRule.clear();
    vecCoveredObsBySimpleRule.clear();
    vecSimpleRule.resize(NumAttrib);
    vecCoveredObsBySimpleRule.resize(data->numOrigObs);
    for (int i=0; i<data->numOrigObs; ++i)
      vecCoveredObsBySimpleRule[i].resize(NumAttrib);

    for (int j=0; j<NumAttrib; ++j) {

      for (int l=0; l<data->distFeat[j]+1; ++l) {
	numObsEachVal[l] = 0;
	vecSumEachVal[l] = 0;
      }

      for (int i=0; i<NumObs; ++i) {
	obs = data->vecTrainData[i];
	++numObsEachVal[data->intData[obs].X[j]];
	vecSumEachVal[data->intData[obs].X[j]] += (int) data->origData[obs].y;
      }

      DEBUGPRX(5, data, "j: " << j << " numObsEachVal: " << numObsEachVal);
      DEBUGPRX(5, data, "j: " << j << " b/f vecSumEachVal: " << vecSumEachVal);

      for (int l=0; l<data->distFeat[j]+1; ++l) {
	vecSumEachVal[l] += (int) (data->numNegTrainObs-data->numPosTrainObs)
	  * numObsEachVal[l] / (double) data->numTrainObs ;
	if (vecSumEachVal[l]==0)
	  if (data->numPosTrainObs > data->numNegTrainObs) vecSumEachVal[l]=1;
      }

      DEBUGPRX(5, data, "j: " << j << " a/f vecSumEachVal: " << vecSumEachVal);

      for (int i=0; i<NumObs; ++i) {
	obs = data->vecTrainData[i];
	if ( vecSumEachVal[data->intData[obs].X[j]]>0 )
	  vecCoveredObsBySimpleRule[obs][j] = true;
	else
	  vecCoveredObsBySimpleRule[obs][j] = false;
      }

      isPrevPos = false;
      k=0;

      for (int l=0; l<=data->distFeat[j]; ++l) {
	if ( vecSumEachVal[l]>0) {
	  if ( !isPrevPos ) {
	    vecSimpleRule[j].vecLower.push_back(l);
	    vecSimpleRule[j].vecUpper.push_back(l);
	    ++k;
	  } else if ( isPrevPos && l==data->distFeat[j] ) {
	    vecSimpleRule[j].vecUpper[k-1]=l;
	  }
	  isPrevPos = true;
	} else {
	  if (isPrevPos)
	    vecSimpleRule[j].vecUpper[k-1] = l-1;
	  isPrevPos = false;
	}
      }

      if (k==0) {
	vecSimpleRule[j].vecLower.push_back(0);
	vecSimpleRule[j].vecUpper.push_back(data->distFeat[j]);
      }

      DEBUGPRX(5, data, "j: " << j << " int vecLower: " << vecSimpleRule[j].vecLower
	       << " int vecUpper: " << vecSimpleRule[j].vecUpper << "\n");

    } // end for each attribute

    setOriginalRule();

    for (int j=0; j<NumAttrib; ++j)
      DEBUGPRX(5, data, "j: " << j << " double vecLower: " << vecSimpleRule[j].vecLower
	       << " double vecUpper: " << vecSimpleRule[j].vecUpper << "\n");

    // set covered test observations by simple rules
    for (int i=0; i<data->vecTestData.size(); ++i) { // for each test dataset
      obs = data->vecTestData[i];
      for (int j=0; j<NumAttrib; ++j) {
	if ( isCoveredBySR(obs,j) ) {
	  vecCoveredObsBySimpleRule[obs][j] = true;
	} else {
	  vecCoveredObsBySimpleRule[obs][j] = false;
	  break; // this observation is not covered
	}
      } // end for each attribute
    } // for each test dataset

    if (data->printBoost()) {
      ucout << "vecCoveredObsBySimpleRule\n";
      for (int i=0; i<data->numOrigObs; ++i) { // for each test dataset
	for (int j=0; j<NumAttrib; ++j) {
	  ucout << vecCoveredObsBySimpleRule[i][j] << " ";
	} // end for each attribute
	ucout << "\n";
      } // for each test dataset
      ucout << "\n";
    }

  } // end setSimpleRules


  bool LPBR::isCoveredBySR(const int& obs, const int& j) {

    for (int k=0; k<vecSimpleRule[j].vecLower.size(); ++k)
      if ( vecSimpleRule[j].vecLower[k] <= data->origData[obs].X[j] &&
	   data->origData[obs].X[j] <= vecSimpleRule[j].vecUpper[k] )
	return true;

    return false;
  }


  // set original lower and upper bounds matrices
  void LPBR::setOriginalRule() {

    multimap<int, double>::iterator it;
    double lower, upper, tmpLower, tmpUpper;

    for (int j=0; j<NumAttrib; ++j) {

      for (int k=0; k<vecSimpleRule[j].vecLower.size(); ++k) {

	///////////////////////////// mid point rule //////////////////////////////
	if ( vecSimpleRule[j].vecLower[k] > 0 ) { // lowerBound

	  tmpLower =  getUpperBoundSR(k, j, -1, false);
	  tmpUpper =  getLowerBoundSR(k, j, 0, false);
	  lower =  (tmpLower + tmpUpper) / 2.0;

	  DEBUGPRX(10, data, "(k,j): (" << k << ", " << j
		   << ") matIntLower[j][k]-1: " << vecSimpleRule[j].vecLower[k] -1
		   << " LeastLower: " << tmpLower << "\n"
		   << " matIntLower[j][k]: " << vecSimpleRule[j].vecLower[k]
		   << " GreatestLower: " << tmpUpper << "\n");

	} else lower=-inf; // if matIntLower[k][j] < 0 and matIntLower[k][j] != rma->distFeat[j]

	if ( vecSimpleRule[j].vecUpper[k] < data->distFeat[j] ) { // upperBound

	  tmpLower = getUpperBoundSR(k, j, 0, true);
	  tmpUpper =  getLowerBoundSR(k, j, 1, true);
	  upper = (tmpLower + tmpUpper) / 2.0;

	  DEBUGPRX(10, data, "(k,j): (" << k << ", " << j
		   << ") matIntUpper[j][k]: " << vecSimpleRule[j].vecUpper[k]
		   << " LeastUpper: " << tmpLower << "\n"
		   << " matIntUpper[j][k]+1: " << vecSimpleRule[j].vecUpper[k]+1
		   << " GreatestUpper: " << tmpUpper << "\n");

	} else upper=inf; // if matIntUpper[k][j] < rma->distFeat[j] and matIntUpper[k][j] != 0

	// store values
	vecSimpleRule[j].vecLower[k]=lower;
	vecSimpleRule[j].vecUpper[k]=upper;

	DEBUGPRX(10, data, "j: " << j << " orig lower: " << lower
		 << " orig upper: " << upper << "\n");

      }
    }

  } // end function REPR::setOriginalBounds()


  double LPBR::getLowerBoundSR(int k, int j, int value, bool isUpper) {
    int boundVal; double min = inf;
    multimap<int, double>::iterator it, itlow, itup;
    if (isUpper) boundVal = vecSimpleRule[j].vecUpper[k];
    else         boundVal = vecSimpleRule[j].vecLower[k];
    return data->vecFeature[j].vecIntMinMax[boundVal+value].minOrigVal;
  }


  double LPBR::getUpperBoundSR(int k, int j, int value, bool isUpper)  {
    int boundVal; double max = -inf;
    if (isUpper) boundVal = vecSimpleRule[j].vecUpper[k];
    else         boundVal = vecSimpleRule[j].vecLower[k];
    return data->vecFeature[j].vecIntMinMax[boundVal+value].maxOrigVal;
  }


  // print RMA information
  void LPBR::printRMAInfo() {

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      DEBUGPRX(1, data, "alpha: " << alpha <<
	       ", incumb: " << rma->incumbentValue<< "\n");
      //DEBUGPRX(20, data, "dataWts: " << dataWts  ;
      //			cout << " RMA solution: " << rma->workingSol.value << "\n" );
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

  }


  void LPBR::printRMPSolution() {

    int numMissClass = 0;
    double sumMu = 0;
    double sumGamma=0;

    DEBUGPRX(1, data, "numTrainObs: " << NumObs << "\n" );
    DEBUGPRX(1, data, "numPrimal: " << vecPrimal.size() << "\n" );
    DEBUGPRX(1, data, "vecPrimal: " << vecPrimal  );
    DEBUGPRX(1, data, "numDual: " << vecDual.size() << "\n" );
    DEBUGPRX(1, data, "vecDual: " << vecDual );

    DEBUGPRX(1, data, "alpha: " << vecDual[NumObs] << "\n" );

    for (int i=0; i<vecDual.size()-1; ++i) {
      if (vecDual[i]>0) ++numMissClass;
      sumMu += vecDual[i];
    }

    DEBUGPRX(1, data, "Sum of Mu: " <<  sumMu << "\n");
    DEBUGPRX(1, data, "# missclassified by soft margin: " << numMissClass << "\n");
    DEBUGPRX(1, data, " 1/numSMMissClass " << 1/(double) numMissClass << "\n");

    for (int i=NumObs+2; i<vecPrimal.size(); ++i)
      sumGamma += vecPrimal[i];

    DEBUGPRX(1, data, "Sum of Gamma: " << sumGamma  << "\n");

    DEBUGPRX(0, data, "rho: " << vecPrimal[NumObs] << "   " );
    DEBUGPRX(0, data, "gamma_0: " << vecPrimal[NumObs+1] << "   " );
    DEBUGPRX(0, data, "D: " << D << "   ");

    //													 << " DualObj:" << sumDual << "\n" );
    //DEBUGPRX(1, data, cout << "sumCheck: " << sumCheck << "\n" );
    //DEBUGPRX(1, data, cout << "checkCons: " << checkConst << "\n" );
  }

  ////////////////////// Evaluate class methods ///////////////////////////////////////

  // evaluate error rate in each iteration
  double LPBR::evaluateEachIter(const int & isTest) {

    double errRate, expY=0;
    int numMissPreAdj=0;
    int numUnknown=0;
    int numMissByGuess=0;
    int obs, size;
    int numVar = NumObs+2; // rho and gamma_0
    if (data->initRules() || data->init1DRules()) numVar += NumAttrib;

    if (isTest) size = data->vecTestData.size();
    else        size = data->vecTrainData.size();

#ifdef ACRO_HAVE_MPI
    if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
      // if writePred is enabled and the last column generation iteration
      if ( data->writePred() ) // && (curIter+1==NumIter)
	if (isTest) predTest.resize(data->vecTestData.size());
	else        predTrain.resize(data->vecTrainData.size());
#ifdef ACRO_HAVE_MPI
    }
#endif //  ACRO_HAVE_MPI

    for (int i=0; i<size; ++i) { // for each obsercation

      expY=vecPrimal[NumObs+1];

      if (isTest) obs = data->vecTestData[i];
      else        obs = data->vecTrainData[i];

      if (data->initRules())
	for (int j=0; j<NumAttrib; ++j)  // for each box solution
	  if (vecPrimal[NumObs+2+j]!=0)  // if for the coefficient of box not 0
	    if ( vecCoveredObsBySimpleRule[obs][j] ) // if this observation is covered
	      expY += vecPrimal[NumObs+2+j];

      if (data->init1DRules())
	for (int j=0; j<NumAttrib; ++j)  // for each box solution
	  if (vecPrimal[NumObs+2+j]!=0)  // if for the coefficient of box not 0
	    if ( vecCoveredObsBySimpleRule[obs][j] ) // if this observation is covered
	      expY += vecSMIsPositive[j] ? vecPrimal[NumObs+2+j] : -vecPrimal[NumObs+2+j];

      for (int k=0; k<matOrigLower.size(); ++k)  // for each box solution
	if (vecPrimal[numVar+k]!=0)  // if for the coefficient of box not 0
	  if ( vecCoveredObsByBox[obs][k] ) // if this observation is covered
	    expY += vecCoveredSign[k] ? vecPrimal[numVar+k] : -vecPrimal[numVar+k];

#ifdef ACRO_HAVE_MPI
      if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

	if ( data->writePred() )// && (curIter+1==NumIter) )
	  if (isTest) predTest[i]  = expY;
	  else        predTrain[i] = expY;

	DEBUGPRX(2, data, "actY : expY = "
		 << data->origData[obs].y << " : " << expY << "\n" ) ;

#ifdef ACRO_HAVE_MPI
      }
#endif //  ACRO_HAVE_MPI

      if ((expY<=-0.00001 || expY>=0.00001) && expY*data->origData[obs].y < 0 )
	++numMissPreAdj;

      if (expY>=-0.00001 && expY<=0.00001) {
	++numUnknown;
	if (data->numPosTrainObs > data->numNegTrainObs && data->origData[obs].y < 0)
	  ++numMissByGuess;
	else if (data->numPosTrainObs <= data->numNegTrainObs && data->origData[obs].y > 0)
	  ++numMissByGuess;
      }

    } // enf dor each observation

    unknownRate.resize(2);
    preAdjErr.resize(2);

    if (isTest) unknownRate[TEST]  = numUnknown / (double) size;
    else        unknownRate[TRAIN] = numUnknown / (double) size;

    if (isTest) preAdjErr[TEST]  = numMissPreAdj / (double) size;
    else        preAdjErr[TRAIN] = numMissPreAdj / (double) size;

    /*
      isTest ? cout << "Test " : cout << "Train ";
      cout << "unknown prediction error rate: " << numUnknown / (double) size << "\n";

      isTest ? cout << "Test " : cout << "Train ";
      cout << "before adjusted error rate: " << numMissClassified / (double) size << "\n";
    */

    errRate = (numMissPreAdj+numMissByGuess) / (double) size;
    //DEBUGPRX(1, data, "Miss Classified Rate: " << errRate << "\n" ) ;

    return errRate;

  }	// evaluateEachIter function


  void LPBR::printEachIterAllErrs() {
    ucout << "Test/Train preAdjErr (unknownRate): "
	  << preAdjErr[TEST] << " (" << unknownRate[TEST] << ")\t"
	  << preAdjErr[TRAIN] << " (" << unknownRate[TRAIN] << ")\n";
  }


  // evaluate error rate in the end of iterations
  double LPBR::evaluateAtFinal(const int & isTest) {

    double errRate, expY=0;
    int numMissClassified=0;
    int obs, size;
    int numVar = data->vecTrainData.size()+1;

    if (isTest) size = data->vecTestData.size();
    else        size = data->vecTrainData.size();

    // if writePred is enabled and the last column generation iteration
    if ( data->writePred() && (curIter+1==NumIter) )
      if (isTest) predTest.resize(data->vecTestData.size());
      else        predTrain.resize(data->vecTrainData.size());

    for (int i=0; i<size; ++i) { // for each obsercation

      expY=0;

      if (isTest) obs = data->vecTestData[i];
      else        obs = data->vecTrainData[i];

      for (int k=0; k<matOrigLower.size(); ++k) { // for each box solution

	// if the coefficients (gamma^+-gamma^-)=0 for the box[k] is not 0
	if (!(vecPrimal[numVar+2*k]-vecPrimal[numVar+2*k+1]==0)) {

	  for (int j=0; j<NumAttrib; ++j) { // for each attribute

	    if (matOrigLower[k][j] <= data->origData[obs].X[j] &&
		data->origData[obs].X[j] <= matOrigUpper[k][j] ) {
	      if ( j==NumAttrib-1) { // all features are covered by the box
		expY += vecPrimal[numVar+2*k]-vecPrimal[numVar+2*k+1];
		DEBUGPRX(20, data, "kth box: " << k
			 << "box exp: " << expY << "\n");
	      }
	    } else break; // this observation is not covered

	  } // end for each attribute, j

	} // end if for the coefficient of box not 0

      } // end for each box

#ifdef ACRO_HAVE_MPI
      if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

	if ( data->writePred() && (curIter+1==NumIter) )
	  if (isTest) predTest[i]  = expY;
	  else        predTrain[i] = expY;

	DEBUGPRX(10, data, "actY : expY = "
		 << data->origData[obs].y << " : " << expY << "\n" ) ;

#ifdef ACRO_HAVE_MPI
      }
#endif //  ACRO_HAVE_MPI

      if ( expY*data->origData[obs].y < 0 )
	++numMissClassified;
      if (expY==0 && data->origData[obs].y < 0)
	++numMissClassified;

    } // enf dor each observation

    errRate = numMissClassified / (double) size;
    DEBUGPRX(10, data, "Miss Classified Rate: " << errRate << "\n" ) ;

    return errRate;

  }	// evaluateAtFinal function


} // namespace boosting
