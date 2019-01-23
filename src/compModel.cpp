/*
 *  File name:   compModel.cpp
 *  Author:      Ai Kagawa
 *  Description: a source file for compModel classes
 */

#include "compModel.h"


namespace comparison {


//////////////////////// Competing model CVs ////////////////////////

void compModel::setCompModelsCV() {

#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

		loadRlibrary();

		avgRunTime.resize(NumCompModel);
		avgTrainMSE.resize(NumCompModel);
		avgTestMSE.resize(NumCompModel);

		for (int i=0; i<NumCompModel; ++i) {
			avgRunTime[i] = 0.0;
			avgTrainMSE[i] = 0.0 ;
			avgTestMSE[i] = 0.0 ;
		}

#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI

}


void compModel::printCompModelsCV() {

#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

		for (int j=0; j<NumCompModel; ++j) {
			modelName(j);
			ucout << "Avg MSE Test/Train: "
				<< avgTestMSE[j] / (double) NumPartition << "\t"
				<< avgTrainMSE[j] / NumPartition << "\n";
		}

		for (int j=0; j<NumCompModel; ++j) {
			modelName(j);
			ucout << "Avg Time: " << avgRunTime[j]
                            / (double) NumPartition << "\n" ;
		}

#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI
}


////////////////////////////// REPR ////////////////////////////////////////


void compREPR::loadRlibrary() {
/*
#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
*/
  /*if (base->compareModels()) {
    //load library, no return value
		string cmd = " require(methods); library(glmnet); library(randomForest); platform = \"linux\"; \
			 rfhome = \"/home1/ak907/R/library/RuleFit\"; \
			 source(\"/home1/ak907/R/library/RuleFit/rulefit.r\");  \
			 library(akima); library(gbm);";
	 	R.parseEvalQ(cmd);
*/

	string cmd = " require(methods); library(glmnet); library(randomForest); platform = \"linux\"; \
			rfhome = \"/home/aik/R/x86_64-pc-linux-gnu-library/3.4/RuleFit\"; \
			source(\"/home/aik/R/x86_64-pc-linux-gnu-library/3.4/RuleFit/rulefit.r\");  \
			library(akima); library(gbm);";
	R.parseEvalQ(cmd);
  //}
/*
#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI
//*/
}


void compREPR::printCompModels() {

#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

		for (int j=RuleFit; j<NumCompModel; ++j) {
			modelName(j);
			ucout << " Test/Train Errors: "
					 << testMSE[j] << "\t" << trainMSE[j] << '\n';
		}

#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI
}


void compREPR::runCompModels() {

	vector<double> vecRBeta;

	int numTrainData = data->vecTrainData.size();
	int numTestData  = data->vecTestData.size();

	trainMSE.resize(NumCompModel);
	testMSE.resize(NumCompModel);

	predTest.resize(NumCompModel+2);
	if ( data->writePred() ) {
		for (int i=0; i<NumCompModel+2; ++i)
			predTest[i].resize(numTestData);
	}

	string cmd;
	Rcpp::NumericMatrix TrainX(numTrainData, data->numAttrib),
											TestX(numTestData, data->numAttrib);
	Rcpp::DoubleVector TrainY(numTrainData), TestY(numTestData);

	for (int l=0; l<numTrainData; ++l) {
		for (int j=0; j<data->numAttrib; ++j)
			TrainX[j*numTrainData+l] = data->origData[data->vecTrainData[l]].X[j];
		TrainY[l] = data->origData[data->vecTrainData[l]].y;
	}

	for (int l=0; l<numTestData; ++l) {
		for (int j=0; j<data->numAttrib; ++j)
			TestX[j*numTestData+l] = data->origData[data->vecTestData[l]].X[j];
		TestY[l] = data->origData[data->vecTestData[l]].y;
	}

	R["trainX"] = TrainX;
	R["trainY"] = TrainY;
	R["testX"] = TestX;
	R["testY"] = TestY;
	R["compIter"] = data->getCompModelIters();

	////////////////////////////////// Random Forest //////////////////////////////////

	tc.startTime();

	cmd = "fit <- randomForest(trainX, trainY, n.tree=compIter)";
	R.parseEval(cmd);

	// TRAIN
	cmd = "mse=0.0; for (i in 1:nrow(trainX))"
	      "{ mse = mse + (trainY[i] - predict(fit, trainX[i,], n.tree=compIter))^2 }; "
				"mse=mse/nrow(trainX); mse";
	trainMSE[RandFore] = Rcpp::as< double >(R.parseEval(cmd)); // parse, eval + return result
	avgTrainMSE[RandFore] += trainMSE[RandFore];

	// TEST
	cmd = "mse=0.0; for (i in 1:nrow(testX))"
				"{ mse = mse + (testY[i] - predict(fit, testX[i,], n.tree=compIter))^2 }; "
				"mse=mse/nrow(testX);mse";

	testMSE[RandFore] = Rcpp::as< double >(R.parseEval(cmd)); // parse, eval + return result
	avgTestMSE[RandFore] += testMSE[RandFore];

	// Predictions
	if (data->writePred()) {
		cmd = "for (i in 1:nrow(testX))"
		      "{ predRandForest[i]=predict(fit, testX[i,], n.tree=compIter) };"
					"predRandForest " ;
		predTest[RandFore] = Rcpp::as< vector<double> >(R.parseEval(cmd));
	}

	cout << "RandomForest ";
	avgRunTime[RandFore] += tc.endWallTime();

	////////////////////////////////// RuleFit //////////////////////////////////
	//cmd = "rfmod = rulefit (x, y, wt=rep(1,nrow(x)), cat.vars=NULL, not.used=NULL, xmiss=9.0e30, rfmode=\"regress\", sparse=1)";
	tc.startTime();
	cmd = "rfmod = rulefit (trainX, trainY, rfmode=\"regress\", sparse=1)";
	R.parseEval(cmd);

	// TRAIN
	cmd = "mse=0.0; for (i in 1:nrow(trainX)) { mse = mse + (trainY[i] - rfpred(trainX[i,]))^2 }; "
			   "mse=mse/nrow(trainX); mse";
	trainMSE[RuleFit] = Rcpp::as< double >(R.parseEval(cmd)); // parse, eval + return result
	avgTrainMSE[RuleFit] += trainMSE[RuleFit];

	// TEST
	cmd = "mse=0.0; for (i in 1:nrow(testX)) {mse = mse + (testY[i] - rfpred(testX[i,]))^2 }; "
				"mse=mse/nrow(testX); mse";
	testMSE[RuleFit] = Rcpp::as< double >(R.parseEval(cmd)); // parse, eval + return result
	avgTestMSE[RuleFit] += testMSE[RuleFit];

	// Predictions
	if (data->writePred()) {
		cmd = "for (i in 1:nrow(testX)) { predRuleFit[i]=rfpred(testX[i,]) }; predRuleFit " ;
		predTest[RuleFit] = Rcpp::as< vector<double> >(R.parseEval(cmd));
	}

	cout << "RuleFit ";
	avgRunTime[RuleFit] += tc.endWallTime();

	////////////////////////////////// Gradient Boosting in R //////////////////////////////////
	tc.startTime();
	cmd = "mse=0.0; obj <- gbm.fit(trainX, trainY, interaction.depth=4, n.trees=compIter, distribution=\"gaussian\");" ;
	R.parseEval(cmd);

	// TRAIN
	cmd = "predGBY = predict(obj, trainX, n.trees = compIter); mean((predGBY - trainY)^2) ";
	trainMSE[GradBoost] = Rcpp::as< double >(R.parseEval(cmd)); // parse, eval + return result
	avgTrainMSE[GradBoost] += trainMSE[GradBoost];

	// TEST
	cmd = "predGBY = predict(obj, testX, n.trees = compIter); mean((predGBY - testY)^2) ";
	testMSE[GradBoost] = Rcpp::as< double >(R.parseEval(cmd)); // parse, eval + return result
	avgTestMSE[GradBoost] += testMSE[GradBoost];

	// Predictions
	if (data->writePred()) {
		cmd = "predGBY = predict(obj, testX, n.trees = compIter)" ;
		predTest[GradBoost] = Rcpp::as<vector<double> >(R.parseEval(cmd)); // parse, eval + return result
		//ucout << "vecGBPred: " << GBPred ;
	}

	cout << "Gradient Boosting ";
	avgRunTime[GradBoost] += tc.endWallTime();

	////////////////////////////////// linear regressing in R //////////////////////////////////

	tc.startTime();
	cmd = "model=lm(trainY~trainX); sm <- summary(model); mean(sm$residuals^2)";

	// TRAIN
	trainMSE[LinReg] = Rcpp::as< double >(R.parseEval(cmd)); // parse, eval + return result
	avgTrainMSE[LinReg] += trainMSE[LinReg] ;

	cmd = "coef=coef(model)"; //[2:length(coef(model))]
	vecRBeta.clear();
	vecRBeta = Rcpp::as<vector<double> >(R.parseEval(cmd)); // parse, eval + return result
	//ucout << "vecRBeta: " << vecRBeta ;

	//******************************* test *********************
	DEBUGPRX(10, data, " y Rpred SqErr \n");
	double pred, mse=0.0;
	predTest[LinReg].resize(numTestData);
	for (int l=0; l<numTestData; ++l) { // for each test data
		pred = vecRBeta[0];
		for (int j=0; j<data->numAttrib; ++j)
			if (!isnan(vecRBeta[j+1]))
				pred += vecRBeta[j+1] * (double) data->origData[data->vecTestData[l]].X[j];
		predTest[LinReg][l] =  pred;
		testMSE[LinReg] += pow ( data->origData[data->vecTestData[l]].y - pred, 2 ) ;
		DEBUGPRX(10, data, " "
			<< data->origData[data->vecTestData[l]].y << " - "	<< pred << " : "
			<< pow ( data->origData[data->vecTestData[l]].y - pred, 2 ) << "\n");
	}
	testMSE[LinReg] /= numTestData;
	avgTestMSE[LinReg] += testMSE[LinReg] ;

	cout << "LinearRegression ";
	avgRunTime[LinReg] += tc.endWallTime();

}


////////////////////////////// LPBR ////////////////////////////////////////


void compLPBR::loadRlibrary() {
///*
#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI
//*/
	cmd = " require(methods); library(glmnet); library(randomForest);  \
			library(rpart); library(ada); library(fastAdaboost); library(akima); library(gbm);";
	R.parseEvalQ(cmd);
//*
#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI
//*/
}


void compLPBR::printCompModels() {

#ifdef ACRO_HAVE_MPI
	if (uMPI::rank==0) {
#endif //  ACRO_HAVE_MPI

		for (int j=AdaBoost; j<NumCompModel; ++j) {
			modelName(j);
			ucout << " Test/Train Errors: "
					 << testMSE[j] << "\t" << trainMSE[j] << '\n';
		}

#ifdef ACRO_HAVE_MPI
	}
#endif //  ACRO_HAVE_MPI
}


void compLPBR::runCompModels() {

	int numTrainData = data->vecTrainData.size();
	int numTestData = data->vecTestData.size();
	int numMiss;

	trainMSE.resize(NumCompModel);
	testMSE.resize(NumCompModel);

	//if ( data->writePred() ) {
		predTrain.resize(NumCompModel+2);
		predTest.resize(NumCompModel+2);
		for (int i=0; i<NumCompModel+2; ++i) {
			predTrain[i].resize(numTrainData);
			predTest[i].resize(numTestData);
		}
	//}

	Rcpp::NumericMatrix TrainX(numTrainData, data->numAttrib),
											TestX(numTestData, data->numAttrib);
	Rcpp::DoubleVector TrainY(numTrainData), TestY(numTestData);

	for (int i=0; i<numTrainData; ++i) {
		TrainY[i] = data->origData[data->vecTrainData[i]].y;
		for (int j=0; j<data->numAttrib; ++j)
			TrainX[j*numTrainData+i] = data->origData[data->vecTrainData[i]].X[j];
	}

	for (int i=0; i<numTestData; ++i) {
		TestY[i] = data->origData[data->vecTestData[i]].y;
		for (int j=0; j<data->numAttrib; ++j)
			TestX[j*numTestData+i] = data->origData[data->vecTestData[i]].X[j];
	}

	R["trainX"] = TrainX;
	R["testX"] = TestX;
	R["trainY"] = TrainY;
	R["testY"] = TestY;
	R["trainY1"] = TrainY;
	R["testY1"] = TestY;
	R["compIter"] = data->getCompModelIters();

/*
cmd =	"trainData <- data.frame(X=trainX, Y=trainY); "
			"testData  <- data.frame(X=testX, Y=testY); "
			"trainData$Y <- factor(trainData$Y);"
			"testData$Y  <- factor(testData$Y);";
			*/
	cmd = "X=trainX; Y=trainY;"
	      "trainData <- data.frame(X, Y); "
				"trainData$Y <- factor(trainData$Y);";
	R.parseEval(cmd);

	////////////////////////////////// Adaboost //////////////////////////////////
	tc.startTime();

	cmd = "ada = adaboost(Y~X, trainData, compIter);"
				"predTrain <- predict(ada, newdata=trainData); predTrain$class";
	predTrain[AdaBoost] = Rcpp::as< vector<double> >(R.parseEval(cmd));

	numMiss=0;
	for (int i=0; i<numTrainData; ++i) {
		if (predTrain[AdaBoost][i] == 2) predTrain[AdaBoost][i]=-1;
		if (predTrain[AdaBoost][i] != TrainY[i]) ++numMiss;
	}
	trainMSE[AdaBoost] = numMiss / (double) numTrainData;
	avgTrainMSE[AdaBoost] += trainMSE[AdaBoost];

	cmd = "X=testX; Y=testY;"
	      "testData  <- data.frame(X, Y); "
				"testData$Y  <- factor(testData$Y);";
	R.parseEval(cmd);

	cmd = "predTest <- predict(ada, newdata=testData); predTest$class";
  predTest[AdaBoost] = Rcpp::as< vector<double> >(R.parseEval(cmd));

	numMiss=0;
	for (int i=0; i<numTestData; ++i) {
		if (predTest[AdaBoost][i] == 2) predTest[AdaBoost][i]=-1;
		if (predTest[AdaBoost][i] != TestY[i]) ++numMiss;
	}
	testMSE[AdaBoost] = numMiss / (double) numTestData;
	avgTestMSE[AdaBoost] += testMSE[AdaBoost];

	ucout << "AdaBoost     ";
	avgRunTime[AdaBoost] += tc.endWallTime();

	/*
		cmd = "ada = adaboost(Y~X, trainData, compIter);"
				  "predTrain <- predict(ada, newdata=trainData); predTrain$error";

		trainMSE[AdaBoost] = Rcpp::as< double >(R.parseEval(cmd));
		avgTrainMSE[AdaBoost] += trainMSE[AdaBoost];

		cmd = "X=testX; Y=testY;"
		      "testData  <- data.frame(X, Y); "
					"testData$Y  <- factor(testData$Y);";
		R.parseEval(cmd);

		cmd = "predTest <- predict(ada, newdata=testData); predTest$error";
		testMSE[AdaBoost] = Rcpp::as< double >(R.parseEval(cmd));
		avgTestMSE[AdaBoost] += testMSE[AdaBoost];

		// Predictions
		if (data->writePred()) {
			cmd = "predTrain$class" ;
			predTrain[AdaBoost] = Rcpp::as< vector<double> >(R.parseEval(cmd));
			for (int i=0; i<numTrainData; ++i)
				if (predTrain[AdaBoost][i] == 2) predTrain[AdaBoost][i]=-1;
			cmd = "predTest$class" ;
			predTest[AdaBoost] = Rcpp::as< vector<double> >(R.parseEval(cmd));
			for (int i=0; i<numTestData; ++i)
				if (predTest[AdaBoost][i] == 2) predTest[AdaBoost][i]=-1;
		}

	*/

/*
	cmd = "ada = adaboost(trainY~trainX, trainData, compIter);"
				"predTrain <- predict(ada, newdata=trainData); predTrain$error";
	trainMSE[AdaBoost] = Rcpp::as< double >(R.parseEval(cmd));
	avgTrainMSE[AdaBoost] += trainMSE[AdaBoost];

	cmd = "predTest <- predict(ada, newdata=testData); predTest$error";
	predTest[AdaBoost] = Rcpp::as< vector<double> >(R.parseEval(cmd));
	avgTestMSE[AdaBoost] += testMSE[AdaBoost];
*/


	////////////////////////////////// Random Forest //////////////////////////////////
//*
	tc.startTime();

	cmd =	"trainData <- data.frame(trainX, trainY); "
				"testData  <- data.frame(testX, testY); "
				"trainY <- factor(trainY);"
				"testY  <- factor(testY);";
	R.parseEval(cmd);

	cmd = "rf <- randomForest(trainX, trainY);" ;
	R.parseEval(cmd);

	cmd = "predTrain = predict(rf, trainX)";
	predTrain[RandFore] = Rcpp::as< vector<double> >(R.parseEval(cmd));
	numMiss=0;
	for (int i=0; i<numTrainData; ++i) {
		if (predTrain[RandFore][i] == 1) predTrain[RandFore][i]=-1;
		else if (predTrain[RandFore][i] == 2) predTrain[RandFore][i]=1 ;
		if (predTrain[RandFore][i] != TrainY[i]) ++numMiss;
	}
	trainMSE[RandFore] = numMiss / (double) numTrainData;
	avgTrainMSE[RandFore] += trainMSE[RandFore];

	cmd = "predTest = predict(rf, testX) ";
	predTest[RandFore] = Rcpp::as< vector<double> >(R.parseEval(cmd));
	numMiss=0;
	for (int i=0; i<numTestData; ++i) {
		if (predTest[RandFore][i] == 1) predTest[RandFore][i]=-1;
		else if (predTest[RandFore][i] == 2) predTest[RandFore][i]=1 ;
		if (predTest[RandFore][i] != TestY[i]) ++numMiss;
	}
	testMSE[RandFore] = numMiss / (double) numTestData;
	avgTestMSE[RandFore] += testMSE[RandFore];

/*
	cmd = "predTrain = predict(rf, trainX);"
	      "err=0.0; for (i in 1:nrow(trainX)) { "
				" if ( trainY[i]!=predTrain[i] ) {err<-err+1;} }; "
        //" else if ( trainY[i]>0 & predTrain[i]==0 ) {err<-err+1;} }; "
				"err<-err/nrow(trainX)" ;
	trainMSE[RandFore] = Rcpp::as< double >(R.parseEval(cmd)); // parse, eval + return result
	avgTrainMSE[RandFore] += trainMSE[RandFore];

	// TEST
	cmd = "predTest = predict(rf, testX); "
				"err=0.0; for (i in 1:nrow(testX)) { "
				" if ( testY[i]!=predTest[i] ) {err<-err+1;} }; "
				//" else if ( testY[i]>0 & predTest[i]==0 ) {err<-err+1;} }; "
				"err<-err/nrow(testX)" ;
	testMSE[RandFore] = Rcpp::as< double >(R.parseEval(cmd)); // parse, eval + return result
	avgTestMSE[RandFore] += testMSE[RandFore];

	// Predictions
	if (data->writePred()) {
		cmd = "predTrain" ;
		predTrain[RandFore] = Rcpp::as< vector<double> >(R.parseEval(cmd));
		cmd = "predTest" ;
		predTest[RandFore] = Rcpp::as< vector<double> >(R.parseEval(cmd));
		for (int i=0; i<numTrainData; ++i) {
			if (predTrain[RandFore][i] == 1) predTrain[RandFore][i]=-1;
			else if (predTrain[RandFore][i] == 2) predTrain[RandFore][i]=1;
		}
		for (int i=0; i<numTestData; ++i) {
			if (predTest[RandFore][i] == 1) predTest[RandFore][i]=-1;
			else if (predTest[RandFore][i] == 2) predTest[RandFore][i]=1 ;
		}
	}
*/
	ucout << "RandomForest ";
	avgRunTime[RandFore] += tc.endWallTime();

	////////////////////////////////// Gradient Boosting in R //////////////////////////////////
/*
	tc.startTime();

	cmd = "for (i in 1:nrow(trainX)) { if (trainY1[i]<0) {trainY1[i] = 0;} };"
				"for (i in 1:nrow(testX))  { if (testY1[i]<0)  {testY1[i]  = 0;} };"
				"trainData <- data.frame(trainX, trainY); "
				"testData  <- data.frame(testX, testY); ";
	R.parseEval(cmd);

	cmd = "gb <- gbm.fit(trainX, trainY1, n.trees=compIter, distribution=\"bernoulli\");" ;
	// cmd = "gb <- gbm( trainY1~trainX, data=trainData, n.trees=compIter, distribution=\"bernoulli\");" ;
	// cmd = "gb <- gbm.fit(trainX, trainY1, interaction.depth=4, n.trees=5000, distribution=\"bernoulli\");" ;
	R.parseEval(cmd);

	// TRAIN
	cmd = "predTrain = predict(gb, trainX, n.trees = compIter);"
				"err=0.0; for (i in 1:nrow(trainX)) { "
				" if ( trainY1[i]!=predTrain[i] ) {err<-err+1;} }; "
				//" else if ( trainY[i]>0 & predTrain[i]==0 ) {err<-err+1;} }; "
				"err<-err/nrow(trainX)" ;
	trainMSE[GradBoost] = Rcpp::as< double >(R.parseEval(cmd));
	avgTrainMSE[GradBoost] += trainMSE[GradBoost];

	// TEST
	cmd = "predTest = predict(gb, testX, n.trees = compIter); "
				"err=0.0; for (i in 1:nrow(testX)) { "
				" if ( testY1[i]!=predTest[i] ) {err<-err+1;} }; "
				//" else if ( testY[i]>0 & predTest[i]==0 ) {err<-err+1;} }; "
				"err<-err/nrow(testX)" ;
	testMSE[GradBoost] = Rcpp::as< double >(R.parseEval(cmd));
	avgTestMSE[GradBoost] += testMSE[GradBoost];

	// Predictions
	if (data->writePred()) {
		cmd = "predTrain" ;
		predTrain[GradBoost] = Rcpp::as< vector<double> >(R.parseEval(cmd));
		cmd = "predTest" ;
		predTest[GradBoost] = Rcpp::as< vector<double> >(R.parseEval(cmd));
	}

	ucout << "Gradient Boosting ";
	avgRunTime[GradBoost] += tc.endWallTime();
*/
}


} // namespace comparison
