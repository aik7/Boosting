/*
 *  File name:   main.cpp
 *  Author:      Ai Kagawa
 *  Description: a dirver to run LPBoost or REPR with cross-validation
 */

#include "./include/crossvalid.h"

using namespace crossvalidation;


int main(int argc, char** argv) {

	CrossValidation cv;
	cv.setupData(argc, argv);
	cv.runOuterCrossValidation();

	return 0;

}
