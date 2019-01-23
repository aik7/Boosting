/*
 *  File name:   compModel.h
 *  Author:      Ai Kagawa
 *  Description: a header file for CompetingModels classes
 */

#include <pebbl_config.h>
#include <pebbl/utilib/ParameterSet.h>
#include <RInside.h>                    // for the embedded R via RInside
#include <Rcpp.h>

#include <vector>

#include "Time.h"
#include "base.h"

#ifndef CompModels_h
#define CompModels_h

namespace comparison {

  using namespace base;
  using namespace std;


class compModel {

public:

  compModel() {}
  virtual ~compModel() {}

  void setCompModelData(Data* d) {data=d;}

  void setCompModelsCV();

  void printCompModelsCV();

  virtual void loadRlibrary() = 0;

  virtual void runCompModels() = 0;

  virtual void printCompModels() = 0;

  virtual void modelName(int c) = 0;

  Time tc;

  Data* data;

  RInside R;				// R object

  string cmd;

  int NumCompModel;

  int NumPartition;

  vector<double> avgTestMSE;	// average MSE for each competing model
  vector<double> avgTrainMSE;	// average MSE for each competing model

  vector<double> trainMSE;	// test and train MSE for competing models
  vector<double> testMSE;	// test and train MSE for competing models

  vector<vector<double> > predTrain;
  vector<vector<double> > predTest;

  vector<double> avgRunTime;	// average running time for all models


};


class compREPR : public compModel {

  enum compModels { RuleFit, RandFore, GradBoost, LinReg};

public:

  compREPR(Data* _data) {
    data = _data;
    NumCompModel = 4;
    NumPartition = 5; // TODO: fix this later
  }

  void loadRlibrary();

  void runCompModels();

  void printCompModels();

  void modelName(int c) {
    if (c==0)      { ucout << "RuleFit:    "; return; }
    else if (c==1) { ucout << "RandForest: "; return; }
    else if (c==2) { ucout << "GradBoost:  "; return; }
    else if (c==3) { ucout << "LinearReg:  "; return; }
  }

};


class compLPBR : public compModel {

  enum compModels { AdaBoost, RandFore, GradBoost};

public:

  compLPBR(Data* _data)  {
    data = _data;
    NumCompModel = 2;
    NumPartition = 5;  // TODO: fix this later
  }

  void loadRlibrary();

  void runCompModels();

  void printCompModels();

  void modelName(int c) {
    //if (c==Ada)      { cout << "Ada:    "; return; }
    if (c==AdaBoost)       { cout << "AdaBoost:   "; return; }
    else if (c==RandFore)  { cout << "RandForest: "; return; }
    else if (c==GradBoost) { cout << "GradBoost:  "; return; }
    //else if (c==3) { cout << "LinearReg:  "; return; }
  }

};


} // namespace comparison

#endif
