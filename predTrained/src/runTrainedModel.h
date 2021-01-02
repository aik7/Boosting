#include <vector>
#include <deque>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;


// for 1D vector output
template<class T>
ostream& operator<<(ostream& os, const vector<T>& v);

// for 2D vector output
template<class T>
ostream& operator<<(ostream& os, const vector<vector<T> >& v);

// for 2D deque output
template<class T>
ostream& operator<<(ostream& os, const deque<deque<T> >& v);


class TrainedREPR {

public:

  TrainedREPR(int argc, char **argv){
    readSavedModel(argc, argv);  // read the saved model
    readX(argc, argv);           // read X-values
  }

  ~TrainedREPR(){}

  // read a saved model
  void readSavedModel(int argc, char **argv);

  // read X-values
  void readX(int argc, char **argv);

  // set matIsObsCovered (info of each observation is covered by each box)
  void setMatIsObsCovered();

  // returns whether or not this observation is covered by k-th box
  bool isObsCovered(int k, int i);   // vector<double> x

  // set predicted y-value of each observation
  void setVecPredY();

  // return vecPredY (retun size: # of observations)
  vector<double> predict();

  /******************* print functions ******************/

  void printVecPredY ()     { cout << "vecPredY: " << vecPredY; }

  void printVecCoeff ()     { cout << "vecCoeff: " << vecCoeff; }

  void printMatTestDataX () { cout << "matTestDataX: " << matTestDataX; }

  void printMatIsCovered () { cout << "matIsObsCovered: " << matIsObsCovered; }

  void printMatLower ()     { cout << "matLower: " << matLower; }

  void printMatUpper ()     { cout << "matUpper: " << matUpper; }

private:

  // # of attributes, # of boxes, # of variables, # of observations
  unsigned int            numAttrib, numBox, numVars, numObs;

  // a vector of predicted y-values (size: # of observations)
  vector<double>          vecPredY;

  // a vector of cofficients of the REPR model (size: # of REPR varaiebls)
  vector<double>          vecCoeff;

  // X matrix for testing (size: [# of observations] * [# of attributes])
  vector<vector<double> > matTestDataX;

  // a matrix includes info of whether or not
  // each observation is covered by each box
  // (size: [# of boxes] * [# of observations])
  deque<deque<bool> >     matIsObsCovered;

  // a matrix contains lower and upper bounds of each box
  // (size: [# of boxes] * [# of attributes])
  vector<vector<double> > matLower;
  vector<vector<double> > matUpper;

}; // end TrainedREPR class


/*************************** utility functions ********************/
