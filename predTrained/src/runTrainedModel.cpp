#include "runTrainedModel.h"


// read shuffled observation from the data file
void TrainedREPR::readSavedModel(int argc, char **argv) {

  string tmp;

  cout << "reading the saved model...\n";
  ifstream s(argv[1]);               // open the data file

  // check whether or not the file is opened correctly
  if (!s) {
    cerr << "Could not open file \"" << argv[1] << "\"\n";
    return;
  }

  s >> tmp >> numAttrib;        // read # of attributes
  s >> tmp >> numBox;           // read # of boxes

  // numVars = 1 + numAttrib + numBox; // set # of variables

  // print out # of attributes, # of boxes, # of variables
  cout << "numAttrib: " << numAttrib << "\n";
  cout << "numBox: "    << numBox    << "\n";
  // cout << "numVars: "   << numVars    << "\n";

  /******************** read coefficient info *******************/

  s >> tmp >> bias;           // read # of boxes

  vecCoeffLinear.resize(numAttrib);  // resize vecCoeffLinear
  vecCoeffBox   .resize(numBox);     // resize vecCoeffBox

  // skip first few strings of each line
  s >> tmp;

  for (unsigned int j = 0; j < numAttrib; ++j)  // for each coefficient
    s >> vecCoeffLinear[j];

  // skip first few strings of each line
  s >> tmp;

  for (unsigned int k = 0; k < numBox; ++k)     // for each box
    s >> vecCoeffBox[k];

  /******************** read standerdization info *******************/
  s >> tmp >> avgY;
  s >> tmp >> sdY;

  // skip first few strings of each line
  s >> tmp >> tmp >> tmp;
  vecAvgX.resize(numAttrib);
  for (unsigned int j = 0; j < numAttrib; ++j)  // for each coefficient
    s >> vecAvgX[j];

  // skip first few strings of each line
  s >> tmp >> tmp >> tmp;
  vecSdX.resize(numAttrib);
  for (unsigned int j = 0; j < numAttrib; ++j)  // for each coefficient
    s >> vecSdX[j];

  /************************** read box info *******************/
  // resize the lower and upper bounds matrices
  matLower.resize(numBox);

  // set matLower by reading from the file
  for (unsigned int k=0; k<numBox; ++k) { // for each box

    // skip first few strings of each line
    s >> tmp >> tmp >> tmp;

    // set the dimension of lower bounds for each observations
    matLower[k].resize(numAttrib);

    for (unsigned int j=0; j<numAttrib; ++j)  // for each attribute
      s >> matLower[k][j];
      // set the lower bound of observation i and its attribute j

  } // end for each box

  matUpper.resize(numBox);

  // set matUpper by reading from the file
  for (unsigned int k=0; k<numBox; ++k) { // for each box

    // skip first few strings of each line
    s >> tmp >> tmp >> tmp;

    // set the dimension of lower bounds for each observations
    matUpper[k].resize(numAttrib);

    for (unsigned int j=0; j<numAttrib; ++j) // for each attribute
      s >> matUpper[k][j];
      // set the upper bound of observation i and its attribute j

  } // end for each box

  // cout << "\n";

  s.close(); // close the data file

  // cout << "bias: "    << bias << "\n";
  // printVecCoeffLinear();
  // printVecCoeffBox();

  // printMatLower();
  // printMatUpper();

} // end readSavedModel function


// read X-values from a file
void TrainedREPR::readX(int argc, char **argv) {

  string line;
  double tmp;

  cout << "reading the test data...\n";
  ifstream s(argv[2]);              // open the data file

  // check whether or not the file is opened correctly
  if (!s) {
    cerr << "Could not open file \"" << argv[2] << "\n";
    return;
  }

  // tc.startTime();

  numObs     = 0;
  numAttrib  = 0;

  // read data from the data file
  if (argc <= 2) {
    cerr << "No filename specified\n";
    return;
  }

  // read how many columns and rows
  while (getline(s, line)) { // for each line
    if (numObs == 0) {
      istringstream streamCol(line);
      while (streamCol >> tmp) // for each #
        ++numAttrib;  // count # of attribute
    }
    ++numObs;  // # of observations
  }
  --numAttrib; // last line is response value

  cout << "(mxn): " << numObs << "\t" << numAttrib << "\n";

  s.clear();
  s.seekg(0, ios::beg);

  matTestDataX.resize(numObs);  // resize the row of matTestDataX

  for (int i = 0; i < numObs; ++i) { // for each observation

    matTestDataX[i].resize(numAttrib);  // resize the column of each row

    for (int j = 0; j < numAttrib; j++)  // for each attribute
      s >> matTestDataX[i][j]; // read X_{ij}

    s >> tmp;

  } // end for each observation

  s.close(); // close the data file

  // printMatTestDataX();

} // end readX function


// wether or not observation "i" is covered by box "k"
bool TrainedREPR::isObsCovered(int k, int i) {

  for (int j=0; j<numAttrib; ++j) { // for each attribute
    if ( ! ( (matLower[k][j] <= matTestDataX[i][j])
             && (matTestDataX[i][j] <= matUpper[k][j]) ) )
      return false; // this observation i is not covered by k-th box
  } // end for each attribute

  return true; // this observation i is covered by k-th box
} // end isObsCovered function


// set up vecIsObsCovered
void TrainedREPR::setMatIsObsCovered() {

  matIsObsCovered.resize(numBox); // resize the row

  for (unsigned int k=0; k<numBox; ++k) { // for each box

    matIsObsCovered[k].resize(numObs);  // resize the column of each row

    // set observation is is covere by k-th box
    for (unsigned int i=0; i<numObs; ++i)  // for each observation
      matIsObsCovered[k][i] = isObsCovered(k, i);

  } // end for each box

  // printMatIsCovered();

}  // end setMatIsObsCovered function


void TrainedREPR::standerdizeX() {
  for (unsigned int j=0; j<numAttrib; ++j)
    for (unsigned int i=0; i<numObs; ++i)
      matTestDataX[i][j] = ( matTestDataX[i][j] - vecAvgX[j] ) / vecSdX[j];
}


void TrainedREPR::mapToOriginalY() {
  for (unsigned int i=0; i<numObs; ++i)
    vecPredY[i] = vecPredY[i] * sdY + avgY;
}


// set predicted y-values
void TrainedREPR::setVecPredY() {

  vecPredY.resize(numObs);  // set the size of vecPredY

  // predict y-value for each observation
  for (unsigned int i=0; i<numObs; ++i) { // for each observation

    vecPredY[i] = bias;        // constant term (\beta_0)

    // add the sum of the product of linear cofficients and variables
    // ( \sum_i=1^n \beta_j * X_ij )
    for (unsigned int j=0; j<numAttrib; ++j)  // for each attribute
      vecPredY[i] += vecCoeffLinear[j] * matTestDataX[i][j];

    // box regression term
    for (unsigned int k=0; k<numBox; ++k)  // for each box
      vecPredY[i] += vecCoeffBox[k] * matIsObsCovered[k][i];

  }  // end for each observation

  // printVecPredY();

}  // end setVecPredY function


// predict y value using X (return the vector of predicted y-values)
vector<double> TrainedREPR::predict() {

  setMatIsObsCovered();  // set vecObsCovered (each observation is covered by each box)

  standerdizeX();        // standerdize X values befoe feed them into the model

  setVecPredY();         // set predicted y-values

  mapToOriginalY();      // mpp the y-value to the original values

  printVecPredY();

  return vecPredY;

} // end predict function


/*************************** utility functions ********************/


// for 1D vector ooutput
template<class T>
ostream &operator<<(ostream &os, const vector<T> &v) {
  os << "(";
  for (typename vector<T>::const_iterator i = v.begin(); i != v.end(); ++i)
    os << " " << *i;
  os << " )\n";
  return os;
}

template ostream &operator<< <bool>(ostream &os, const vector<bool> &v);

template ostream &operator<< <unsigned int>(ostream &os,
                                            const vector<unsigned int> &v);


// for 2D vector output
template<class T>
ostream &operator<<(ostream &os, const vector<vector<T> > &v) {
  for (unsigned int i = 0; i < v.size(); ++i) {
    os << "(";
    for (unsigned int j = 0; j < v[i].size(); ++j)
      os << v[i][j] << " ";
    os << " )\n";
  }
  return os;
}

template ostream &operator<< <unsigned int>(ostream &os,
                 const vector<vector<unsigned int> > &v);

template ostream &operator<< <double>(ostream &os,
                 const vector<vector<double> > &v);

template ostream &operator<< <bool>(ostream &os,
                 const vector<vector<bool> > &v);


// for 2D deque output
template<class T>
ostream &operator<<(ostream &os, const deque<deque<T> > &v) {
 for (unsigned int i = 0; i < v.size(); ++i) {
   os << "(";
   for (unsigned int j = 0; j < v[i].size(); ++j)
     os << v[i][j] << " ";
   os << " )\n";
 }
 return os;
}

template ostream &operator<< <bool>(ostream &os,
                const deque<deque<bool> > &v);
