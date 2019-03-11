# MAKEFILE

#CXX=g++
CXX=mpicxx

# This should be altered to point to wherever your acro-pebbl directory actually lives
ACRO_ROOT=/home1/ak907/acro-pebbl
MPI_ROOT=/opt/openmpi/1.10.1
GRB=/home1/ak907/gurobi752/linux64
R_HOME=/home1/ak907/R
Rcpp=/home1/ak907/R/library/Rcpp
RInside=/home1/ak907/R/library/RInside

#Rcpp=/home1/ak907/R/library/Rcpp
#RInside=/home1/ak907/R/library/RInside

######################################### SYMBOLS ########################################
SYMBOLS=HAVE_CONFIG_H ANSI_HDRS ANSI_NAMESPACES
DEFSYMBOLS=$(patsubst %,-D%,$(SYMBOLS))

######################################## INCLUDES ##########################################
HEADERDIRS=. $(ACRO_ROOT)/include $(ACRO_ROOT)/include/pebbl $(ACRO_ROOT)/include/utilib \
             $(GRB)/include $(R_HOME)/include $(Rcpp)/include $(RInside)/include #$(MPI_ROOT)/include 
INCLUDES=$(patsubst %,-I%,$(HEADERDIRS))

######################################### LIB ############################################
LIBDIRS=$(ACRO_ROOT)/lib $(GRB)/lib $(R_HOME)/lib $(Rcpp)/libs $(RInside)/lib $(RInside)/libs #$(MPI_ROOT)/lib
LIBLOCATIONS=$(patsubst %,-L%,$(LIBDIRS))
LIBS=pebbl utilib gurobi_c++ gurobi75 Rblas Rlapack R RInside #mpi #mpi_cxx open-rte open-pal
LIBSPECS=$(patsubst %,-l%,$(LIBS)) 

########################################## FLAGS ##########################################
DEBUGFLAGS=-g -fpermissive -O0 #-D_GLIBCXX_USE_CXX11_ABI=1 #-O0 
MISCCXXFLAGS= -std=c++98 

# include
CXXFLAGS=$(DEFSYMBOLS) $(INCLUDES) $(MISCCXXFLAGS) $(DEBUGFLAGS) \
         $(RCPPFLAGS) $(RCPPINCL) $(RCPPEIGENINCL) $(RINSIDEINCL)
# Library
LDFLAGS=$(DEBUGFLAGS) $(LIBLOCATIONS) \
        $(RLDFLAGS) $(RRPATH) $(RBLAS) $(RLAPACK) $(RCPPLIBS) $(RINSIDELIBS) 

#####################################################################################

SRCDIR=./src
OBJDIR=./obj
_HEADERS=Time.h allParams.h rmaParams.h base.h crossvalid.h \
         boosting.h lpbr.h repr.h serRMA.h parRMA.h greedyRMA.h \
 　　　　compModel.h
_SOURCES=main.cpp allParams.cpp rmaParams.cpp base.cpp crossvalid.cpp \
         boosting.cpp lpbr.cpp repr.cpp serRMA.cpp parRMA.cpp greedyRMA.cpp \
         compModel.cpp
_OBJECTS=$(_SOURCES:.cpp=.o)

SOURCES = $(patsubst %,$(SRCDIR)/%,$(_SOURCES))
HEADERS = $(patsubst %,$(SRCDIR)/%,$(_HEADERS))
OBJECTS = $(patsubst %,$(OBJDIR)/%,$(_OBJECTS))
EXECUTABLE=boosting

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CXX) $(LDFLAGS) $(OBJECTS) $(LIBSPECS) -o $@

$(OBJDIR)/%.o:  $(SRCDIR)/%.cpp $(HEDADERS) 
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

.PHONY: clean
clean:
	rm $(OBJDIR)/*.o $(EXECUTABLE)

