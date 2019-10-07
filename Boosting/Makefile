# MAKEFILE
CXX=mpicxx

# This should be altered to point to wherever your acro-pebbl directory actually lives
BOOSTING=/home/user1/Projects/boosting/Boosting
ACRO_ROOT=/home/user1/pebbl
MPI_ROOT=/opt/openmpi/1.10.1
#R_HOME=/home/user1/R
#Rcpp=/home/user1/R/library/Rcpp
#RInside=/home/user1/R/library/RInside

######################################### SYMBOLS ########################################
SYMBOLS=HAVE_CONFIG_H ANSI_HDRS ANSI_NAMESPACES
DEFSYMBOLS=$(patsubst %,-D%,$(SYMBOLS))

######################################## INCLUDES ##########################################
HEADERDIRS=$(BOOSTING) $(ACRO_ROOT)/src/pebbl $(ACRO_ROOT)/build/src $(ACRO_ROOT)/src
           #$(ACRO_ROOT)/include/pebbl $(ACRO_ROOT)/include/utilib \
           #$(R_HOME)/include $(Rcpp)/include $(RInside)/include #$(MPI_ROOT)/include
INCLUDES=$(patsubst %,-I%,$(HEADERDIRS))

######################################### LIB ############################################
LIBDIRS=$(ACRO_ROOT)/build/src/pebbl #$(R_HOME)/lib $(Rcpp)/libs $(RInside)/lib $(RInside)/libs #$(MPI_ROOT)/lib
LIBLOCATIONS=$(patsubst %,-L%,$(LIBDIRS))
LIBS=pebbl #utilib #Rblas Rlapack R RInside #mpi #mpi_cxx open-rte open-pal
LIBSPECS=$(patsubst %,-l%,$(LIBS))

########################################## FLAGS ##########################################
DEBUGFLAGS=-g -fpermissive -O0 #-D_GLIBCXX_USE_CXX11_ABI=1 #-O0
MISCCXXFLAGS= -std=c++98

# include
CXXFLAGS=$(DEFSYMBOLS) $(INCLUDES) $(MISCCXXFLAGS) $(DEBUGFLAGS)
# Library
LDFLAGS=$(DEBUGFLAGS) $(LIBLOCATIONS)

#####################################################################################

HDRDIR=$(BOOSTING)/include
SRCDIR=$(BOOSTING)/src
OBJDIR=$(BOOSTING)/obj
_HEADERS=Time.h allParams.h rmaParams.h base.h crossvalid.h compModel.h \
         boosting.h lpbr.h repr.h serRMA.h parRMA.h greedyRMA.h 
_SOURCES=driver.cpp allParams.cpp rmaParams.cpp base.cpp crossvalid.cpp compModel.cpp \
         boosting.cpp lpbr.cpp repr.cpp serRMA.cpp parRMA.cpp greedyRMA.cpp          
_OBJECTS=$(_SOURCES:.cpp=.o)

SOURCES = $(patsubst %,$(SRCDIR)/%,$(_SOURCES))
HEADERS = $(patsubst %,$(HDRDIR)/%,$(_HEADERS))
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
