# MAKEFILE

CXX=mpicxx

RMA_DIR=/home/kagawa/Projects/thesis/rma/RMA
BOOST_DIR=/home/kagawa/Projects/thesis/boosting/Boosting
PEBBL_DIR=/home/kagawa/Projects/thesis/pebbl
#MPI_ROOT=/opt/openmpi/1.10.1

######################################### SYMBOLS ########################################
SYMBOLS=HAVE_CONFIG_H ANSI_HDRS ANSI_NAMESPACES
DEFSYMBOLS=$(patsubst %, -D%, $(SYMBOLS))

######################################## INCLUDES ##########################################
HEADERDIRS=$(RMA_DIR)/src $(BOOST_DIR)/src $(PEBBL_DIR)/src $(PEBBL_DIR)/build/src #$(MPI_ROOT)/include
INCLUDES=$(patsubst %,-I%,$(HEADERDIRS))

######################################### LIB ############################################
LIBDIRS=$(PEBBL_DIR)/build/src/pebbl #$(MPI_ROOT)/lib
LIBLOCATIONS=$(patsubst %,-L%,$(LIBDIRS))
LIBS=pebbl #mpi mpi_cxx open-rte open-pal
LIBSPECS=$(patsubst %,-l%,$(LIBS))

########################################## FLAGS ##########################################
DEBUGFLAGS=-g -fpermissive -O0
MISCCXXFLAGS= -std=c++11  #98

# include
CXXFLAGS=$(DEFSYMBOLS) $(INCLUDES) $(MISCCXXFLAGS) $(DEBUGFLAGS)
# Library
LDFLAGS=$(DEBUGFLAGS) $(LIBLOCATIONS)

#####################################################################################

SRCDIRRMA=$(RMA_DIR)/src
SRCDIRBOOST=./src
OBJDIR=./obj
_HEADERSRMA=Time.h argRMA.h dataRMA.h serRMA.h parRMA.h greedyRMA.h
_HEADERSBOOST=argBoost.h baseBoost.h boosting.h lpbr.h repr.h
_SOURCESRMA=argRMA.cpp dataRMA.cpp serRMA.cpp parRMA.cpp greedyRMA.cpp
_SOURCESBOOST=argBoost.cpp baseBoost.cpp boosting.cpp lpbr.cpp repr.cpp driver.cpp

_OBJECTS=$(_SOURCESRMA:.cpp=.o) $(_SOURCESBOOST:.cpp=.o)

SOURCES = $(patsubst %, $(SRCDIRRMA)/%, $(_SOURCESRMA)) $(patsubst %, $(SRCDIRBOOST)/%, $(_SOURCESBOOST))
HEADERS = $(patsubst %, $(SRCDIRRMA)/%, $(_HEADERSRMA)) $(patsubst %, $(SRCDIRBOOST)/%, $(_HEADERSBOOST))
OBJECTS = $(patsubst %, $(OBJDIR)/%, $(_OBJECTS))

EXECUTABLE=boosting

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) $(LIBSPECS) -o $@

$(OBJDIR)/%.o:  $(SRCDIRRMA)/%.cpp $(SRCDIRBOOST)/%.cpp $(HEADERDIRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm $(OBJDIR)/*.o $(EXECUTABLE)
