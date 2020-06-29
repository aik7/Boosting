# MAKEFILE

CXX=mpicxx

RMA_DIR=/home/aik/RMA
BOOST_DIR=/home/aik/Boosting
PEBBL_DIR=/home/aik/installpebbl
CLP_DIR=/home/aik/coinbrew/Clp #Projects/thesis/code/coinbrew/Clp #coin-Clp
CUTIL_DIR=/home/aik/coinbrew/CoinUtils/CoinUtils
MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi

######################################### SYMBOLS ########################################
SYMBOLS=HAVE_CONFIG_H ANSI_HDRS ANSI_NAMESPACES
DEFSYMBOLS=$(patsubst %, -D%, $(SYMBOLS))

######################################## INCLUDES ##########################################
HEADERDIRS=$(RMA_DIR)/src $(BOOST_DIR)/src $(PEBBL_DIR)/include \
           /home/aik/coinbrew/Clp/include/coin \
	   /home/aik/coinbrew/CoinUtils/CoinUtils/src \
	   /home/aik/coinbrew/Clp/Clp/src \
           /home/aik/coinbrew/build/Clp/1.17.6/src/ \
           /home/aik/coinbrew/build/CoinUtils/2.11.4/src/ \
           $(CLP_DIR)/include/coin $(CUTIL_DIR)/src $(MPI_DIR)/include
INCLUDES=$(patsubst %,-I%,$(HEADERDIRS))

######################################### LIB ############################################
LIBDIRS=$(PEBBL_DIR)/lib $(MPI_DIR)/lib $(CLP_DIR)/lib  \
        /home/aik/coinbrew/Clp/lib /home/aik/coinbrew/lib
LIBLOCATIONS=$(patsubst %,-L%,$(LIBDIRS))
LIBS=pebbl mpi mpi_cxx open-rte open-pal Clp CoinUtils
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
SRCDIRBOOST=$(BOOST_DIR)/src
OBJDIRRMA=$(RMA_DIR)/obj
OBJDIRBOOST=$(BOOST_DIR)/src
_HEADERSRMA=Time.h argRMA.h dataRMA.h serRMA.h parRMA.h greedyRMA.h baseRMA.h #driverRMA.h
_HEADERSBOOST=argBoost.h dataBoost.h baseBoost.h boosting.h repr.h # lpbr.h
_SOURCESRMA=argRMA.cpp dataRMA.cpp serRMA.cpp parRMA.cpp greedyRMA.cpp baseRMA.cpp #driverRMA.cpp  #
_SOURCESBOOST=driver.cpp argBoost.cpp dataBoost.cpp boosting.cpp repr.cpp #lpbr.cpp baseBoost.cpp

_OBJECTSRMA=$(_SOURCESRMA:.cpp=.o)
_OBJECTSBOOST=$(_SOURCESBOOST:.cpp=.o)

SOURCES = $(patsubst %, $(SRCDIRRMA)/%, $(_SOURCESRMA))  $(patsubst %, $(SRCDIRBOOST)/%, $(_SOURCESBOOST))
HEADERS = $(patsubst %, $(SRCDIRRMA)/%, $(_HEADERSRMA))  $(patsubst %, $(SRCDIRBOOST)/%, $(_HEADERSBOOST))
OBJECTS = $(patsubst %, $(OBJDIRRMA)/%, $(_OBJECTSRMA))  $(patsubst %, $(OBJDIRBOOST)/%, $(_OBJECTSBOOST))

print-%:
	@echo '$(SOURCES)'

EXECUTABLE=boosting

all: $(SOURCES) $(EXECUTABLE)
	@echo $(Sources)

$(EXECUTABLE): $(OBJECTS)
	@echo $(SOURCES)
	$(CXX) $(LDFLAGS) $(OBJECTS) $(LIBSPECS) -o $@

$(OBJDIR)/%.o:  $(SRCDIRRMA)/%.cpp $(SRCDIRBOOST)/%.cpp $(HEADERDIRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm $(OBJDIRBOOST)/*.o $(EXECUTABLE) #$(OBJDIRRMA)/*.o
