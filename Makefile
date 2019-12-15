# MAKEFILE

CXX=mpicxx

RMA_DIR=/home/kagawa/Projects/thesis/code/RMA
BOOST_DIR=/home/kagawa/Projects/thesis/code/Boosting
PEBBL_DIR=/home/kagawa/Projects/thesis/code/pebbl/installpebbl
MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi

######################################### SYMBOLS ########################################
SYMBOLS=HAVE_CONFIG_H ANSI_HDRS ANSI_NAMESPACES
DEFSYMBOLS=$(patsubst %, -D%, $(SYMBOLS))

######################################## INCLUDES ##########################################
HEADERDIRS=$(RMA_DIR)/src $(BOOST_DIR)/src $(PEBBL_DIR)/include $(MPI_DIR)/include
INCLUDES=$(patsubst %,-I%,$(HEADERDIRS))

######################################### LIB ############################################
LIBDIRS=$(PEBBL_DIR)/lib $(MPI_DIR)/lib
LIBLOCATIONS=$(patsubst %,-L%,$(LIBDIRS))
LIBS=pebbl mpi mpi_cxx open-rte open-pal
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
