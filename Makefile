LIBRARY_DIR=$(HOME)/work/itensor
include $(LIBRARY_DIR)/this_dir.mk
include $(LIBRARY_DIR)/options.mk


################################################################
#Options --------------

HEADERS=

ifdef app
APP=$(app)
else
#APP=triangular_metts
APP=mpo_ancilla_hub
endif

TENSOR_HEADERS=

#################################################################

OBJECTS=$(APP).o

#Mappings --------------
REL_TENSOR_HEADERS=$(patsubst %,$(ITENSOR_INCLUDEDIR)/%, $(TENSOR_HEADERS))
GOBJECTS=$(patsubst %,.debug_objs/%, $(OBJECTS))

#Define Flags ----------
CCFLAGS=-I. $(ITENSOR_INCLUDEFLAGS) $(OPTIMIZATIONS) -Wno-unused-variable
CCGFLAGS=-I. $(ITENSOR_INCLUDEFLAGS) $(DEBUGFLAGS)
LIBFLAGS=-L$(ITENSOR_LIBDIR) $(ITENSOR_LIBFLAGS)
LIBGFLAGS=-L$(ITENSOR_LIBDIR) $(ITENSOR_LIBGFLAGS)

#Rules ------------------

%.o: %.cc $(HEADERS) $(REL_TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cc $(HEADERS) $(REL_TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

build: $(APP)
debug: $(APP)-g

$(APP): $(OBJECTS) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS) -o $(APP) $(LIBFLAGS)

$(APP)-g: mkdebugdir $(GOBJECTS) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS) -o $(APP)-g $(LIBGFLAGS)

mkdebugdir:
	mkdir -p .debug_objs

democlean:
	rm -f *density*

clean:
	rm -fr *.o .debug_objs $(APP) $(APP)-g
