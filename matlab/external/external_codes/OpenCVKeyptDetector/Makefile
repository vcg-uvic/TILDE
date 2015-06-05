UNAME=$(shell uname)
ifeq ($(UNAME),Linux)
CC=g++
CFLAGS = -I/usr/include/ `pkg-config opencv --cflags` -O2 -std=c++11
LDFLAGS = `pkg-config opencv --libs` -lopencv_core -lopencv_flann -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_legacy -lopencv_contrib -lopencv_nonfree
else
CC=g++
CFLAGS = `pkg-config opencv --cflags` -std=c++11
LDFLAGS = `pkg-config opencv --libs`
endif

CXXSOURCES=$(wildcard opencvKeypointDetector.cpp)
CXXOBJECTS=$(CXXSOURCES:.cpp=.o)
EXECUTABLE=opencvKeypointDetector

all: release

debug: CFLAGS += -g -O0 #-DVERBOSE_OPT
debug: $(EXECUTABLE)

release: CFLAGS += -O2 -DNDEBUG
release: $(EXECUTABLE)

$(EXECUTABLE): $(CXXSOURCES) $(CXXOBJECTS) $(CSOURCES) $(COBJECTS)
	# make -C src/liblinear-1.94 lib
	$(CC) $(COBJECTS) $(CXXOBJECTS) $(CFLAGS) $(LDFLAGS) -o $@ 

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -rf $(COBJECTS) $(CXXOBJECTS) $(EXECUTABLE)
