EXTRA_DIST = sfopKernels.cl

# if compiling for GPU, set define
if GPUVERSION
AM_CPPFLAGS = -DGPU
endif

# sfop library
lib_LTLIBRARIES = libsfop.la
libsfop_la_LDFLAGS = -no-undefined -version-info 9:0:9 

# library sources
if GPUVERSION
libsfop_la_SOURCES = CImageCpu.cpp CLayer.cpp COctave.cpp CSfop.cpp CImageCl.cpp COpenCl.cpp
else
libsfop_la_SOURCES = CImageCpu.cpp CLayer.cpp COctave.cpp CSfop.cpp
endif

# binaries
if TESTPROGRAM
bin_PROGRAMS = sfop sfop_test
else
bin_PROGRAMS = sfop 
endif

# sources for main and test program
sfop_SOURCES = main.cpp
if GPUVERSION
sfop_test_SOURCES = test.cpp CSfopTest.cpp CLayerTest.cpp CImageTest.cpp CImageCpuTest.cpp CImageClTest.cpp
else
sfop_test_SOURCES = test.cpp CSfopTest.cpp CLayerTest.cpp CImageTest.cpp CImageCpuTest.cpp
endif

# linker flags for main and test program
sfop_LDFLAGS = -lsfop
sfop_test_LDFLAGS = -lsfop

# headers to be installed
if GPUVERSION
include_HEADERS = CSfop.h CFeature.h COctave.h CLayer.h CImageFactory.h CImg.h CImage.h CImageCpu.h CImageCl.h COpenCl.h
else
include_HEADERS = CSfop.h CFeature.h COctave.h CLayer.h CImageFactory.h CImg.h CImage.h CImageCpu.h
endif

