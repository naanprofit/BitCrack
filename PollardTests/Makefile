NAME=PollardTests
CPPSRC=main.cpp ../KeyFinder/PollardEngine.cpp
ifeq ($(BUILD_OPENCL),1)
CPPSRC+= ../CLKeySearchDevice/CLPollardDevice.cpp
endif

CUDASRC=cuda_scalar_one.cu
BINDIR?=.

BUILD_CUDA?=0
BUILD_OPENCL?=0

# Header include paths
INCLUDE?=-I.. -I../util -I../AddressUtil -I../CryptoUtil -I../KeyFinderLib -I../KeyFinder -I../secp256k1lib -I../Logger
ifeq ($(BUILD_CUDA),1)
INCLUDE+=-I../cudaUtil -I../CudaKeySearchDevice -I../cudaMath
endif
ifeq ($(BUILD_OPENCL),1)
INCLUDE+=-I../clUtil -I../CLKeySearchDevice -I../embedcl
endif

.RECIPEPREFIX := ;

OBJS=
ifeq ($(BUILD_CUDA),1)
OBJS+=cuda_scalar_one.o ../CudaKeySearchDevice/windowKernel.o
endif

CXXFLAGS+=$(if $(filter 1,$(BUILD_CUDA)),-DBUILD_CUDA=1) $(if $(filter 1,$(BUILD_OPENCL)),-DBUILD_OPENCL=1)
NVCCFLAGS+=$(if $(filter 1,$(BUILD_CUDA)),-DBUILD_CUDA=1) $(if $(filter 1,$(BUILD_OPENCL)),-DBUILD_OPENCL=1)

LIBS_LOCAL=-laddressutil -lsecp256k1 -lcryptoutil -lutil -llogger
ifeq ($(BUILD_OPENCL),1)
LIBS_LOCAL+=-lclutil -lutil -lOpenCL -L${OPENCL_LIB} -lCLKeySearchDevice
endif
ifeq ($(BUILD_CUDA),1)
LIBS_LOCAL+=-lcudautil -L${CUDA_LIB} -lCudaKeySearchDevice -lcudadevrt -lcudart
endif

all: $(OBJS)
ifeq ($(BUILD_CUDA),1)
;${NVCC} -o pollardtests.bin ${CPPSRC} $(OBJS) ${INCLUDE} -I${CUDA_INCLUDE} $(if $(BUILD_OPENCL),-I${OPENCL_INCLUDE}) ${NVCCFLAGS} ${LIBS} ${LIBS_LOCAL}
else
;${CXX} -o pollardtests.bin ${CPPSRC} $(OBJS) ${INCLUDE} -I${CUDA_INCLUDE} $(if $(BUILD_OPENCL),-I${OPENCL_INCLUDE}) ${CXXFLAGS} ${LIBS} ${LIBS_LOCAL}
endif
;mkdir -p $(BINDIR)
;cp pollardtests.bin $(BINDIR)/pollardtests

pollard-tests: all
;$(BINDIR)/pollardtests

cuda_scalar_one.o: cuda_scalar_one.cu
;${NVCC} -c cuda_scalar_one.cu -o cuda_scalar_one.o ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH} ${NVCCFLAGS}

clean:
;rm -f pollardtests.bin cuda_scalar_one.o
