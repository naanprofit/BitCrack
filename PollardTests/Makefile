NAME=PollardTests
CPPSRC=main.cpp ../KeyFinder/PollardEngine.cpp
ifeq ($(BUILD_CUDA),1)
CPPSRC+= ../CudaKeySearchDevice/CudaPollardDevice.cpp
endif
CUDASRC=cuda_scalar_one.cu
BINDIR?=.

BUILD_CUDA?=0
BUILD_OPENCL?=0

.RECIPEPREFIX := ;

OBJS=
ifeq ($(BUILD_CUDA),1)
OBJS+=cuda_scalar_one.o
endif

CXXFLAGS+=$(if $(BUILD_CUDA),-DBUILD_CUDA=1) $(if $(BUILD_OPENCL),-DBUILD_OPENCL=1)
NVCCFLAGS+=$(if $(BUILD_CUDA),-DBUILD_CUDA=1) $(if $(BUILD_OPENCL),-DBUILD_OPENCL=1)

LIBS_LOCAL=-laddressutil -lsecp256k1 -lcryptoutil -lutil -llogger
ifeq ($(BUILD_OPENCL),1)
LIBS_LOCAL+=-lclutil -lutil -lOpenCL -L${OPENCL_LIB} -lCLKeySearchDevice
endif
ifeq ($(BUILD_CUDA),1)
LIBS_LOCAL+=-L${CUDA_LIB} -lCudaKeySearchDevice -lcudadevrt -lcudart -lkeyfinder
endif

all: $(OBJS)
ifeq ($(BUILD_CUDA),1)
;${NVCC} -x cu -o pollardtests.bin ${CPPSRC} $(OBJS) ${INCLUDE} -I${CUDA_INCLUDE} ${NVCCFLAGS} ${LIBS} ${LIBS_LOCAL}
else
;${CXX} -o pollardtests.bin ${CPPSRC} $(OBJS) ${INCLUDE} ${CXXFLAGS} ${LIBS} ${LIBS_LOCAL}
endif
;mkdir -p $(BINDIR)
;cp pollardtests.bin $(BINDIR)/pollardtests

pollard-tests: all
;$(BINDIR)/pollardtests

cuda_scalar_one.o: cuda_scalar_one.cu
;${NVCC} -c cuda_scalar_one.cu -o cuda_scalar_one.o ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH} ${NVCCFLAGS}

clean:
;rm -f pollardtests.bin cuda_scalar_one.o
