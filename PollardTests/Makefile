NAME=PollardTests
BUILD_CUDA?=0
BUILD_OPENCL?=0
CPPSRC=main.cpp
ifeq ($(BUILD_CUDA),0)
CPPSRC+=../KeyFinder/PollardEngine.cpp
endif
CUDASRC=cuda_scalar_one.cu
BINDIR?=.

BUILD_CUDA:=$(if $(BUILD_CUDA),$(BUILD_CUDA),0)
BUILD_OPENCL:=$(if $(BUILD_OPENCL),$(BUILD_OPENCL),0)

.RECIPEPREFIX := ;

OBJS=
ifeq ($(BUILD_CUDA),1)
OBJS+=cuda_scalar_one.o PollardEngine.o
endif

CXXFLAGS+=-DBUILD_CUDA=$(BUILD_CUDA) -DBUILD_OPENCL=$(BUILD_OPENCL)

LIBS_LOCAL=-laddressutil -lsecp256k1 -lcryptoutil -lutil -llogger -pthread
ifeq ($(BUILD_OPENCL),1)
LIBS_LOCAL+=-lclutil -lutil -lOpenCL -L${OPENCL_LIB} -lCLKeySearchDevice
endif
ifeq ($(BUILD_CUDA),1)
LIBS_LOCAL+=-lcudart -lcudadevrt -L${CUDA_LIB} -lCudaKeySearchDevice -lkeyfinder
endif

all: $(OBJS)
ifeq ($(BUILD_CUDA),1)
;${NVCC} ${NVCCFLAGS} -DBUILD_CUDA=1 -DBUILD_OPENCL=$(BUILD_OPENCL) ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH} -o pollardtests.bin ${CPPSRC} $(OBJS) ${LIBS} ${LIBS_LOCAL}
else
;${CXX} -o pollardtests.bin ${CPPSRC} $(OBJS) ${INCLUDE} ${CXXFLAGS} ${LIBS} ${LIBS_LOCAL}
endif
;mkdir -p $(BINDIR)
;cp pollardtests.bin $(BINDIR)/pollardtests

cuda_scalar_one.o: cuda_scalar_one.cu
;${NVCC} -c cuda_scalar_one.cu -o cuda_scalar_one.o ${NVCCFLAGS} \
    -gencode=arch=compute_89,code=sm_89 ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH}

PollardEngine.o: ../KeyFinder/PollardEngine.cpp
;${NVCC} -c ../KeyFinder/PollardEngine.cpp -o PollardEngine.o ${NVCCFLAGS} -DBUILD_CUDA=1 -DBUILD_OPENCL=$(BUILD_OPENCL) ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH}

clean:
;rm -f pollardtests.bin cuda_scalar_one.o PollardEngine.o
