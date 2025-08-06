CPPSRC=ConfigFile.cpp DeviceManager.cpp main.cpp
# Sources compiled with NVCC when BUILD_CUDA=1 (``-x cu`` handles both .cu and
# .cpp files in this list)
CUSRC=PollardEngine.cpp \
      ../CudaKeySearchDevice/windowKernel.cu \
      ../CudaKeySearchDevice/CudaPollard.cu
CUOBJ=$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(CUSRC)))

.RECIPEPREFIX := ;

all:
ifeq ($(BUILD_CUDA), 1)
;# Compile CUDA kernels with NVCC.  ``-x cu`` ensures the files are
;# treated as CUDA sources even though they carry a ``.cu`` extension.
;for file in $(CUSRC) ; do \
;${NVCC} -x cu -c $$file -o $(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$$file)) ${NVCCFLAGS} \
;    -gencode=arch=compute_89,code=sm_89 ${INCLUDE} -I${CUDA_INCLUDE} -I../cudaMath ; \
;done
;${NVCC} -DBUILD_CUDA -o cuKeyFinder.bin ${CPPSRC} $(CUOBJ) \
;    ${INCLUDE} -I${CUDA_INCLUDE} ${NVCCFLAGS} \
;    -gencode=arch=compute_89,code=sm_89 \
;    ${LIBS} -L${CUDA_LIB} -lkeyfinder -laddressutil -lsecp256k1 \
;    -lcryptoutil -lsecp256k1 -lcudautil -llogger -lutil \
;    -lCudaKeySearchDevice ../CudaKeySearchDevice/cuda_libs.o \
;    -lcudadevrt -lcudart -lcmdparse
;mkdir -p $(BINDIR)
;cp cuKeyFinder.bin $(BINDIR)/cuBitCrack
endif
ifeq ($(BUILD_OPENCL),1)
;${CXX} -DBUILD_OPENCL -o clKeyFinder.bin ${CPPSRC} ${INCLUDE} -I${OPENCL_INCLUDE} ${CXXFLAGS} ${LIBS} -L${OPENCL_LIB} -lkeyfinder -laddressutil -lsecp256k1 -lcryptoutil -lsecp256k1 -lCLKeySearchDevice -lclutil -lOpenCL -llogger -lutil -lcmdparse
;mkdir -p $(BINDIR)
;cp clKeyFinder.bin $(BINDIR)/clBitCrack
endif
ifeq ($(CPU),1)
;${CXX} -o KeyFinder.bin ${CPPSRC} ${INCLUDE} ${CXXFLAGS} ${LIBS} -lkeyfinder -laddressutil -lsecp256k1 -lcryptoutil -llogger -lutil -lcmdparse -pthread
;mkdir -p $(BINDIR)
;cp KeyFinder.bin $(BINDIR)/BitCrack
endif

clean:
;rm -rf cuKeyFinder.bin $(CUOBJ)
;rm -rf clKeyFinder.bin
;rm -rf KeyFinder.bin
