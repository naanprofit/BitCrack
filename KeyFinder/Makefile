CPPSRC=ConfigFile.cpp DeviceManager.cpp PollardEngine.cpp main.cpp
CUSRC=../CudaKeySearchDevice/windowKernel.cu
CUOBJ=windowKernel.o

all:
ifeq ($(BUILD_CUDA), 1)
	${NVCC} -c ${CUSRC} -o ${CUOBJ} ${NVCCFLAGS} -gencode=arch=compute_89,code=sm_89 ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH}
	${NVCC} -DBUILD_CUDA -o cuKeyFinder.bin ${CPPSRC} ${CUOBJ} ${INCLUDE} -I${CUDA_INCLUDE} ${NVCCFLAGS} -gencode=arch=compute_89,code=sm_89 ${LIBS} -L${CUDA_LIB} -lkeyfinder -laddressutil -lsecp256k1 -lcryptoutil -lsecp256k1 -lcudautil -llogger -lutil -lCudaKeySearchDevice -lcudadevrt -lcudart -lcmdparse
	mkdir -p $(BINDIR)
	cp cuKeyFinder.bin $(BINDIR)/cuBitCrack
endif
ifeq ($(BUILD_OPENCL),1)
	${CXX} -DBUILD_OPENCL -o clKeyFinder.bin ${CPPSRC} ${INCLUDE} -I${OPENCL_INCLUDE} ${CXXFLAGS} ${LIBS} -L${OPENCL_LIB} -lkeyfinder -laddressutil -lsecp256k1 -lcryptoutil -lsecp256k1 -lCLKeySearchDevice -lclutil -lOpenCL -llogger -lutil -lcmdparse
	mkdir -p $(BINDIR)
	cp clKeyFinder.bin $(BINDIR)/clBitCrack
endif
ifeq ($(CPU),1)
	${CXX} -o KeyFinder.bin ${CPPSRC} ${INCLUDE} ${CXXFLAGS} ${LIBS} -lkeyfinder -laddressutil -lsecp256k1 -lcryptoutil -llogger -lutil -lcmdparse -pthread
	mkdir -p $(BINDIR)
	cp KeyFinder.bin $(BINDIR)/BitCrack
endif

clean:
	rm -rf cuKeyFinder.bin
	rm -rf clKeyFinder.bin
	rm -rf KeyFinder.bin
	rm -rf ${CUOBJ}
