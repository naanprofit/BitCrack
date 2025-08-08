CPPSRC=ConfigFile.cpp DeviceManager.cpp PollardEngine.cpp main.cpp

all:
ifeq ($(BUILD_CUDA), 1)
	${NVCC} -DBUILD_CUDA -std=c++11 -rdc=true -o cuKeyFinder.bin ${CPPSRC} ../CudaKeySearchDevice/windowKernel.o \
	${INCLUDE} -I${CUDA_INCLUDE} ${NVCCFLAGS} ${LIBS} -L${CUDA_LIB} -L${LIBDIR} -L../CudaKeySearchDevice \
	-lCudaKeySearchDevice -lkeyfinder -laddressutil -lsecp256k1 -lcryptoutil -lcudautil -llogger -lutil \
	-lcudart -lcudadevrt -lcmdparse
		mkdir -p $(BINDIR)
		cp cuKeyFinder.bin $(BINDIR)/cuBitCrack
endif
ifeq ($(BUILD_OPENCL),1)
		${CXX} -DBUILD_OPENCL -o clKeyFinder.bin ${CPPSRC} ${INCLUDE} -I${OPENCL_INCLUDE} ${CXXFLAGS} -L${LIBDIR} -L${OPENCL_LIB} -lkeyfinder -laddressutil -lsecp256k1 -lcryptoutil -lsecp256k1 -lCLKeySearchDevice -lclutil -lOpenCL -llogger -lutil -lcmdparse
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
