NAME=CudaKeySearchDevice
CPPSRC:=$(wildcard *.cpp)
CUSRC:=$(wildcard *.cu)
MATHSRC:=$(CUDA_MATH)/sha256_constants.cu $(CUDA_MATH)/ripemd160_constants.cu

CPPOBJS:=$(addsuffix .o,$(CPPSRC))
CUOBJS:=$(addsuffix .o,$(CUSRC))
OBJS:=$(CPPOBJS) $(CUOBJS)

.RECIPEPREFIX := ;

all:    cuda

cuda:
;rm -f *.o
;rm -f ${CUDA_MATH}/sha256_constants.cu.o ${CUDA_MATH}/ripemd160_constants.cu.o cuda_libs.o
;for file in ${CPPSRC} ; do\
;   base=$(basename $$file); \
;   ${NVCC} -x cu -c $$file -o $${base}.o ${NVCCFLAGS} -gencode=arch=compute_89,code=sm_89 ${INCLUDE} -I${CUDA_INCLUDE};\
;done
;for file in ${CUSRC} ; do\
;   base=$(basename $$file .cu); \
;   ${NVCC} -c $$file -o $${base}.o ${NVCCFLAGS} -gencode=arch=compute_89,code=sm_89 ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH};\
;done
;for file in ${MATHSRC} ; do\
;   file_name=$${file##*/}; \
;   ${NVCC} -c $$file -o ${CUDA_MATH}/$${file_name%.cu}.cu.o ${NVCCFLAGS} -gencode=arch=compute_89,code=sm_89 ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH};\
;done
;${NVCC} ${NVCCFLAGS} -gencode=arch=compute_89,code=sm_89 -dlink -o cuda_libs.o ${OBJS} ../cudaMath/sha256_constants.cu.o ../cudaMath/ripemd160_constants.cu.o -lcudadevrt -lcudart
;ar rvs ${LIBDIR}/lib$(NAME).a ${OBJS} ../cudaMath/sha256_constants.cu.o ../cudaMath/ripemd160_constants.cu.o

clean:
;rm -f *.o
;rm -f *.a
;rm -f cuda_libs.o
;rm -f ${CUDA_MATH}/sha256_constants.cu.o ${CUDA_MATH}/ripemd160_constants.cu.o

