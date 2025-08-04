NAME=CudaKeySearchDevice
CPPSRC:=$(wildcard *.cpp)
CUSRC:=$(wildcard *.cu)
MATHSRC:=$(CUDA_MATH)/sha256_constants.cu $(CUDA_MATH)/ripemd160_constants.cu

.RECIPEPREFIX := ;

all:    cuda

cuda:
;for file in ${CPPSRC} ; do\
;   ${NVCC} -x cu -c $$file -o $$file".o" ${NVCCFLAGS} ${INCLUDE} -I${CUDA_INCLUDE};\
;done

;for file in ${CUSRC} ; do\
;   base=$(basename $$file); \
;   ${NVCC} -c $$file -o $$base".o" ${NVCCFLAGS} ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH};\
;done

;for file in ${MATHSRC} ; do\
;   base=$(basename $$file); \
;   ${NVCC} -c $$file -o ${CUDA_MATH}/$$base".o" ${NVCCFLAGS} ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH};\
;done

;${NVCC} ${NVCCFLAGS} -dlink -o cuda_libs.o *.o ../cudaMath/sha256_constants.cu.o ../cudaMath/ripemd160_constants.cu.o -lcudadevrt -lcudart

;ar rvs ${LIBDIR}/lib$(NAME).a *.o ../cudaMath/sha256_constants.cu.o ../cudaMath/ripemd160_constants.cu.o

clean:
;rm -f *.o
;rm -f *.a

