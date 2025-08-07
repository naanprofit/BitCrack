NAME=CudaKeySearchDevice
CPPSRC:=$(wildcard *.cpp)
CUSRC:=$(wildcard *.cu)
MATHSRC:=$(CUDA_MATH)/sha256_constants.cu $(CUDA_MATH)/ripemd160_constants.cu

.RECIPEPREFIX := ;

all:    cuda

cuda:
;rm -f *.o
;rm -f ${CUDA_MATH}/sha256_constants.cu.o ${CUDA_MATH}/ripemd160_constants.cu.o cuda_libs.o
;for file in ${CPPSRC} ; do\
;   base=$${file%.*}; \
;   ${NVCC} -x cu -c $$file -o "$${base}.cpp.o" ${NVCCFLAGS} -gencode=arch=compute_89,code=sm_89 ${INCLUDE} -I${CUDA_INCLUDE};\
;done

;for file in ${CUSRC} ; do\
;   base=$${file%.*}; \
;   ${NVCC} -c $$file -o "$${base}.cu.o" ${NVCCFLAGS} -gencode=arch=compute_89,code=sm_89 ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH};\
;done

;for file in ${MATHSRC} ; do\
;   file_name=$${file##*/}; \
;   ${NVCC} -c $$file -o ${CUDA_MATH}/$${file_name%.cu}.cu.o ${NVCCFLAGS} -gencode=arch=compute_89,code=sm_89 ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH};\
;done

;${NVCC} ${NVCCFLAGS} -gencode=arch=compute_89,code=sm_89 -dlink -o cuda_libs.o *.cpp.o *.cu.o ../cudaMath/sha256_constants.cu.o ../cudaMath/ripemd160_constants.cu.o -lcudadevrt -lcudart

;ar rvs ${LIBDIR}/lib$(NAME).a *.cpp.o *.cu.o ../cudaMath/sha256_constants.cu.o ../cudaMath/ripemd160_constants.cu.o

clean:
;rm -f *.o
;rm -f *.a
;rm -f cuda_libs.o
;rm -f ${CUDA_MATH}/sha256_constants.cu.o ${CUDA_MATH}/ripemd160_constants.cu.o

