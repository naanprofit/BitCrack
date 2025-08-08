#include "CLPollardDevice.h"
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include "clContext.h"
#include "clutil.h"

#include "../Logger/Logger.h"
#include "../util/util.h"

using namespace secp256k1;
using cl::clCall;

CLPollardDevice::CLPollardDevice(PollardEngine &engine,
                                 unsigned int windowBits,
                                 const std::vector<unsigned int> &offsets,
                                 const std::vector<std::array<unsigned int,5>> &targets,
                                 bool debug)
    : _engine(engine), _windowBits(windowBits), _offsets(offsets),
      _targets(targets), _debug(debug) {}

uint256 CLPollardDevice::maskBits(unsigned int bits) {
    uint256 m(0);
    for(unsigned int i = 0; i < bits; ++i) {
        m.v[i/32] |= (1u << (i % 32));
    }
    return m;
}

// Extract ``bits`` bits starting at ``offset`` from the 160-bit hash ``h``.
// The input hash is expected in little-endian word order to match the
// representation used throughout the engine and GPU kernels.
uint256 CLPollardDevice::hashWindowLE(const uint32_t h[5], uint32_t offset, uint32_t bits) {
    uint256 out(0);
    uint32_t word  = offset / 32;
    uint32_t bit   = offset % 32;
    uint32_t words = (bits + 31) / 32;
    for(uint32_t i = 0; i < words && word + i < 5; ++i) {
        uint64_t val = ((uint64_t)h[word + i]) >> bit;
        if(bit && word + i + 1 < 5) {
            val |= ((uint64_t)h[word + i + 1]) << (32 - bit);
        }
        out.v[i] = static_cast<uint32_t>(val & 0xffffffffULL);
    }
    uint32_t maskBits = bits % 32;
    if(maskBits) {
        uint32_t mask = (1u << maskBits) - 1u;
        out.v[words - 1] &= mask;
    }
    for(uint32_t i = words; i < 8; ++i) {
        out.v[i] = 0u;
    }
    return out;
}

uint256 CLPollardDevice::hashWindowBE(const uint32_t h[5], uint32_t offsetBE, uint32_t bits) {
    uint32_t offsetLE = PollardEngine::convertOffset(offsetBE, bits);
    return hashWindowLE(h, offsetLE, bits);
}

namespace {
struct TargetWindowCL {
    cl_uint targetIdx;
    cl_uint offset;
    cl_uint bits;
    cl_uint target[5];
};

struct PollardWindowCL {
    cl_uint targetIdx;
    cl_uint offset;
    cl_uint bits;
    // 256-bit scalar fragment returned by the device
    cl_uint k[8];
};

void runWalk(PollardEngine &engine,
             unsigned int windowBits,
             const std::vector<unsigned int> &offsets,
             const std::vector<std::array<unsigned int,5>> &targets,
             uint64_t steps,
             const uint256 &seed,
             const uint256 *start,
             bool wild,
             bool sequential,
             bool debug) {
    auto devices = cl::getDevices();
    if(devices.empty()) {
        return;
    }

    cl_device_id devId = devices[0].id;
    cl::CLContext ctx(devId);

    size_t local = 0;
    clGetDeviceInfo(devId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
    cl_uint computeUnits = 1;
    clGetDeviceInfo(devId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits, NULL);
    size_t global = local * computeUnits;
    engine.setStepCount(static_cast<uint64_t>(steps) * static_cast<uint64_t>(global));

    if(debug) {
        Logger::log(LogLevel::Debug,
                    std::string("CL ") + (wild ? "wild" : "tame") +
                    " walk global=" + util::format(static_cast<uint64_t>(global)) +
                    " local=" + util::format(static_cast<uint64_t>(local)) +
                    " steps=" + util::format(steps));
    }

    std::ifstream shaFile("clMath/sha256.cl");
    std::ifstream secpFile("clMath/secp256k1.cl");
    std::ifstream rmdFile("clMath/ripemd160.cl");
    std::ifstream pollardFile("CLKeySearchDevice/clPollard.cl");

    std::string sha((std::istreambuf_iterator<char>(shaFile)), std::istreambuf_iterator<char>());
    std::string secp((std::istreambuf_iterator<char>(secpFile)), std::istreambuf_iterator<char>());
    std::string rmd((std::istreambuf_iterator<char>(rmdFile)), std::istreambuf_iterator<char>());
    std::string pollard((std::istreambuf_iterator<char>(pollardFile)), std::istreambuf_iterator<char>());

    std::string src = sha + secp + rmd + pollard;

    const char *srcPtr = src.c_str();
    size_t srcLen = src.size();
    cl_int err = 0;

    cl_program program = clCreateProgramWithSource(ctx.getContext(), 1, &srcPtr, &srcLen, &err);
    clCall(err);

    err = clBuildProgram(program, 1, &devId, NULL, NULL, NULL);
    if(err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, devId, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, devId, CL_PROGRAM_BUILD_LOG, logSize, log.data(), NULL);
        std::cerr << std::string(log.begin(), log.end()) << std::endl;
        clReleaseProgram(program);
        throw cl::CLException(err);
    }

    cl_kernel kernel = clCreateKernel(program, "pollard_walk", &err);
    clCall(err);

    cl_uint maxOut = static_cast<cl_uint>(std::min<uint64_t>(1024u, steps));

    cl_mem d_out = clCreateBuffer(ctx.getContext(), CL_MEM_WRITE_ONLY, sizeof(PollardWindowCL) * maxOut, NULL, &err);
    cl_mem d_count = clCreateBuffer(ctx.getContext(), CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err);
    cl_mem d_seeds = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(cl_uint) * global * 8, NULL, &err);
    cl_mem d_starts = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(cl_uint) * global * 8, NULL, &err);
    cl_mem d_stride = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(cl_uint) * global * 8, NULL, &err);
    cl_mem d_startX = NULL;
    cl_mem d_startY = NULL;
    if(wild) {
        d_startX = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(cl_uint) * global * 8, NULL, &err);
        d_startY = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(cl_uint) * global * 8, NULL, &err);
    }


    std::vector<TargetWindowCL> windowList;
    for(size_t t = 0; t < targets.size(); ++t) {
        for(unsigned int offBE : offsets) {
            if(offBE + windowBits > 160) {
                continue;
            }
            unsigned int offLE = PollardEngine::convertOffset(offBE, windowBits);
            TargetWindowCL tw;
            tw.targetIdx = static_cast<cl_uint>(t);
            tw.offset    = offLE;
            tw.bits      = windowBits;
            uint256 hv   = CLPollardDevice::hashWindowBE(targets[t].data(), offBE, windowBits);
            hv.exportWords(tw.target, 5);
            windowList.push_back(tw);
        }
    }

    cl_uint windowCount = static_cast<cl_uint>(windowList.size());
    cl_mem d_windows = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(TargetWindowCL) * windowCount, NULL, &err);

    std::vector<cl_uint> h_seeds(global * 8);
    std::vector<cl_uint> h_starts(global * 8);
    std::vector<cl_uint> h_stride(global * 8);

    uint256 base = start ? *start : uint256(0);
    uint256 strideVal = sequential ? uint256(global) : uint256(0);
    uint256 startBase = base;
    if(wild && sequential) {
        uint256 offset = multiplyModN(strideVal, uint256(steps - 1));
        startBase = subModN(base, offset);
    }

    std::vector<cl_uint> h_startX;
    std::vector<cl_uint> h_startY;
    if(wild) {
        h_startX.resize(global * 8);
        h_startY.resize(global * 8);
    }

    ecpoint basePoint;
    if(wild && !sequential && start) {
        basePoint = multiplyPoint(*start, G());
    }
    for(size_t i = 0; i < global; ++i) {
        uint256 sSeed = seed + uint256(static_cast<uint64_t>(i));
        sSeed.exportWords(&h_seeds[i*8], 8);
        strideVal.exportWords(&h_stride[i*8], 8);
        if(wild) {
            uint256 s = sequential ? subModN(startBase, uint256(i)) : uint256(0);
            s.exportWords(&h_starts[i*8], 8);
            ecpoint p;
            if(sequential) {
                p = multiplyPoint(s, G());
            } else {
                uint256 idx((uint64_t)i);
                p = addPoints(basePoint, multiplyPoint(idx, G()));
            }
            for(int w = 0; w < 8; ++w) {
                h_startX[i * 8 + w] = p.x.v[w];
                h_startY[i * 8 + w] = p.y.v[w];
            }
        } else {
            uint256 sStart = addModN(base, uint256(i));
            sStart.exportWords(&h_starts[i*8], 8);
        }
    }

    cl_command_queue q = ctx.getQueue();
    cl_uint zero = 0;
    clEnqueueWriteBuffer(q, d_seeds, CL_TRUE, 0, sizeof(cl_uint) * global * 8, h_seeds.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(q, d_starts, CL_TRUE, 0, sizeof(cl_uint) * global * 8, h_starts.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(q, d_stride, CL_TRUE, 0, sizeof(cl_uint) * global * 8, h_stride.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(q, d_count, CL_TRUE, 0, sizeof(cl_uint), &zero, 0, NULL, NULL);
    if(windowCount > 0) {
        clEnqueueWriteBuffer(q, d_windows, CL_TRUE, 0, sizeof(TargetWindowCL) * windowCount, windowList.data(), 0, NULL, NULL);
    }
    if(wild) {
        clEnqueueWriteBuffer(q, d_startX, CL_TRUE, 0, sizeof(cl_uint) * global * 8, h_startX.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(q, d_startY, CL_TRUE, 0, sizeof(cl_uint) * global * 8, h_startY.data(), 0, NULL, NULL);
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_count);
    clSetKernelArg(kernel, 2, sizeof(cl_uint), &maxOut);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_seeds);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_starts);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_startX);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_startY);
    cl_uint stepsArg = static_cast<cl_uint>(steps);
    clSetKernelArg(kernel, 7, sizeof(cl_uint), &stepsArg);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_windows);
    clSetKernelArg(kernel, 9, sizeof(cl_uint), &windowCount);
    clSetKernelArg(kernel, 10, sizeof(cl_mem), &d_stride);

    clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, &local, 0, NULL, NULL);

    std::vector<PollardWindowCL> h_out(maxOut);
    cl_uint h_count = 0;
    clEnqueueReadBuffer(q, d_count, CL_FALSE, 0, sizeof(cl_uint), &h_count, 0, NULL, NULL);
    if(maxOut > 0) {
        clEnqueueReadBuffer(q, d_out, CL_FALSE, 0, sizeof(PollardWindowCL) * maxOut, h_out.data(), 0, NULL, NULL);
    }
    clFinish(q);

    for(cl_uint i = 0; i < h_count && i < maxOut; ++i) {
        unsigned int offsetLE = h_out[i].offset;
        unsigned int modBits  = 160u - offsetLE;
        secp256k1::uint256 rem;
        for(int j = 0; j < 8; ++j) {
            rem.v[j] = h_out[i].k[j];
        }
        if(modBits < 256) {
            unsigned int word = modBits / 32;
            unsigned int bit = modBits % 32;
            if(bit) {
                unsigned int mask = (1u << bit) - 1u;
                rem.v[word] &= mask;
                for(unsigned int k = word + 1; k < 8; ++k) {
                    rem.v[k] = 0u;
                }
            } else {
                for(unsigned int k = word; k < 8; ++k) {
                    rem.v[k] = 0u;
                }
            }
        }
        secp256k1::uint256 mod(0);
        if(modBits < 256) {
            mod.v[modBits / 32] = (1u << (modBits % 32));
        }
        PollardEngine::Constraint c{mod, rem};
        engine.processWindow(h_out[i].targetIdx, offsetLE, c);
    }

    clReleaseMemObject(d_out);
    clReleaseMemObject(d_count);
    clReleaseMemObject(d_seeds);
    clReleaseMemObject(d_starts);
    clReleaseMemObject(d_stride);
    if(d_startX) clReleaseMemObject(d_startX);
    if(d_startY) clReleaseMemObject(d_startY);
    clReleaseMemObject(d_windows);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}
} // namespace

void CLPollardDevice::startTameWalk(const uint256 &start, uint64_t steps,
                                    const uint256 &seed, bool sequential) {
    runWalk(_engine, _windowBits, _offsets, _targets, steps, seed, &start,
            false, sequential, _debug);
}

void CLPollardDevice::startWildWalk(const uint256 &start, uint64_t steps,
                                    const uint256 &seed, bool sequential) {
    runWalk(_engine, _windowBits, _offsets, _targets, steps, seed, &start,
            true, sequential, _debug);
}

extern "C" bool runCLHashWindow(const unsigned int h[5], unsigned int offset,
                                  unsigned int bits, unsigned int out[5]) {
    // Lightweight wrapper used by unit tests to validate the OpenCL window
    // extraction logic.  ``offset`` is specified from the most-significant bit
    // of the hash (big-endian).
    if(offset + bits > 160u) {
        return false;
    }
    uint256 v = CLPollardDevice::hashWindowBE(h, offset, bits);
    v.exportWords(out, 5);
    return true;
}
