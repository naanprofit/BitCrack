#include "CLPollardDevice.h"
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include "clContext.h"
#include "clutil.h"

using namespace secp256k1;

CLPollardDevice::CLPollardDevice(PollardEngine &engine,
                                 unsigned int windowBits,
                                 const std::vector<unsigned int> &offsets,
                                 const std::vector<std::array<unsigned int,5>> &targets)
    : _engine(engine), _windowBits(windowBits), _offsets(offsets), _targets(targets) {}

uint256 CLPollardDevice::maskBits(unsigned int bits) {
    uint256 m(0);
    for(unsigned int i = 0; i < bits; ++i) {
        m.v[i/32] |= (1u << (i % 32));
    }
    return m;
}

uint64_t CLPollardDevice::hashWindowLE(const unsigned int h[5], unsigned int offset, unsigned int bits) {
    unsigned int word = offset / 32;
    unsigned int bit = offset % 32;
    uint64_t val = 0;
    if(word < 5) {
        val = ((uint64_t)h[word]) >> bit;
        if(bit + bits > 32 && word + 1 < 5) {
            val |= ((uint64_t)h[word + 1]) << (32 - bit);
        }
    }
    if(bit + bits > 64 && word + 2 < 5) {
        val |= ((uint64_t)h[word + 2]) << (64 - bit);
    }
    if(bits < 64) {
        uint64_t mask = (bits == 64) ? 0xffffffffffffffffULL : ((1ULL << bits) - 1ULL);
        val &= mask;
    }
    return val;
}

namespace {
struct TargetWindowCL {
    cl_uint targetIdx;
    cl_uint offset;
    cl_uint bits;
    cl_ulong target;
};

struct PollardWindowCL {
    cl_uint targetIdx;
    cl_uint offset;
    cl_uint bits;
    cl_uint k[8];
};

void runWalk(PollardEngine &engine,
             unsigned int windowBits,
             const std::vector<unsigned int> &offsets,
             const std::vector<std::array<unsigned int,5>> &targets,
             uint64_t steps,
             uint64_t seed,
             const uint256 *start,
             const ecpoint *startPoint) {
    auto devices = cl::getDevices();
    if(devices.empty()) {
        return;
    }

    cl::CLContext ctx(devices[0].id);

    std::ifstream shaFile("clMath/sha256.cl");
    std::ifstream secpFile("clMath/secp256k1.cl");
    std::ifstream rmdFile("clMath/ripemd160.cl");
    std::ifstream pollardFile("CLKeySearchDevice/clPollard.cl");

    std::string sha((std::istreambuf_iterator<char>(shaFile)), std::istreambuf_iterator<char>());
    std::string secp((std::istreambuf_iterator<char>(secpFile)), std::istreambuf_iterator<char>());
    std::string rmd((std::istreambuf_iterator<char>(rmdFile)), std::istreambuf_iterator<char>());
    std::string pollard((std::istreambuf_iterator<char>(pollardFile)), std::istreambuf_iterator<char>());

    std::string src = sha + secp + rmd + pollard;
    cl::CLProgram prog(ctx, src.c_str());
    cl_program program = prog.getProgram();

    cl_int err = 0;
    cl_kernel kernel = clCreateKernel(program, "pollard_random_walk", &err);

    size_t global = 1;
    cl_uint maxOut = static_cast<cl_uint>(steps * global);

    cl_mem d_out = clCreateBuffer(ctx.getContext(), CL_MEM_WRITE_ONLY, sizeof(PollardWindowCL) * maxOut, NULL, &err);
    cl_mem d_count = clCreateBuffer(ctx.getContext(), CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err);
    cl_mem d_seeds = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(cl_ulong) * global, NULL, &err);
    cl_mem d_starts = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(cl_ulong) * global, NULL, &err);
    cl_mem d_startX = NULL;
    cl_mem d_startY = NULL;
    if(startPoint) {
        d_startX = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(cl_uint) * global * 8, NULL, &err);
        d_startY = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(cl_uint) * global * 8, NULL, &err);
    }

    std::vector<TargetWindowCL> windowList;
    for(size_t t = 0; t < targets.size(); ++t) {
        for(unsigned int off : offsets) {
            if(off + windowBits > 160) {
                continue;
            }
            TargetWindowCL tw;
            tw.targetIdx = static_cast<cl_uint>(t);
            tw.offset = off;
            tw.bits = windowBits;
            tw.target = CLPollardDevice::hashWindowLE(targets[t].data(), off, windowBits);
            windowList.push_back(tw);
        }
    }

    cl_uint windowCount = static_cast<cl_uint>(windowList.size());
    cl_mem d_windows = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, sizeof(TargetWindowCL) * windowCount, NULL, &err);

    std::vector<cl_ulong> h_seeds(global);
    std::vector<cl_ulong> h_starts(global);
    uint64_t base = 0ULL;
    if(start) {
        base = ((uint64_t)start->v[1] << 32) | start->v[0];
    }
    for(size_t i = 0; i < global; ++i) {
        h_seeds[i] = seed + i;
        h_starts[i] = start ? (base + i) : 0ULL;
    }

    std::vector<cl_uint> h_startX;
    std::vector<cl_uint> h_startY;
    if(startPoint) {
        h_startX.resize(global * 8);
        h_startY.resize(global * 8);
        for(size_t i = 0; i < global; ++i) {
            uint256 idx((uint64_t)i);
            ecpoint p = addPoints(*startPoint, multiplyPoint(idx, G()));
            for(int w = 0; w < 8; ++w) {
                h_startX[i * 8 + w] = p.x.v[w];
                h_startY[i * 8 + w] = p.y.v[w];
            }
        }
    }

    cl_command_queue q = ctx.getQueue();
    cl_uint zero = 0;
    clEnqueueWriteBuffer(q, d_seeds, CL_TRUE, 0, sizeof(cl_ulong) * global, h_seeds.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(q, d_starts, CL_TRUE, 0, sizeof(cl_ulong) * global, h_starts.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(q, d_count, CL_TRUE, 0, sizeof(cl_uint), &zero, 0, NULL, NULL);
    if(windowCount > 0) {
        clEnqueueWriteBuffer(q, d_windows, CL_TRUE, 0, sizeof(TargetWindowCL) * windowCount, windowList.data(), 0, NULL, NULL);
    }
    if(startPoint) {
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

    clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);

    std::vector<PollardWindowCL> h_out(maxOut);
    cl_uint h_count = 0;
    clEnqueueReadBuffer(q, d_count, CL_FALSE, 0, sizeof(cl_uint), &h_count, 0, NULL, NULL);
    if(maxOut > 0) {
        clEnqueueReadBuffer(q, d_out, CL_FALSE, 0, sizeof(PollardWindowCL) * maxOut, h_out.data(), 0, NULL, NULL);
    }
    clFinish(q);

    for(cl_uint i = 0; i < h_count && i < maxOut; ++i) {
        PollardWindow w;
        w.targetIdx = h_out[i].targetIdx;
        w.offset = h_out[i].offset;
        w.bits = h_out[i].bits;
        for(int j = 0; j < 8; ++j) {
            w.scalarFragment.v[j] = h_out[i].k[j];
        }
        engine.processWindow(w);
    }

    clReleaseMemObject(d_out);
    clReleaseMemObject(d_count);
    clReleaseMemObject(d_seeds);
    clReleaseMemObject(d_windows);
    clReleaseMemObject(d_starts);
    if(d_startX) clReleaseMemObject(d_startX);
    if(d_startY) clReleaseMemObject(d_startY);
    clReleaseKernel(kernel);
}
} // namespace

void CLPollardDevice::startTameWalk(const uint256 &start, uint64_t steps, uint64_t seed) {
    runWalk(_engine, _windowBits, _offsets, _targets, steps, seed, &start, nullptr);
}

void CLPollardDevice::startWildWalk(const ecpoint &start, uint64_t steps, uint64_t seed) {
    runWalk(_engine, _windowBits, _offsets, _targets, steps, seed, nullptr, &start);
}
