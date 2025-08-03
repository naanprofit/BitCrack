#ifndef SIMPLE_RING_BUFFER_H
#define SIMPLE_RING_BUFFER_H
#include <vector>

template<typename T>
class SimpleRingBuffer {
    std::vector<T> _buf;
    size_t _head = 0;
    size_t _tail = 0;
    size_t _count = 0;
public:
    explicit SimpleRingBuffer(size_t capacity = 1024) : _buf(capacity) {}
    bool push(const T &v) {
        if(_count == _buf.size()) return false;
        _buf[_tail] = v;
        _tail = (_tail + 1) % _buf.size();
        ++_count;
        return true;
    }
    bool pop(T &out) {
        if(_count == 0) return false;
        out = _buf[_head];
        _head = (_head + 1) % _buf.size();
        --_count;
        return true;
    }
    void clear() { _head = _tail = _count = 0; }
    bool empty() const { return _count == 0; }
};

#endif
