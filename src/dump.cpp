/* ===================================================
 * Copyright (C) 2018 speed-clouds All Right Reserved.
 *      Author: chenshuangping@speed-clouds.com
 *    Filename: 1.cpp
 *     Created: 2018-04-16 15:48
 * Description:
 * ===================================================
 */
#include <stdlib.h>

#include "kx/kx_file.h"
#include "kx/CLI11.hpp"

using namespace kx;

static inline void le_print(const char *p, int size) {
    for (int i=size-1; i>=0; i--)
        printf("%02x", p[i] & 0xff);
}

static inline void be_print(const char *p, int size) {
    for (int i=0; i<size; i++)
        printf("%02x", p[i] & 0xff);
}

static inline const char *next_number(int stride, int num_bytes, const char *p, int index) {
    return p + index * num_bytes;
}

static inline const char *next_number_r(int stride, int num_bytes, const char *p, int index) {
    return p + (stride - 1 - index) * num_bytes;
}

class dumper {
public:
    dumper(int num_bytes, int stride, int reverse_stride = false, int reverse_number = false, int no_space = false):
        num_bytes_(num_bytes), stride_(stride),
        reverse_stride_(reverse_stride),
        reverse_number_(reverse_number),
        no_space_(no_space)
    {}

    void dump(const void *data, int size) {
        int strides = size / stride_size();
        int remain = size % stride_size();
        const char *p = (const char *)data;

        auto print = reverse_number_ ? be_print : le_print;
        auto next_number_ptr = reverse_stride_ ? next_number_r : next_number;

        for (int i=0; i<strides; i++, p += stride_size()) {
            for (int j=0; j<stride_; j++) {
                print(next_number_ptr(stride_, num_bytes_, p, j), num_bytes_);
                if (!no_space_)
                    printf(" ");
            }
            printf("\n");
        }

        for (size = remain; size > 0; p += num_bytes_, size -= num_bytes_) {
            print(p, std::min(num_bytes_, size));
            if (!no_space_)
                printf(" ");
        }

        if (remain)
            printf("\n");
    }

    void dump(const std::vector<char> &data) {
        dump(&data[0], data.size());
    }

private:
    int stride_size() { return stride_ * num_bytes_; }

private:
    int num_bytes_;
    int stride_;
    bool reverse_stride_;
    bool reverse_number_;
    bool no_space_;
};

int main(int argc, char *argv[])
{
    CLI::App app{"dump model program"};

    uint64_t offset = 0;
    app.add_option("-o,--offset", offset, "offset", true);

    int stride = 32;
    app.add_option("-s,--stride", stride, "stride", true);

    int count = 0;
    app.add_option("-c,--count", count, "count");

    int bytes = 4;
    app.add_set("-b,--bytes", bytes, {1,2,4,8}, "the bytes of a number");

    std::string input_file;
    app.add_option("--input", input_file, "input file")->required();

    bool be = false;
    app.add_flag("--be", be, "use Big-endian mode");

    bool reverse = false;
    app.add_flag("-r", reverse, "print from right to left");

    bool no_space = false;
    app.add_flag("--nospace", no_space);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    std::vector<char> data;
    if (read_file(input_file, data, count * bytes, offset * bytes) <= 0) {
        printf("can not read file: \"%s\"\n", input_file.c_str());
        return -1;
    }

    dumper(bytes, stride, reverse, be, no_space).dump(data);

    return 0;
}

