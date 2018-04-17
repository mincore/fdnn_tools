/* ===================================================
 * Copyright (C) 2018 speed-clouds All Right Reserved.
 *      Author: chenshuangping@speed-clouds.com
 *    Filename: 1.cpp
 *     Created: 2018-04-16 15:48
 * Description:
 * ===================================================
 */
#include <stdlib.h>
#include <kx/kx_file.h>
#include "CLI11.hpp"

static inline void le_print(char *p, int size) {
    for (int i=size-1; i>=0; i--)
        printf("%02x", p[i] & 0xff);
}

static void dump(char *p, int bytes, int stride, int count) {
    for (int i=0; i<count; i++, p+=bytes) {
        le_print(p, bytes);
        printf(" ");
        if (i > 0 && (i+1)%stride == 0) {
            printf("\n");
        }
    }
    if (count % stride != 0)
        printf("\n");
}

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

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    std::vector<char> data;
    if (read_file(input_file, data, 0, offset * bytes) <= 0) {
        printf("can not read file: \"%s\"\n", input_file.c_str());
        return -1;
    }

    auto size = data.size() / bytes;

    if (count == 0)
        count = size;

    if (offset + count >= size) {
        count = size - offset;
    }

    dump(&data[0], bytes, stride, count);

    return 0;
}

