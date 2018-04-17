/* ===================================================
 * Copyright (C) 2018 chenshuangping All Right Reserved.
 *      Author: mincore@STRIDE3.com
 *    Filename: w.cpp
 *     Created: 2018-04-13 13:46
 * Description:
 * ===================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <kx/kx_file.h>
#include "CLI11.hpp"

#define STRIDE 32
#define HALF_STRIDE (STRIDE/2)

typedef unsigned int u32;

static inline int round_up(int x, int base) {
    return base * (x/base + (!!x%base));
}

struct weight {
    weight(int dim): conv_dim(dim) {
        switch (dim) {
        case 1: block_w_convs = 16; block_h_convs = 10; block_pad_w = 0; break;
        case 3: block_w_convs =  5; block_h_convs = 10; block_pad_w = 1; break;
        case 5: block_w_convs =  2; block_h_convs = 10; block_pad_w = 6; break;
        case 7: block_w_convs =  2; block_h_convs =  5; block_pad_w = 2; break;
        }
    }

    int conv_dim;
    int block_w_convs;
    int block_h_convs;
    int block_pad_w;
    int input_convs;
    int output_convs;

    int block_convs() { return block_w_convs * block_h_convs; }
    int block_w() { return block_w_convs * conv_dim; }
    int block_h() { return block_h_convs * conv_dim; }

    int cell_w_convs() { return block_w_convs; }
    int cell_h_convs() { return round_up(input_convs, block_convs()) / block_w_convs; }
    int cell_convs() { return cell_w_convs() * cell_h_convs(); }
    int cell_w() { return cell_w_convs() * conv_dim; }
    int cell_h() { return cell_h_convs() * conv_dim; }

    int conv_size() { return conv_dim * conv_dim; }
    int size() { return STRIDE * cell_h() * output_convs; }

    int get_fpga_addr(int cell, int conv, int pixel) {
        int n_cell_stride = cell/2;
        int n_conv_stride = conv / block_w_convs;

        int n_pixel_stride = pixel % conv_dim;
        n_pixel_stride += n_cell_stride * cell_h();
        n_pixel_stride += n_conv_stride * conv_dim;

        int addr = n_pixel_stride * STRIDE;
        addr += (cell % 2) ? HALF_STRIDE : 0;
        addr += (conv % block_w_convs) * conv_dim;
        addr += pixel / conv_dim;

        return addr;
    }

    bool gen_fpga_data(const std::vector<u32> &input, std::vector<u32> &output) {
        if ((int)input.size() < output_convs*input_convs*conv_size())
            return false;

        output.resize(size());
        memset(&output[0], 0, sizeof(u32)*output.size());

        int n = 0;
        for (int i=0; i<output_convs; i++) {
            for (int j=0; j<input_convs; j++) {
                for (int k=0; k<conv_size(); k++) {
                    output[get_fpga_addr(i, j, k)] = input[n++];
                }
            }
        }

        return true;
    }

    bool fill_fpga_data(int cell, int conv, const std::vector<u32> &input, std::vector<u32> &output) {
        if ((int)input.size() != conv_size())
            return false;

        if ((int)output.size() != size())
            return false;

        for (int k=0; k<conv_size(); k++) {
            output[get_fpga_addr(cell, conv, k)] = input[k];
        }

        return true;
    }
};

static void test_3x3() {
    weight w(3);
    w.input_convs = 64;

    assert(w.block_pad_w + w.block_w() == HALF_STRIDE);
    assert(w.block_convs() == 50);
    assert(w.cell_w_convs() == 5);
    assert(w.cell_h_convs() == 20);
    assert(w.cell_convs() == 100);
    assert(w.cell_w() == 5*3);
    assert(w.cell_h() == 2*10*3);

    int base = 2*10*3*32 + 1*3*32 + 16 + 3;

    assert(w.get_fpga_addr(3, 6, 0) == base);
    assert(w.get_fpga_addr(3, 6, 3) == base + 32*0 + 1);
    assert(w.get_fpga_addr(3, 6, 4) == base + 32*1 + 1);
    assert(w.get_fpga_addr(3, 6, 5) == base + 32*2 + 1);
    assert(w.get_fpga_addr(3, 6, 8) == base + 32*2 + 2);
}

static void test_5x5() {
    weight w(5);
    w.input_convs = 64;

    assert(w.block_pad_w + w.block_w() == HALF_STRIDE);
    assert(w.block_convs() == 20);
    assert(w.cell_w_convs() == 2);
    assert(w.cell_h_convs() == 4*10);
    assert(w.cell_convs() == 80);
    assert(w.cell_w() == 2*5);
    assert(w.cell_h() == 4*10*5);

    int base = 4*10*5*32 + 2*5*32 + 16 + 5;

    assert(w.get_fpga_addr(3, 5, 0) == base);
    assert(w.get_fpga_addr(3, 5, 3) == base + 32*3 + 0);
    assert(w.get_fpga_addr(3, 5, 4) == base + 32*4 + 0);
    assert(w.get_fpga_addr(3, 5, 5) == base + 32*0 + 1);
    assert(w.get_fpga_addr(3, 5, 8) == base + 32*3 + 1);
}

static void test_7x7() {
    weight w(7);
    w.input_convs = 64;

    assert(w.block_pad_w + w.block_w() == HALF_STRIDE);
    assert(w.block_convs() == 10);
    assert(w.cell_w_convs() == 2);
    assert(w.cell_h_convs() == 7*5);
    assert(w.cell_convs() == 70);
    assert(w.cell_w() == 2*7);
    assert(w.cell_h() == 7*5*7);

    int base = 7*5*7*32 + 2*7*32 + 16 + 7;

    assert(w.get_fpga_addr(3, 5, 0) == base);
    assert(w.get_fpga_addr(3, 5, 3) == base + 32*3 + 0);
    assert(w.get_fpga_addr(3, 5, 4) == base + 32*4 + 0);
    assert(w.get_fpga_addr(3, 5, 5) == base + 32*5 + 0);
    assert(w.get_fpga_addr(3, 5, 8) == base + 32*1 + 1);
}

static void test_1x1() {
    weight w(1);
    w.input_convs = 256;

    assert(w.block_pad_w + w.block_w() == HALF_STRIDE);
    assert(w.block_convs() == 160);
    assert(w.cell_w_convs() == 16);
    assert(w.cell_h_convs() == 20);
    assert(w.cell_convs() == 320);
    assert(w.cell_w() == 16*1);
    assert(w.cell_h() == 2*10*1);

    int base = 20*1*32 + 16;

    assert(w.get_fpga_addr(3, 0, 0) == base);
    assert(w.get_fpga_addr(3, 3, 0) == base + 32*0 + 3);
    assert(w.get_fpga_addr(3, 4, 0) == base + 32*0 + 4);
    assert(w.get_fpga_addr(3, 17, 0) == base + 32*1 + 1);
    assert(w.get_fpga_addr(3, 18, 0) == base + 32*1 + 2);
}

static bool trans_weight(weight &w, const std::string &file_in, const std::string &file_out)
{
    std::vector<u32> input;
    std::vector<u32> output;

    read_file(file_in, input);
    bool ret = w.gen_fpga_data(input, output);
    if (ret)
        write_file(file_out, output);
    return ret;
}

static void test() {
    test_3x3();
    test_5x5();
    test_7x7();
    test_1x1();
}

class param_t {
public:
    virtual ~param_t() {}
    virtual bool run() = 0;
    bool ok() { return sub && *sub; }
    std::string name() { return sub ? sub->get_name() : ""; }

protected:
    CLI::App* sub = NULL;
};

class test_param_t: public param_t {
public:
    test_param_t(CLI::App &app) {
        sub = app.add_subcommand("test", "test");
    }

    bool run() {
        test();
        return true;
    }
};

class trans_param_t: public param_t {
public:
    trans_param_t(CLI::App &app) {
        sub = app.add_subcommand("trans", "trans");
        sub->add_option("--input", input_file, "the file to read")->required();
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--input_convs", input_convs, "input_convs")->required();
        sub->add_option("--output_convs", output_convs, "output_convs")->required();
    }

    bool run() {
        return trans();
    }

private:
    bool trans() {
        weight w(dim);
        w.input_convs = input_convs;
        w.output_convs = output_convs;
        return trans_weight(w, input_file, output_file);
    }

private:
    std::string input_file;
    std::string output_file;
    int dim;
    int input_convs;
    int output_convs;
};

class fill_param_t: public param_t {
public:
    fill_param_t(CLI::App &app) {
        sub = app.add_subcommand("fill", "fill a conv with specified value");
        sub->add_option("--input", input_file, "the file to read");
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--input_convs", input_convs, "input_convs")->required();
        sub->add_option("--output_convs", output_convs, "output_convs")->required();
        sub->add_option("--cell", cell, "which cell to fill")->required();
        sub->add_option("--conv", conv, "which conv to fill")->required();
        sub->add_option("--value", value, "fill by value")->required();
        sub->add_option("--with_index", with_index, "with_index");
    }

    bool run() {
        weight w(dim);
        w.input_convs = input_convs;
        w.output_convs = output_convs;

        std::vector<u32> conv_input(w.conv_size(), value);
        std::vector<u32> output;

        printf("dim:%d input_convs:%d output_convs:%d cell:%d conv:%d value:%08x\n", dim, input_convs, output_convs, cell, conv, value);

        if (with_index) {
            for (size_t i=0; i<conv_input.size(); i++) {
                conv_input[i] = (conv_input[i] & 0xFFFFFF00) | i;
            }
        }

        if (read_file(input_file, output) != (size_t)w.size()*4) {
            output.resize(w.size());
            memset(&output[0], 0, output.size()*4);
        }

        w.fill_fpga_data(cell, conv, conv_input, output);
        write_file(output_file, output);

        return true;
    }

private:
    std::string input_file;
    std::string output_file;
    int dim;
    int input_convs;
    int output_convs;
    int cell;
    int conv;
    u32 value;
    bool with_index = false;
};

int main(int argc, char *argv[])
{
    CLI::App app{"generic model program"};

    std::vector<std::shared_ptr<param_t> > params = {
        std::make_shared<test_param_t>(app),
        std::make_shared<trans_param_t>(app),
        std::make_shared<fill_param_t>(app)
    };

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    for (auto &param : params) {
        if (param->ok()) {
            printf("%s %s\n", param->name().c_str(), param->run() ? "done" : "failed");
        }
    }

    return 0;
}
