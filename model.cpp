/* ===================================================
 * Copyright (C) 2018 chenshuangping All Right Reserved.
 *      Author: mincore@163.com
 *    Filename: w.cpp
 *     Created: 2018-04-13 13:46
 * Description:
 * ===================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "kx/kx_file.h"
#include "kx/CLI11.hpp"

#define STRIDE 32
#define HALF_STRIDE (STRIDE/2)

typedef unsigned int u32;

using namespace kx;

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
    int output_count;

    int block_convs() { return block_w_convs * block_h_convs; }
    int block_w() { return block_w_convs * conv_dim; }
    int block_h() { return block_h_convs * conv_dim; }

    int cell_w_convs() { return block_w_convs; }
    int cell_h_convs() { return round_up(input_convs, block_convs()) / block_w_convs; }
    int cell_convs() { return cell_w_convs() * cell_h_convs(); }
    int cell_w() { return cell_w_convs() * conv_dim; }
    int cell_h() { return cell_h_convs() * conv_dim; }

    int conv_size() { return conv_dim * conv_dim; }
    int size() { return STRIDE * cell_h() * output_count; }

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

    bool trans(const std::vector<u32> &input, std::vector<u32> &output) {
        if ((int)input.size() < output_count*input_convs*conv_size())
            return false;

        output.resize(size());
        memset(&output[0], 0, sizeof(u32)*output.size());

        int n = 0;
        for (int i=0; i<output_count; i++) {
            for (int j=0; j<input_convs; j++) {
                for (int k=0; k<conv_size(); k++) {
                    output[get_fpga_addr(i, j, k)] = input[n++];
                }
            }
        }

        return true;
    }

    bool fill_weight(int cell, int conv, const std::vector<u32> &input, std::vector<u32> &output) {
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

class conv_fcw {
public:
    conv_fcw(int input_count, int output_count):
        input_count_(input_count), output_count_(output_count) {
        cell_n_stride_ = (round_up(input_count, block_n_pixel_) / block_n_pixel_) * block_n_stride_;
    }

    int get_fpga_addr(int cell, int group) {
        int addr = cell * cell_n_stride_ * STRIDE + (group/2) * group_n_stride_ * STRIDE;
        if (group%2)
            addr += HALF_STRIDE;
        return addr;
    }

    int input_count() { return input_count_; }
    int output_count() { return output_count_; }
    int size() { return cell_n_stride_ * output_count_ * STRIDE; }
    int cell_n_group() { return cell_n_stride_ / group_n_stride_; }

private:
    const int group_n_stride_ = 3;
    const int block_n_stride_ = 15;
    const int block_n_pixel_ = block_n_stride_ * STRIDE;

private:
    int input_count_;
    int output_count_;
    int cell_n_stride_;
};

static void test_conv_fcw() {
    conv_fcw w(512, 2);
    assert(w.size() == 1920);
    assert(w.get_fpga_addr(1, 3) == 1072);
}

class fc_fcw {
public:
    fc_fcw(int input_count, int output_count):
        input_count_(input_count), output_count_(output_count) {
        int input_n_stride = input_count_ / STRIDE;
        cell_n_stride_ = round_up(input_n_stride, block_n_stride_);
    }

    int get_fpga_addr(int cell) { return cell * cell_n_stride_ * STRIDE; }
    int input_count() { return input_count_; }
    int output_count() { return output_count_; }
    int size() { return cell_n_stride_ * output_count_ * STRIDE; }

private:
    const int block_n_stride_ = 15;

private:
    int input_count_;
    int output_count_;
    int cell_n_stride_;
};

static void test_fc_fcw() {
    fc_fcw w(512, 2);
    assert(w.size() == 1920);
    assert(w.get_fpga_addr(1) == 960);
}

class bias {
public:
    bias(int input_count): input_count_(input_count) {}

    int get_fpga_addr(int index) { return (index/2)*STRIDE + (index%2); }
    int size() { return (input_count_/2) * STRIDE; }

private:
    int input_count_;
};

static void test_bias() {
    bias b(8);
    assert(b.size() == 4*STRIDE);
    assert(b.get_fpga_addr(5) == 2*STRIDE+1);
    assert(b.get_fpga_addr(6) == 3*STRIDE+0);
}

class fc_bias {
public:
    fc_bias(int input_count): input_count_(input_count) {}

    int get_fpga_addr(int index) { return index*STRIDE; }
    int size() { return input_count_ * STRIDE; }

private:
    int input_count_;
};

static void test_fc_bias() {
    fc_bias b(8);
    assert(b.size() == 8*STRIDE);
    assert(b.get_fpga_addr(5) == 5*STRIDE);
}

struct feature_maps {
    feature_maps(int dim, int img_h1, int img_count1): conv_h(dim), img_origin_h(img_h1), img_count(img_count1) {
        switch (dim) {
        case 1: stride_imgs = 32; round_imgs = 160; break;
        case 3: stride_imgs = 10; round_imgs = 50; break;
        case 5: stride_imgs = 4; round_imgs = 20; break;
        case 7: stride_imgs = 2; round_imgs = 10; break;
        }

        if (dim >= 3)
            img_h = round_up(img_origin_h + tensor_pad(dim), dim);
    }

    int tensor_pad(int k) {
        return (k - 1)/2;
    }

    int round_imgs;
    int conv_h;
    int img_origin_h;
    int img_count;
    int stride_imgs;
    int img_h;

    int img_pad() { return (img_h - img_origin_h) / 2; }

    int round_num() { return round_up(img_count, round_imgs) / round_imgs; }
    int round_h_imgs() { return round_imgs/stride_imgs; }
    int round_size() { return  round_h_imgs() * part_num() * img_h * STRIDE; }
    int size() { return round_num() * round_size(); }

    int part_num() { return img_h/conv_h; }
    int part_size() { return img_h*conv_h; }
    int map_size() { return img_h*img_h; }
    int map_pad() { return (STRIDE - (stride_imgs*conv_h)) / 2; }

    int img_addr(int img, int part) {
        return (img/stride_imgs)*img_h*STRIDE
            + (img%stride_imgs)*conv_h
            + (img%stride_imgs>= (stride_imgs/2) ? map_pad() : 0)
            + part * img_h * STRIDE;
    }

    int pixel_addr(int index) {
        int x = index % img_h;
        int y = index / img_h;
        return x * STRIDE + y;
    }

    void pad_input(int pad, const std::vector<u32> &src, std::vector<u32> &dst) {
        dst = std::vector<u32>(img_h*img_h*img_count, 0);

        const u32 *psrc = &src[0];
        u32 *pdst = &dst[0] + 2*img_h;

        for (int i=0; i<img_count * img_origin_h; i++) {
            memcpy(pdst, psrc, img_origin_h*4);
            pdst += (i > 0 && (i%img_origin_h == 0)) ? 4*img_h : img_h;
            psrc += img_origin_h;
        }
    }

    bool trans(const std::vector<u32> &input1, std::vector<u32> &output) {
        if ((int)input1.size() != img_origin_h * img_origin_h * img_count) {
            printf("error, input size:%zd, img_origin_h:%d, img_count:%d\n", input1.size(), img_origin_h, img_count);
            return false;
        }

        int pad = img_pad();
        std::vector<u32> input;
        pad_input(pad, input1, input);

        output.resize(size());

        const u32 *in = &input[0];
        u32 *out = &output[0];

        for (int img=0; img<img_count; img++) {
            for (int part=0; part<part_num(); part++) {
                int addr = img_addr(img, part);
                in += fill_part(addr, in, out);
            }
        }

        return true;
    }

    int fill_part(int addr, const u32 *in, u32 *out) {
        for (int i=0; i<part_size(); i++) {
            int subaddr = pixel_addr(i);
            out[addr + subaddr] = in[i];
        }
        return part_size();
    }

    bool fill_img(int img, const std::vector<u32> &input, std::vector<u32> &output) {
        if ((int)input.size() != map_size())
            return false;

        if ((int)input.size() != map_size() * img_count)
            return false;

        const u32 *in = &input[0];
        u32 *out = &output[0];

        for (int part=0; part<part_num(); part++) {
            in += fill_part(img_addr(img, part), in, out);
        }

        return true;
    }
};

static void test_feature_map() {
    feature_maps fms(3, 12, 54);

    assert(fms.round_num() == 2);
    assert(fms.map_pad() == 1);
    assert(fms.size() == 6*12*4*32);
    assert(fms.img_addr(3, 0) == 3*3);
    assert(fms.img_addr(3, 2) == fms.img_addr(3, 0) + 12*2*32);
    assert(fms.img_addr(7, 0) == 7*3+1);
    assert(fms.img_addr(13, 0) == 12*32 + 3*3);
    assert(fms.img_addr(17, 0) == 12*32 + 7*3 + 1);
    assert(fms.img_addr(17, 2) == fms.img_addr(17, 0) + 12*2*32);
}

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
    bool ret = w.trans(input, output);
    if (ret)
        write_file(file_out, output);
    return ret;
}

static void test() {
    test_3x3();
    test_5x5();
    test_7x7();
    test_1x1();
    test_conv_fcw();
    test_fc_fcw();
    test_bias();
    test_fc_bias();
    test_feature_map();
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
        sub = app.add_subcommand("trans", "trans weight to fpga format");
        sub->add_option("--input", input_file, "the file to read")->required();
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--input_convs", input_convs, "input_convs")->required();
        sub->add_option("--output_count", output_count, "output_count")->required();
    }

    bool run() {
        return trans();
    }

private:
    bool trans() {
        weight w(dim);
        w.input_convs = input_convs;
        w.output_count = output_count;
        return trans_weight(w, input_file, output_file);
    }

private:
    std::string input_file;
    std::string output_file;
    int dim;
    int input_convs;
    int output_count;
};

class trans_img_param_t: public param_t {
public:
    trans_img_param_t(CLI::App &app) {
        sub = app.add_subcommand("trans-img", "trans img to fpga format");
        sub->add_option("--input", input_file, "the file to read")->required();
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--img_h", img_h, "img height")->required();
        sub->add_option("--img_count", img_count, "img_count")->required();
    }

    bool run() {
        return trans();
    }

private:
    bool trans() {
        feature_maps fms(3, img_h, img_count);

        std::vector<u32> input;
        std::vector<u32> output;

        read_file(input_file, input);

        bool ret = fms.trans(input, output);
        if (ret)
            write_file(output_file, output);

        return ret;
    }

private:
    std::string input_file;
    std::string output_file;
    int dim;
    int img_h;
    int img_count;
};

class fill_param_t: public param_t {
public:
    fill_param_t(CLI::App &app) {
        sub = app.add_subcommand("fill", "fill a conv with specified value");
        sub->add_option("--input", input_file, "the file to read");
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--input_convs", input_convs, "input_convs")->required();
        sub->add_option("--output_count", output_count, "output_count")->required();
        sub->add_option("--cell", cell, "which cell to fill")->required();
        sub->add_option("--conv", conv, "which conv to fill")->required();
        sub->add_option("--value", value, "fill by value")->required();
        sub->add_option("--with_index", with_index, "with_index");
    }

    bool run() {
        weight w(dim);
        w.input_convs = input_convs;
        w.output_count = output_count;

        std::vector<u32> conv_input(w.conv_size(), value);
        std::vector<u32> output;

        printf("dim:%d input_convs:%d output_count:%d cell:%d conv:%d value:%08x\n",
                dim, input_convs, output_count, cell, conv, value);

        if (with_index) {
            for (size_t i=0; i<conv_input.size(); i++) {
                conv_input[i] = (conv_input[i] & 0xFFFFFF00) | i;
            }
        }

        if (read_file(input_file, output) != (size_t)w.size()*4) {
            output.resize(w.size());
            memset(&output[0], 0, output.size()*4);
        }

        w.fill_weight(cell, conv, conv_input, output);
        write_file(output_file, output);

        return true;
    }

private:
    std::string input_file;
    std::string output_file;
    int dim;
    int input_convs;
    int output_count;
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
        std::make_shared<trans_img_param_t>(app),
        std::make_shared<fill_param_t>(app),
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
