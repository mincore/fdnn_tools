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

template<class T>
bool format_to_fpga(T &t, std::vector<u32> &output, const std::string &input_file, const std::string &output_file = "")
{
    std::vector<u32> input;
    if (!read_file(input_file, input))
        return false;

    bool ret = t.format(input, output);
    if (ret && !output_file.empty())
        write_file(output_file, output);

    return ret;
}

struct weight {
    weight(int dim, int inputs, int outputs): dim_(dim), inputs_(inputs), outputs_(outputs) {
        switch (dim_) {
        case 1: block_w_convs_ = 16; block_h_convs_ = 10; block_pad_w_ = 0; break;
        case 3: block_w_convs_ =  5; block_h_convs_ = 10; block_pad_w_ = 1; break;
        case 5: block_w_convs_ =  2; block_h_convs_ = 10; block_pad_w_ = 6; break;
        case 7: block_w_convs_ =  2; block_h_convs_ =  5; block_pad_w_ = 2; break;
        }
    }

    int dim_;
    int block_w_convs_;
    int block_h_convs_;
    int block_pad_w_;
    int inputs_;
    int outputs_;

    int block_convs() { return block_w_convs_ * block_h_convs_; }
    int block_w() { return block_w_convs_ * dim_; }
    int block_h() { return block_h_convs_ * dim_; }

    int cell_w_convs() { return block_w_convs_; }
    int cell_h_convs() { return round_up(inputs_, block_convs()) / block_w_convs_; }
    int cell_convs() { return cell_w_convs() * cell_h_convs(); }
    int cell_w() { return cell_w_convs() * dim_; }
    int cell_h() { return cell_h_convs() * dim_; }

    int conv_size() { return dim_ * dim_; }
    int size() { return STRIDE * cell_h() * outputs_; }

    int get_pixel_addr(int cell, int conv, int pixel) {
        int n_cell_stride = cell/2;
        int n_conv_stride = conv / block_w_convs_;

        int n_pixel_stride = pixel % dim_;
        n_pixel_stride += n_cell_stride * cell_h();
        n_pixel_stride += n_conv_stride * dim_;

        int addr = n_pixel_stride * STRIDE;
        addr += (cell % 2) ? HALF_STRIDE : 0;
        addr += (conv % block_w_convs_) * dim_;
        addr += pixel / dim_;

        return addr;
    }

    bool format(const std::vector<u32> &input, std::vector<u32> &output) {
        if ((int)input.size() < outputs_*inputs_*conv_size())
            return false;

        output = std::vector<u32>(size(), 0);

        int n = 0;
        for (int i=0; i<outputs_; i++) {
            for (int j=0; j<inputs_; j++) {
                for (int k=0; k<conv_size(); k++) {
                    output[get_pixel_addr(i, j, k)] = input[n++];
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
            output[get_pixel_addr(cell, conv, k)] = input[k];
        }

        return true;
    }
};

class conv_fcw {
public:
    conv_fcw(int inputs, int outputs): inputs_(inputs), outputs_(outputs) {
        cell_n_stride_ = (round_up(inputs_, block_n_pixel_) / block_n_pixel_) * block_n_stride_;
        cell_n_groups_ = cell_n_stride_ / group_n_stride_ * 2;
    }

    int get_group_addr(int cell, int group) {
        int addr = cell * cell_n_stride_ * STRIDE + (group/2) * group_n_stride_ * STRIDE;
        if (group%2)
            addr += HALF_STRIDE;
        return addr;
    }

    int size() { return cell_n_stride_ * outputs_ * STRIDE; }
    int group_size() { return group_n_stride_*HALF_STRIDE; }

    bool format(const std::vector<u32> &input, std::vector<u32> &output) {
        if ((int)input.size() < outputs_*inputs_)
            return false;

        output = std::vector<u32>(size(), 0);
        const u32 *in = &input[0];
        u32 *out = &output[0];

        for (int i=0; i<outputs_; i++) {
            for (int j=0; j<cell_n_groups_; j++) {
                out = &output[get_group_addr(i, j)];
                in += fill_group(in, out);
            }
        }

        return true;
    }

private:
    int fill_group(const u32 *in, u32 *out) {
        for (int k=0; k<group_n_stride_; k++) {
            memcpy(out, in, HALF_STRIDE*4);
        }
        return group_size();
    }

private:
    const int group_n_stride_ = 3;
    const int block_n_stride_ = 15;
    const int block_n_pixel_ = block_n_stride_ * STRIDE;

private:
    int inputs_;
    int outputs_;
    int cell_n_stride_;
    int cell_n_groups_;
};

static void test_conv_fcw() {
    conv_fcw w(512, 2);
    assert(w.size() == 1920);
    assert(w.get_group_addr(1, 3) == 1072);
}

class fc_fcw {
public:
    fc_fcw(int inputs, int outputs): inputs_(inputs), outputs_(outputs) {
        cell_n_stride_ = round_up(inputs_ / STRIDE, block_n_stride_);
    }

    int get_cell_addr(int cell) { return cell * cell_n_stride_ * STRIDE; }
    int size() { return cell_n_stride_ * outputs_ * STRIDE; }
    int cell_size() { return block_n_stride_*HALF_STRIDE; }

    bool format(const std::vector<u32> &input, std::vector<u32> &output) {
        if ((int)input.size() < outputs_*inputs_)
            return false;

        output = std::vector<u32>(size(), 0);
        const u32 *in = &input[0];

        for (int i=0; i<outputs_; i++) {
            u32 *out = &output[get_cell_addr(i)];
            in += fill_cell(in, out);
        }

        return true;
    }

private:
    int fill_cell(const u32 *in, u32 *out) {
        memcpy(out, in, inputs_);
        return cell_size();
    }

private:
    const int block_n_stride_ = 15;

private:
    int inputs_;
    int outputs_;
    int cell_n_stride_;
};

static void test_fc_fcw() {
    fc_fcw w(512, 2);
    assert(w.size() == 1920);
    assert(w.get_cell_addr(1) == 960);
}

class bias {
public:
    bias(int inputs): inputs_(inputs) {}

    int get_bias_addr(int index) { return (index/2)*STRIDE + (index%2); }
    int size() { return (inputs_/2) * STRIDE; }

    bool format(const std::vector<u32> &input, std::vector<u32> &output) {
        if ((int)input.size() < inputs_)
            return false;

        output = std::vector<u32>(size(), 0);

        for (int i=0; i<inputs_; i++) {
            output[get_bias_addr(i)] = input[i];
        }

        return true;
    }

private:
    int inputs_;
};

static void test_bias() {
    bias b(8);
    assert(b.size() == 4*STRIDE);
    assert(b.get_bias_addr(5) == 2*STRIDE+1);
    assert(b.get_bias_addr(6) == 3*STRIDE+0);
}

class fc_bias {
public:
    fc_bias(int inputs): inputs_(inputs) {}

    int get_bias_addr(int index) { return index*STRIDE; }
    int size() { return inputs_ * STRIDE; }

    bool format(const std::vector<u32> &input, std::vector<u32> &output) {
        if ((int)input.size() < inputs_)
            return false;

        output = std::vector<u32>(size(), 0);

        for (int i=0; i<inputs_; i++) {
            output[get_bias_addr(i)] = input[i];
        }

        return true;
    }

private:
    int inputs_;
};

static void test_fc_bias() {
    fc_bias b(8);
    assert(b.size() == 8*STRIDE);
    assert(b.get_bias_addr(5) == 5*STRIDE);
}

struct feature_maps {
    feature_maps(int dim, int img_h1, int img_count1): conv_h(dim), img_origin_h(img_h1), img_count(img_count1) {
        switch (dim) {
        case 1: stride_imgs = 32; round_imgs = 160; break;
        case 3: stride_imgs = 10; round_imgs = 50; break;
        case 5: stride_imgs = 4; round_imgs = 20; break;
        case 7: stride_imgs = 2; round_imgs = 10; break;
        }

        img_h = img_origin_h;

        if (dim >= 3) {
            if (img_origin_h % dim != 0)
                img_h = round_up(img_origin_h + tensor_pad(dim), dim);
        }
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

    bool format(const std::vector<u32> &input1, std::vector<u32> &output) {
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
    assert(fms.size() == 2*5*12*4*32);
    assert(fms.img_addr(3, 0) == 3*3);
    assert(fms.img_addr(3, 2) == fms.img_addr(3, 0) + 12*2*32);
    assert(fms.img_addr(7, 0) == 7*3+1);
    assert(fms.img_addr(13, 0) == 12*32 + 3*3);
    assert(fms.img_addr(17, 0) == 12*32 + 7*3 + 1);
    assert(fms.img_addr(17, 2) == fms.img_addr(17, 0) + 12*2*32);
}

static void test_3x3() {
    weight w(3, 64, 3);

    assert(w.block_pad_w_ + w.block_w() == HALF_STRIDE);
    assert(w.block_convs() == 50);
    assert(w.cell_w_convs() == 5);
    assert(w.cell_h_convs() == 20);
    assert(w.cell_convs() == 100);
    assert(w.cell_w() == 5*3);
    assert(w.cell_h() == 2*10*3);

    int base = 2*10*3*32 + 1*3*32 + 16 + 3;

    assert(w.get_pixel_addr(3, 6, 0) == base);
    assert(w.get_pixel_addr(3, 6, 3) == base + 32*0 + 1);
    assert(w.get_pixel_addr(3, 6, 4) == base + 32*1 + 1);
    assert(w.get_pixel_addr(3, 6, 5) == base + 32*2 + 1);
    assert(w.get_pixel_addr(3, 6, 8) == base + 32*2 + 2);
}

static void test_5x5() {
    weight w(5, 64, 3);

    assert(w.block_pad_w_ + w.block_w() == HALF_STRIDE);
    assert(w.block_convs() == 20);
    assert(w.cell_w_convs() == 2);
    assert(w.cell_h_convs() == 4*10);
    assert(w.cell_convs() == 80);
    assert(w.cell_w() == 2*5);
    assert(w.cell_h() == 4*10*5);

    int base = 4*10*5*32 + 2*5*32 + 16 + 5;

    assert(w.get_pixel_addr(3, 5, 0) == base);
    assert(w.get_pixel_addr(3, 5, 3) == base + 32*3 + 0);
    assert(w.get_pixel_addr(3, 5, 4) == base + 32*4 + 0);
    assert(w.get_pixel_addr(3, 5, 5) == base + 32*0 + 1);
    assert(w.get_pixel_addr(3, 5, 8) == base + 32*3 + 1);
}

static void test_7x7() {
    weight w(7, 64, 3);

    assert(w.block_pad_w_ + w.block_w() == HALF_STRIDE);
    assert(w.block_convs() == 10);
    assert(w.cell_w_convs() == 2);
    assert(w.cell_h_convs() == 7*5);
    assert(w.cell_convs() == 70);
    assert(w.cell_w() == 2*7);
    assert(w.cell_h() == 7*5*7);

    int base = 7*5*7*32 + 2*7*32 + 16 + 7;

    assert(w.get_pixel_addr(3, 5, 0) == base);
    assert(w.get_pixel_addr(3, 5, 3) == base + 32*3 + 0);
    assert(w.get_pixel_addr(3, 5, 4) == base + 32*4 + 0);
    assert(w.get_pixel_addr(3, 5, 5) == base + 32*5 + 0);
    assert(w.get_pixel_addr(3, 5, 8) == base + 32*1 + 1);
}

static void test_1x1() {
    weight w(1, 256, 3);

    assert(w.block_pad_w_ + w.block_w() == HALF_STRIDE);
    assert(w.block_convs() == 160);
    assert(w.cell_w_convs() == 16);
    assert(w.cell_h_convs() == 20);
    assert(w.cell_convs() == 320);
    assert(w.cell_w() == 16*1);
    assert(w.cell_h() == 2*10*1);

    int base = 20*1*32 + 16;

    assert(w.get_pixel_addr(3, 0, 0) == base);
    assert(w.get_pixel_addr(3, 3, 0) == base + 32*0 + 3);
    assert(w.get_pixel_addr(3, 4, 0) == base + 32*0 + 4);
    assert(w.get_pixel_addr(3, 17, 0) == base + 32*1 + 1);
    assert(w.get_pixel_addr(3, 18, 0) == base + 32*1 + 2);
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
        printf("testing 3x3\n");
        test_3x3();
        printf("testing 5x5\n");
        test_5x5();
        printf("testing 7x7\n");
        test_7x7();
        printf("testing 1x1\n");
        test_1x1();
        printf("testing conv_fcw\n");
        test_conv_fcw();
        printf("testing fc_fcw\n");
        test_fc_fcw();
        printf("testing bias\n");
        test_bias();
        printf("testing fc_bias\n");
        test_fc_bias();
        printf("testing feature_map\n");
        test_feature_map();
        return true;
    }
};

class format_weight_param_t: public param_t {
public:
    format_weight_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-weight", "format weight to fpga format");
        sub->add_option("--input", input_file_, "the file to read")->required();
        sub->add_option("--output", output_file_, "the file to write")->required();
        sub->add_set("--dim", dim_, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--inputs", inputs_, "input count")->required();
        sub->add_option("--outputs", outputs_, "output count")->required();
    }

    bool run() {
        weight w(dim_, inputs_, outputs_);
        std::vector<u32> output;
        return format_to_fpga(w, output, input_file_, output_file_);
    }

private:
    std::string input_file_;
    std::string output_file_;
    int dim_;
    int inputs_;
    int outputs_;
};

class format_convfcw_param_t: public param_t {
public:
    format_convfcw_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-convfcw", "format conv_fc weight to fpga format");
        sub->add_option("--input", input_file_, "the file to read")->required();
        sub->add_option("--output", output_file_, "the file to write")->required();
        sub->add_option("--inputs", inputs_, "input count")->required();
        sub->add_option("--outputs", outputs_, "output count")->required();
    }

    bool run() {
        conv_fcw w(inputs_, outputs_);
        std::vector<u32> output;
        return format_to_fpga(w, output, input_file_, output_file_);
    }

private:
    std::string input_file_;
    std::string output_file_;
    int inputs_;
    int outputs_;
};

class format_fcfcw_param_t: public param_t {
public:
    format_fcfcw_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-fcfcw", "format fc_fc weight to fpga format");
        sub->add_option("--input", input_file_, "the file to read")->required();
        sub->add_option("--output", output_file_, "the file to write")->required();
        sub->add_option("--inputs", inputs_, "input count")->required();
        sub->add_option("--outputs", outputs_, "output count")->required();
    }

    bool run() {
        fc_fcw w(inputs_, outputs_);
        std::vector<u32> output;
        return format_to_fpga(w, output, input_file_, output_file_);
    }

private:
    std::string input_file_;
    std::string output_file_;
    int inputs_;
    int outputs_;
};

class format_bias_param_t: public param_t {
public:
    format_bias_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-bias", "format bias to fpga format");
        sub->add_option("--input", input_file_, "the file to read")->required();
        sub->add_option("--output", output_file_, "the file to write")->required();
        sub->add_option("--inputs", inputs_, "input count")->required();
    }

    bool run() {
        bias b(inputs_);
        std::vector<u32> output;
        return format_to_fpga(b, output, input_file_, output_file_);
    }

private:
    std::string input_file_;
    std::string output_file_;
    int inputs_;
};

class format_fcbias_param_t: public param_t {
public:
    format_fcbias_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-fcbias", "format fcbias to fpga format");
        sub->add_option("--input", input_file_, "the file to read")->required();
        sub->add_option("--output", output_file_, "the file to write")->required();
        sub->add_option("--inputs", inputs_, "input count")->required();
    }

    bool run() {
        fc_bias b(inputs_);
        std::vector<u32> output;
        return format_to_fpga(b, output, input_file_, output_file_);
    }

private:
    std::string input_file_;
    std::string output_file_;
    int inputs_;
};

class format_img_param_t: public param_t {
public:
    format_img_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-img", "format img to fpga format");
        sub->add_option("--input", input_file, "the file to read")->required();
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--img_h", img_h, "img height")->required();
        sub->add_option("--img_count", img_count, "img_count")->required();
    }

    bool run() {
        feature_maps fms(3, img_h, img_count);
        std::vector<u32> output;
        return format_to_fpga(fms, output, input_file, output_file);
    }

private:
    std::string input_file;
    std::string output_file;
    int dim;
    int img_h;
    int img_count;
};

class fill_conv_param_t: public param_t {
public:
    fill_conv_param_t(CLI::App &app) {
        sub = app.add_subcommand("fill-conv", "fill a conv with specified value");
        sub->add_option("--input", input_file, "the file to read");
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--inputs_", inputs_, "inputs_")->required();
        sub->add_option("--outputs_", outputs_, "outputs_")->required();
        sub->add_option("--cell", cell, "which cell to fill")->required();
        sub->add_option("--conv", conv, "which conv to fill")->required();
        sub->add_option("--value", value, "fill by value")->required();
        sub->add_option("--with_index", with_index, "with_index");
    }

    bool run() {
        weight w(dim, inputs_, outputs_);

        std::vector<u32> conv_input(w.conv_size(), value);
        std::vector<u32> output;

        printf("dim:%d inputs_:%d outputs_:%d cell:%d conv:%d value:%08x\n",
                dim, inputs_, outputs_, cell, conv, value);

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
    int inputs_;
    int outputs_;
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
        std::make_shared<format_weight_param_t>(app),
        std::make_shared<format_convfcw_param_t>(app),
        std::make_shared<format_fcfcw_param_t>(app),
        std::make_shared<format_bias_param_t>(app),
        std::make_shared<format_fcbias_param_t>(app),
        std::make_shared<format_img_param_t>(app),
        std::make_shared<fill_conv_param_t>(app),
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
