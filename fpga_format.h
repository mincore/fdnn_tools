/* ===================================================
 * Copyright (C) 2018 chenshuangping All Right Reserved.
 *      Author: chenshuangping@speed-clouds.com
 *    Filename: format.cpp
 *     Created: 2018-04-13 13:46
 * Description:
 * ===================================================
 */
#ifndef _FORMAT_H
#define _FORMAT_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <vector>
#include "file.h"

#define STRIDE 32
#define HALF_STRIDE (STRIDE/2)

using namespace kx;

static inline int round_up(int x, int base) {
    return base * (x/base + (x%base ? 1 : 0));
}

template<class T>
static void transpose(const T *src, T *dst, int d0, int d1, int d2, int d3)
{
    auto r = *reinterpret_cast<const T(*)[d3][d2][d1][d0]>(src);
    int n = 0;

    for (int i=0; i<d0; i++) {
        for (int j=0; j<d1; j++) {
            for (int k=0; k<d2; k++) {
                for (int l=0; l<d3; l++) {
                    dst[n++] = r[l][k][j][i];
                }
            }
        }
    }
}

template<class T>
bool format_to_fpga(T &t, std::vector<float> &output,
        const std::string &input_file,
        const std::string &output_file = "")
{
    std::vector<float> input;
    if (!read_file(input_file, input))
        return false;


    bool ret = t.format(input, output);
    if (ret && !output_file.empty())
        write_file(output_file, output);

    return ret;
}

template<class T>
bool format_to_fpga(T &t, const float *src, int size, std::vector<float> &output)
{
    std::vector<float> input(src, src+size);
    return t.format(input, output);
}

template<class T>
bool format_to_fpga(T &t, const std::vector<float> &input, std::vector<float> &output)
{
    return t.format(input, output);
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

    bool format(const std::vector<float> &input, std::vector<float> &output) {
        if ((int)input.size() < outputs_*inputs_*conv_size())
            return false;

        output = std::vector<float>(size(), 0);

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

    bool fill_weight(int cell, int conv, const std::vector<float> &input, std::vector<float> &output) {
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

    bool format(const std::vector<float> &input, std::vector<float> &output) {
        if ((int)input.size() < outputs_*inputs_)
            return false;

        output = std::vector<float>(size(), 0);
        const float *in = &input[0];
        float *out = &output[0];

        for (int i=0; i<outputs_; i++) {
            for (int j=0; j<cell_n_groups_; j++) {
                out = &output[get_group_addr(i, j)];
                in += fill_group(in, out);
            }
        }

        return true;
    }

private:
    int fill_group(const float *in, float *out) {
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

    bool format(const std::vector<float> &input, std::vector<float> &output) {
        if ((int)input.size() < outputs_*inputs_)
            return false;

        output = std::vector<float>(size(), 0);
        const float *in = &input[0];

        for (int i=0; i<outputs_; i++) {
            float *out = &output[get_cell_addr(i)];
            in += fill_cell(in, out);
        }

        return true;
    }

private:
    int fill_cell(const float *in, float *out) {
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

    bool format(const std::vector<float> &input, std::vector<float> &output) {
        if ((int)input.size() < inputs_)
            return false;

        output = std::vector<float>(size(), 0);

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

    bool format(const std::vector<float> &input, std::vector<float> &output) {
        if ((int)input.size() < inputs_)
            return false;

        output = std::vector<float>(size(), 0);

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
    feature_maps(int dim, int img_h, int img_count): conv_h_(dim), img_origin_h_(img_h), img_count_(img_count) {
        switch (dim) {
        case 1: stride_imgs_ = 32; round_imgs_ = 160; break;
        case 3: stride_imgs_ = 10; round_imgs_ = 50;  break;
        case 5: stride_imgs_ = 4;  round_imgs_ = 20;  break;
        case 7: stride_imgs_ = 2;  round_imgs_ = 10;  break;
        }

        tensor_pad_ = (dim == 1) ? 0 : (dim - 1)/2;
        img_h_ = (dim == 1) ? img_origin_h_ : round_up(img_origin_h_ + 2*tensor_pad_, dim);
    }

    int round_imgs_;
    int conv_h_;
    int img_origin_h_;
    int img_count_;
    int stride_imgs_;
    int img_h_;
    int tensor_pad_;

    int round_num() { return round_up(img_count_, round_imgs_) / round_imgs_; }
    int round_h_imgs() { return round_imgs_/stride_imgs_; }
    int round_size() { return  round_h_imgs() * part_num() * img_h_ * STRIDE; }
    int size() { return round_num() * round_size(); }

    int part_num() { return img_h_/conv_h_; }
    int part_size() { return img_h_*conv_h_; }
    int map_size() { return img_h_*img_h_; }
    int map_pad() { return (STRIDE - (stride_imgs_*conv_h_)) / 2; }

    int img_addr(int img, int part) {
        return (img/stride_imgs_)*img_h_*STRIDE
            + (img%stride_imgs_)*conv_h_
            + (img%stride_imgs_>= (stride_imgs_/2) ? map_pad() : 0)
            + part * img_h_ * STRIDE;
    }

    int pixel_addr(int index) {
        int x = index % img_h_;
        int y = index / img_h_;
        return x * STRIDE + y;
    }

    void pad_input(const std::vector<float> &src, std::vector<float> &dst) {
        dst = std::vector<float>(img_h_*img_h_*img_count_, 0);

        const float *psrc = &src[0];
        float *pdst = &dst[0] + tensor_pad_*img_h_ + tensor_pad_;
        int diff = img_h_ - img_origin_h_;

        for (int i=0; i<img_count_ * img_origin_h_; i++) {
            memcpy(pdst, psrc, img_origin_h_*sizeof(float));
            pdst += (i > 0 && (i%img_origin_h_ == 0)) ? (diff*img_h_) : img_h_;
            psrc += img_origin_h_;
        }
    }

    bool format(const std::vector<float> &input1, std::vector<float> &output) {
        if ((int)input1.size() != img_origin_h_ * img_origin_h_ * img_count_) {
            printf("error, input size:%zd, img_origin_h_:%d, img_count_:%d\n", input1.size(), img_origin_h_, img_count_);
            return false;
        }

        std::vector<float> input;
        pad_input(input1, input);
        output.resize(size());

        const float *in = &input[0];
        float *out = &output[0];

        for (int img=0; img<img_count_; img++) {
            for (int part=0; part<part_num(); part++) {
                int addr = img_addr(img, part);
                in += fill_part(addr, in, out);
            }
        }

        return true;
    }

    int fill_part(int addr, const float *in, float *out) {
        for (int i=0; i<part_size(); i++) {
            int subaddr = pixel_addr(i);
            out[addr + subaddr] = in[i];
        }
        return part_size();
    }

    bool fill_img(int img, const std::vector<float> &input, std::vector<float> &output) {
        if ((int)input.size() != map_size())
            return false;

        if ((int)input.size() != map_size() * img_count_)
            return false;

        const float *in = &input[0];
        float *out = &output[0];

        for (int part=0; part<part_num(); part++) {
            in += fill_part(img_addr(img, part), in, out);
        }

        return true;
    }
};

class bn_conv {
public:
    bn_conv(int inputs): inputs_(inputs) {}

    int get_weight_addr(int index) { return (index/2)*STRIDE + (index%2); }
    int get_bias_addr(int index) { return get_weight_addr(index) + 2; }

    int size() { return (inputs_/2) * STRIDE; }

    bool format(const void *pDataW, const void *pDataB, std::vector<float> &output) {
        const char *pweight = (const char *)pDataW;
        const char *pbias = (const char *)pDataB;
        output = std::vector<float>(size(), 0);

        for (int i=0; i<inputs_; i++) {
            output[get_weight_addr(i)] = pweight[i];
            output[get_bias_addr(i)] = pbias[i];
        }

        return true;
    }

private:
    int inputs_;
};

class bn_fc {
public:
    bn_fc(int inputs): inputs_(inputs) {}

    int get_weight_addr(int index) { return index*STRIDE; }
    int get_bias_addr(int index) { return get_weight_addr(index) + 2; }

    int size() { return inputs_ * STRIDE; }

    bool format(const void *pDataW, const void *pDataB, std::vector<float> &output) {
        const char *pweight = (const char *)pDataW;
        const char *pbias = (const char *)pDataB;
        output = std::vector<float>(size(), 0);

        for (int i=0; i<inputs_; i++) {
            output[get_weight_addr(i)] = pweight[i];
            output[get_bias_addr(i)] = pbias[i];
        }

        return true;
    }

private:
    int inputs_;
};

static void test_feature_map() {
    feature_maps fms(3, 10, 54);

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

#endif