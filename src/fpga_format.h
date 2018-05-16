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

template<class F>
static void transpose(const F *src, F *dst, int d0, int d1, int d2, int d3)
{
    auto r = *reinterpret_cast<const F(*)[d3][d2][d1][d0]>(src);
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

template<class T, class F>
bool format_to_fpga(T &t, std::vector<F> &output,
        const std::string &input_file,
        const std::string &output_file = "")
{
    std::vector<F> input;
    if (!read_file(input_file, input))
        return false;

    bool ret = t.format(input, output);
    if (ret && !output_file.empty())
        write_file(output_file, output);

    return ret;
}

template<class T, class F>
bool format_to_fpga(T &t, const F *src, int size, std::vector<F> &output)
{
    std::vector<F> input(src, src+size);
    return t.format(input, output);
}

template<class T, class F>
bool format_to_fpga(T &t, const std::vector<F> &input, std::vector<F> &output)
{
    return t.format(input, output);
}

struct weight {
    weight(int dim, int inputs, int outputs): dim_(dim), inputs_(inputs), outputs_(outputs) {
        switch (dim_) {
        case 1: block_w_convs_ = 16; block_h_convs_ =  3*8; break;
        case 3: block_w_convs_ =  5; block_h_convs_ =  8; break;
        case 5: block_w_convs_ =  2; block_h_convs_ =  8; break;
        case 7: block_w_convs_ =  1; block_h_convs_ =  8; break;
        }
        block_pad_w_ = (HALF_STRIDE - block_w_convs_ * dim);
    }

    int dim_;
    int block_w_convs_;
    int block_h_convs_;
    int block_pad_w_;
    int inputs_;
    int outputs_;

    int conv_w() { return dim_; }
    int conv_h() { return dim_*dim_; }
    int conv_size() { return conv_w() * conv_h(); }
    int size() { return STRIDE * cell_h() * outputs_/2; }

    int block_convs() { return block_w_convs_ * block_h_convs_; }
    int block_w() { return block_w_convs_ * conv_w(); }
    int block_h() { return block_h_convs_ * conv_h(); }

    int cell_w_convs() { return block_w_convs_; }
    int cell_h_convs() { return round_up(inputs_, block_convs()) / block_w_convs_; }
    int cell_convs() { return cell_w_convs() * cell_h_convs(); }
    int cell_w() { return cell_w_convs() * conv_w(); }
    int cell_h() { return cell_h_convs() * conv_h(); }

    int get_cell_addr(int cell) {
        return (cell/2) * cell_h() * STRIDE + ((cell%2) ? HALF_STRIDE : 0);
    }

    int get_conv_addr(int conv, int sub_conv) {
        int w_convs = conv % block_w_convs_;
        int h_convs = conv / block_w_convs_;
        return (h_convs*conv_h() + sub_conv*dim_) * STRIDE + w_convs * conv_w();
    }

    template<class F>
    void fill_conv(int cell, int conv,
            const F *pconv, std::vector<F> &output) {
        int cell_addr = get_cell_addr(cell);
        int count = dim_*dim_;

        // sub_conv
        for (int sub_conv=0; sub_conv<dim_; sub_conv++) {
            int conv_addr = cell_addr + get_conv_addr(conv, sub_conv);
            // n: the number in a conv
            for (int n=0; n<count; n++) {
                int x = n / dim_;
                int y = ( n % dim_ + sub_conv  ) % dim_;
                int addr = conv_addr + x * STRIDE + y;
                output[addr] = pconv[n];
            }
        }
    }

    template<class F>
    bool format(const std::vector<F> &input, std::vector<F> &output) {
        if ((int)input.size() < outputs_*inputs_*dim_*dim_)
            return false;

        output = std::vector<F>(size(), 0);

        for (int cell=0; cell<outputs_; cell++) {
            for (int conv=0; conv<inputs_; conv++) {
                const F *pconv = &input[(cell*inputs_+conv)*dim_*dim_];
                fill_conv(cell, conv, pconv, output);
            }
        }

        return true;
    }
};

class conv_fcw {
public:
    conv_fcw(int dim, int inputs, int outputs): dim_(dim), inputs_(inputs), outputs_(outputs) {
        assert(inputs % 2 == 0);
        int dim_inputs = inputs/2*dim_*dim_;
        cell_n_groups_ = (round_up(dim_inputs, group_size()) / group_size()) * 2;
        cell_size_ = round_up(dim_inputs, block_n_stride_ * HALF_STRIDE) * 2;
    }

    int get_addr(int cell, int i_input, int index) {
        int n = (i_input/2) * dim_*dim_ + index;
        int addr = cell * cell_size_;

        addr +=  (i_input % 2 == 0 ? 0 : HALF_STRIDE);
        addr += (n / HALF_STRIDE) * STRIDE;
        addr += (n % HALF_STRIDE);

        return addr;
    }

    int group_size() { return group_n_stride_*HALF_STRIDE; }
    int size() { return cell_size_ * outputs_; }

    template<class F>
    bool format(const std::vector<F> &input, std::vector<F> &output) {
        if ((int)input.size() < outputs_*inputs_*dim_*dim_)
            return false;

        output = std::vector<F>(size(), 0);

        int n = 0;
        for (int i=0; i<outputs_; i++) {
            for (int j=0; j<inputs_; j++) {
                for (int k=0; k<dim_*dim_; k++) {
                    output[get_addr(i, j, k)] = input[n++];
                }
            }
        }

        return true;
    }

private:
    const int group_n_stride_ = 3;
    const int block_n_stride_ = 12;

private:
    int dim_;
    int inputs_;
    int outputs_;
    int cell_n_groups_;
    int cell_size_;
};

class fc_fcw {
public:
    fc_fcw(int inputs, int outputs): inputs_(inputs), outputs_(outputs) {
        cell_n_stride_ = round_up(inputs_, STRIDE * block_n_stride_) / STRIDE;
    }

    int get_cell_addr(int cell) { return cell * cell_n_stride_ * STRIDE; }
    int cell_size() { return cell_n_stride_ * STRIDE; }
    int size() { return outputs_ * cell_size(); }

    template<class F>
    bool format(const std::vector<F> &input, std::vector<F> &output) {
        if ((int)input.size() < outputs_*inputs_)
            return false;

        output = std::vector<F>(size(), 0);
        const F *in = &input[0];

        for (int cell=0; cell<outputs_; cell++) {
            F *out = &output[get_cell_addr(cell)];
            memcpy(out, in, inputs_*sizeof(F));
            in += inputs_;
        }

        return true;
    }

private:
    const int block_n_stride_ = 12;

private:
    int inputs_;
    int outputs_;
    int cell_n_stride_;
};

class bias {
public:
    bias(int inputs): inputs_(inputs) {}

    int get_bias_addr(int index) { return (index/2)*stride_ + (index%2); }
    int size() { return (inputs_/2) * stride_; }

    template<class F>
    bool format(const std::vector<F> &input, std::vector<F> &output) {
        if ((int)input.size() < inputs_)
            return false;

        output = std::vector<F>(size(), 0);

        for (int i=0; i<inputs_; i++) {
            output[get_bias_addr(i)] = input[i];
        }

        return true;
    }

private:
    int inputs_;
    const int stride_ = 8;
};

class fc_bias {
public:
    fc_bias(int inputs): inputs_(inputs) {}

    int get_bias_addr(int index) { return index*stride_; }
    int size() { return inputs_ * stride_; }

    template<class F>
    bool format(const std::vector<F> &input, std::vector<F> &output) {
        if ((int)input.size() < inputs_)
            return false;

        output = std::vector<F>(size(), 0);

        for (int i=0; i<inputs_; i++) {
            output[get_bias_addr(i)] = input[i];
        }

        return true;
    }

private:
    int inputs_;
    const int stride_ = 8;
};

struct feature_maps {
    feature_maps(int dim, int img_h, int img_count, int img_channel = 1, bool for_same_conv = false):
        conv_h_(dim), img_origin_h_(img_h), img_count_(img_count) {
        switch (dim) {
        case 1: stride_imgs_ = 32; break;
        case 3: stride_imgs_ = 10; break;
        case 5: stride_imgs_ = 4;  break;
        case 7: stride_imgs_ = 2;  break;
        }

        const int round_cols = 4;
        round_imgs_ = stride_imgs_ * round_cols;

        if (for_same_conv)
            pad0_ = (dim == 1) ? 0 : (dim - 1)/2;

        img_h_ = round_up(img_origin_h_ + 2*pad0_, dim);
        pad1_ = img_h_ - img_origin_h_ - pad0_;
    }

    int round_imgs_;
    int conv_h_;
    int img_origin_h_;
    int img_count_;
    int stride_imgs_;
    int img_h_;
    int pad0_ = 0;
    int pad1_ = 0;

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

    template<class F>
    void pad_input(const std::vector<F> &src, std::vector<F> &dst) {
        dst = std::vector<F>(img_h_*img_h_*img_count_, 0);

        const F *psrc = &src[0];
        F *pdst = &dst[0] + pad0_ * img_h_ + pad0_;
        int diff = img_h_ - img_origin_h_ + 1;

        for (int i=0; i<img_count_ * img_origin_h_; i++) {
            memcpy(pdst, psrc, img_origin_h_*sizeof(F));
            psrc += img_origin_h_;
            pdst += ((i+1)%img_origin_h_ == 0) ? (diff*img_h_) : img_h_;
        }
    }

    template<class F>
    bool format(const std::vector<F> &input1, std::vector<F> &output) {
        if ((int)input1.size() != img_origin_h_ * img_origin_h_ * img_count_)
            return false;

        std::vector<F> input;
        pad_input(input1, input);
        output.resize(size());

        const F *in = &input[0];
        F *out = &output[0];

        for (int img=0; img<img_count_; img++) {
            for (int part=0; part<part_num(); part++) {
                int addr = img_addr(img, part);
                in += fill_part(addr, in, out);
            }
        }

        return true;
    }

    template<class F>
    int fill_part(int addr, const F *in, F *out) {
        for (int i=0; i<part_size(); i++) {
            int subaddr = pixel_addr(i);
            out[addr + subaddr] = in[i];
        }
        return part_size();
    }
};

class bn_conv {
public:
    bn_conv(int inputs): inputs_(inputs) {}

    int get_weight_addr(int index) { return (index/2)*STRIDE + (index%2); }
    int get_bias_addr(int index) { return get_weight_addr(index) + 2; }

    int size() { return (inputs_/2) * STRIDE; }

    template<class F>
    bool format(const void *pDataW, const void *pDataB, std::vector<F> &output) {
        const F *pweight = (const F *)pDataW;
        const F *pbias = (const F *)pDataB;
        output = std::vector<F>(size(), 0);

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

    template<class F>
    bool format(const void *pDataW, const void *pDataB, std::vector<F> &output) {
        const F *pweight = (const F *)pDataW;
        const F *pbias = (const F *)pDataB;
        output = std::vector<F>(size(), 0);

        for (int i=0; i<inputs_; i++) {
            output[get_weight_addr(i)] = pweight[i];
            output[get_bias_addr(i)] = pbias[i];
        }

        return true;
    }

private:
    int inputs_;
};

#endif
