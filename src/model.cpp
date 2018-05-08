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
#include <random>

#include "CLI11.hpp"
#include "fpga_format.h"

using namespace kx;

CLI::App app{"generic model program"};

class xrand {
public:
    operator uint32_t () {
        return rd();
    }

    operator float() {
        return ((float)(rd() % 0xffff)) / 0xffff;
    }

private:
    std::random_device rd;
};

xrand rd;

class param_t {
public:
    param_t(const std::string &cmd, const std::string &desc) {
        sub = app.add_subcommand(cmd, desc);
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_option("--value", value, "make by a value");
        sub->add_flag("-f,--float", use_float, "use float");
        sub->add_flag("--rand", rand, "rand");
        sub->add_flag("--save-src", save_src, "save xxx.bin.src file");
    }

    virtual ~param_t() {}
    virtual bool run() = 0;
    bool ok() { return sub && *sub; }
    std::string name() { return sub ? sub->get_name() : ""; }

protected:
    CLI::App* sub = NULL;
    std::string output_file;
    bool use_float = false;
    bool rand = false;
    bool save_src = false;
    uint32_t value = 0;
};

class test_param_t: public param_t {
public:
    test_param_t(): param_t("test", "test") {
        auto opts = sub->get_options();
        for (auto &i: opts) {
            if (i->get_name() == "--output") {
                sub->remove_option(i);
                break;
            }
        }
    }

    bool run() {
        printf("testing 1x1\n");
        test_1x1();
        printf("testing 3x3\n");
        test_3x3();
        printf("testing 5x5\n");
        test_5x5();
        printf("testing 7x7\n");
        test_7x7();
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

class make_weight_param_t: public param_t {
public:
    make_weight_param_t(): param_t("make-weight", "make a weight with specified value") {
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--inputs", inputs, "inputs")->required();
        sub->add_option("--outputs", outputs, "outputs")->required();
    }

    bool run() {
        if (use_float)
            _run<float>();
        else
            _run<uint32_t>();

        return true;
    }

    template<class T>
    bool _run() {
        std::vector<T> input;
        std::vector<T> output;
        make_input(input);
        
        if (save_src) {
            write_file(output_file + ".src", input);
        }
        
        weight w(dim, inputs, outputs);
        w.format(input, output);
        write_file(output_file, output);

        return true;
    }

    template<class T>
    void make_input(std::vector<T> &input) {
        int size = dim*dim;
        input = std::vector<T>(outputs*inputs*size, value);
        if (value)
            return;

        int n = 0;

        for (int i=0; i<outputs; i++) {
            for (int j=0; j<inputs; j++) {
                T v = 0.0;
                for (int k=0; k<size; k++) {
                    input[n++] = rand ? rd : v++;
                }
            }
        }
    }

private:
    int dim;
    int inputs;
    int outputs;
    bool use_float = false;
};

template<class A>
class make_bias_param_t: public param_t {
public:
    make_bias_param_t(const std::string &name): param_t(std::string("make-") + name, "make a bias with specified value") {
        sub->add_option("--inputs", inputs, "inputs")->required();
    }

    bool run() {
        if (use_float)
            _run<float>();
        else
            _run<uint32_t>();

        return true;
    }

    template<class T>
    bool _run() {
        std::vector<T> input;
        std::vector<T> output;
        make_input(input);
        
        if (save_src) {
            write_file(output_file + ".src", input);
        }

        A b(inputs);
        b.format(input, output);
        write_file(output_file, output);

        return true;
    }

    template<class T>
    void make_input(std::vector<T> &input) {
        input = std::vector<T>(inputs, value);
        if (value)
            return;

        T v = 0;
        for (size_t i=0; i<input.size(); i++, v++) {
            input[i] = rand ? rd : v;
        }
    }

private:
    int inputs;
};

template<class A>
class make_fcw_param_t: public param_t {
public:
    make_fcw_param_t(const std::string &name): param_t(std::string("make-")+name, "make a fc weight with specified value") {
        sub->add_option("--inputs", inputs, "inputs")->required();
        sub->add_option("--outputs", outputs, "outputs")->required();
    }

    bool run() {
        if (use_float)
            _run<float>();
        else
            _run<uint32_t>();

        return true;
    }


    template<class T>
    bool _run() {
        std::vector<T> input;
        std::vector<T> output;
        make_input(input);
        
        if (save_src) {
            write_file(output_file + ".src", input);
        }

        A w(inputs, outputs);
        w.format(input, output);
        write_file(output_file, output);

        return true;
    }

    template<class T>
    void make_input(std::vector<T> &input) {
        input = std::vector<T>(inputs*outputs, value);
        if (value)
            return;

        for (int i=0; i<outputs; i++) {
            int n = 0;
            T v = 0;
            for (int j=0; j<inputs; j++, n++, v++) {
                input[i*inputs + n] = rand ? rd : v;
            }
        }
    }

private:
    int inputs;
    int outputs;
};

class make_img_param_t: public param_t {
public:
    make_img_param_t(): param_t("make-img", "make an img with specified value") {
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--imgh", img_h, "the height of img")->required();
        sub->add_option("--same-conv", same_conv, "padding by same conv");
        sub->add_option("--channel", channel, "the channel of img, default 1");
        sub->add_flag("--fm", for_fm, "for feature_map");
        sub->add_flag("--fm-inc-one-by-one", fm_inc_one_by_one, "feature_map inc one by one");
    }

    bool run() {
        if (use_float)
            _run<float>();
        else
            _run<uint32_t>();

        return true;
    }

    template<class T>
    void _run() {
        std::vector<T> input;
        std::vector<T> output;
        make_input(input);
        
        if (for_fm) {
            feature_maps fms(dim, img_h, channel, same_conv);
            fms.format(input, output);
            write_file(output_file, output);
        } else {
            write_file(output_file, input);
        }
    }

    template<class T>
    void make_input(std::vector<T> &input) {
        input = std::vector<T>(channel*img_h*img_h, value);
        if (value)
            return;

        T v = 0.0;
        int n = 0;

        if (for_fm) {
            for (int k=0; k<channel; k++) {
                v = 0;
                for (int i=0; i<img_h; i++) {
                    for (int j=0; j<img_h; j++) {
                        input[n++] = rand ? rd : (fm_inc_one_by_one ? (k+1) : v++);
                    }
                }
            }
        } else {
            for (int i=0; i<img_h; i++) {
                for (int j=0; j<img_h; j++) {
                    v++;
                    for (int k=0; k<channel; k++) {
                        input[n++] = rand ? rd : v;
                    }
                }
            }
        }
    }

private:
    int dim;
    int img_h;
    int channel = 1;
    uint32_t same_conv = 0;
    bool use_float = false;
    bool for_fm = false;
    bool fm_inc_one_by_one = false;
};

int main(int argc, char *argv[])
{
    std::vector<std::shared_ptr<param_t> > params = {
        std::make_shared<test_param_t>(),
        std::make_shared<make_weight_param_t>(),
        std::make_shared<make_bias_param_t<bias> >("bias"),
        std::make_shared<make_bias_param_t<fc_bias> >("fcbias"),
        std::make_shared<make_fcw_param_t<conv_fcw> >("convfcw"),
        std::make_shared<make_fcw_param_t<fc_fcw> >("fcfcw"),
        std::make_shared<make_img_param_t>(),
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
