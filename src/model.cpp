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

#include "CLI11.hpp"
#include "fpga_format.h"

using namespace kx;

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

class format_weight_param_t: public param_t {
public:
    format_weight_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-weight", "format weight to fpga format");
        sub->add_option("--input", input_file_, "the file to read")->required();
        sub->add_option("--output", output_file_, "the file to write")->required();
        sub->add_set("--dim", dim_, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--inputs", inputs, "input count")->required();
        sub->add_option("--outputs", outputs, "output count")->required();
    }

    bool run() {
        weight w(dim_, inputs, outputs);
        std::vector<uint32_t> output;
        return format_to_fpga(w, output, input_file_, output_file_);
    }

private:
    std::string input_file_;
    std::string output_file_;
    int dim_;
    int inputs;
    int outputs;
};

class format_convfcw_param_t: public param_t {
public:
    format_convfcw_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-convfcw", "format conv_fc weight to fpga format");
        sub->add_option("--input", input_file_, "the file to read")->required();
        sub->add_option("--output", output_file_, "the file to write")->required();
        sub->add_option("--inputs", inputs, "input count")->required();
        sub->add_option("--outputs", outputs, "output count")->required();
    }

    bool run() {
        conv_fcw w(inputs, outputs);
        std::vector<uint32_t> output;
        return format_to_fpga(w, output, input_file_, output_file_);
    }

private:
    std::string input_file_;
    std::string output_file_;
    int inputs;
    int outputs;
};

class format_fcfcw_param_t: public param_t {
public:
    format_fcfcw_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-fcfcw", "format fc_fc weight to fpga format");
        sub->add_option("--input", input_file_, "the file to read")->required();
        sub->add_option("--output", output_file_, "the file to write")->required();
        sub->add_option("--inputs", inputs, "input count")->required();
        sub->add_option("--outputs", outputs, "output count")->required();
    }

    bool run() {
        fc_fcw w(inputs, outputs);
        std::vector<uint32_t> output;
        return format_to_fpga(w, output, input_file_, output_file_);
    }

private:
    std::string input_file_;
    std::string output_file_;
    int inputs;
    int outputs;
};

class format_bias_param_t: public param_t {
public:
    format_bias_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-bias", "format bias to fpga format");
        sub->add_option("--input", input_file_, "the file to read")->required();
        sub->add_option("--output", output_file_, "the file to write")->required();
        sub->add_option("--inputs", inputs, "input count")->required();
    }

    bool run() {
        bias b(inputs);
        std::vector<uint32_t> output;
        return format_to_fpga(b, output, input_file_, output_file_);
    }

private:
    std::string input_file_;
    std::string output_file_;
    int inputs;
};

class format_fcbias_param_t: public param_t {
public:
    format_fcbias_param_t(CLI::App &app) {
        sub = app.add_subcommand("format-fcbias", "format fcbias to fpga format");
        sub->add_option("--input", input_file_, "the file to read")->required();
        sub->add_option("--output", output_file_, "the file to write")->required();
        sub->add_option("--inputs", inputs, "input count")->required();
    }

    bool run() {
        fc_bias b(inputs);
        std::vector<uint32_t> output;
        return format_to_fpga(b, output, input_file_, output_file_);
    }

private:
    std::string input_file_;
    std::string output_file_;
    int inputs;
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
        feature_maps fms(dim, img_h, img_count);
        std::vector<uint32_t> output;
        return format_to_fpga(fms, output, input_file, output_file);
    }

private:
    std::string input_file;
    std::string output_file;
    int dim;
    int img_h;
    int img_count;
};

class make_weight_param_t: public param_t {
public:
    make_weight_param_t(CLI::App &app) {
        sub = app.add_subcommand("make-weight", "make a weight with specified value");
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--inputs", inputs, "inputs")->required();
        sub->add_option("--outputs", outputs, "outputs")->required();
        sub->add_option("--value", value, "make by value");
    }

    bool run() {
        auto input = make_input();
        write_file(output_file + ".src", input);

        weight w(dim, inputs, outputs);
        std::vector<uint32_t> output;

        w.format(input, output);
        write_file(output_file, output);

        return true;
    }

    void make_weight(weight &w, int cell, int conv, int val, std::vector<uint32_t> &output) {
        std::vector<uint32_t> input(w.conv_h());
        for (size_t i=0; i<input.size(); i++) {
            input[i] = (cell << 16) | (conv << 8) | (val == 0 ? i : val);
        }

        w.fill_conv(cell, conv, &input[0], output);
    }

    std::vector<uint32_t> make_input() {
        int size = dim*dim;
        std::vector<uint32_t> input(outputs*inputs*size);

        int n = 0;
        for (int i=0; i<outputs; i++) {
            for (int j=0; j<inputs; j++) {
                for (int k=0; k<size; k++) {
                    input[n++] = (i << 16) | (j << 8) | k;
                }
            }
        }

        return input;
    }

private:
    std::string output_file;
    int dim;
    int inputs;
    int outputs;
    uint32_t value = 0;
};

template<class T>
class make_bias_param_t: public param_t {
public:
    make_bias_param_t(CLI::App &app, const std::string &name) {
        sub = app.add_subcommand(std::string("make-") + name, "make a bias with specified value");
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_option("--inputs", inputs, "inputs")->required();
        sub->add_option("--value", value, "make by value");
    }

    bool run() {
        auto input = make_input();
        write_file(output_file + ".src", input);

        T b(inputs);
        std::vector<uint32_t> output;

        b.format(input, output);
        write_file(output_file, output);

        return true;
    }

    std::vector<uint32_t> make_input() {
        std::vector<uint32_t> input(inputs, value);

        if (value == 0) {
            for (size_t i=0; i<input.size(); i++) {
                input[i] = i;
            }
        }

        return input;
    }

private:
    std::string output_file;
    int inputs;
    uint32_t value = 0;
};

template<class T>
class make_fcw_param_t: public param_t {
public:
    make_fcw_param_t(CLI::App &app, const std::string &name) {
        sub = app.add_subcommand(std::string("make-")+name, "make a fc weight with specified value");
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_option("--inputs", inputs, "inputs")->required();
        sub->add_option("--outputs", outputs, "outputs")->required();
        sub->add_option("--value", value, "make by value");
    }

    bool run() {
        auto input = make_input();
        write_file(output_file + ".src", input);

        T w(inputs, outputs);
        std::vector<uint32_t> output;

        w.format(input, output);
        write_file(output_file, output);

        return true;
    }

    std::vector<uint32_t> make_input() {
        std::vector<uint32_t> input(inputs*outputs, value);

        if (value == 0) {
            int n = 0;
            for (int i=0; i<outputs; i++) {
                for (int j=0; j<inputs; j++) {
                    input[n++] = (i << 8) | j;
                }
            }
        }

        return input;
    }

private:
    std::string output_file;
    int inputs;
    int outputs;
    int value = 0;
};

class make_img_param_t: public param_t {
public:
    make_img_param_t(CLI::App &app) {
        sub = app.add_subcommand("make-img", "make an img with specified value");
        sub->add_set("--dim", dim, {1,3,5,7}, "the dim of conv")->required();
        sub->add_option("--output", output_file, "the file to write")->required();
        sub->add_option("--inputs", inputs, "how many imgs")->required();
        sub->add_option("--imgh", img_h, "the height of img")->required();
        sub->add_option("--value", value, "make by value");
        sub->add_option("--same-conv", same_conv, "padding by same conv");
        sub->add_flag("-f,--float", use_float, "use float");
    }

    bool run() {
        feature_maps fms(dim, img_h, inputs, same_conv);

        if (use_float) {
            std::vector<float> input = make_inputf();
            std::vector<float> output;

            write_file(output_file + ".src", input);
            fms.format(input, output);
            write_file(output_file, output);
        } else {
            std::vector<uint32_t> input = make_input();
            std::vector<uint32_t> output;

            write_file(output_file + ".src", input);
            fms.format(input, output);
            write_file(output_file, output);
        }

        return true;
    }

    std::vector<uint32_t> make_input() {
        std::vector<uint32_t> input(img_h*img_h*inputs, value);
        if (value)
            return input;

        int v = 0;
        int n = 0;
        int x = img_h/dim;
        int y = img_h%dim;

        for (int i=0; i<inputs; i++) {
            for (int j=0; j<img_h; j++) {
                for (int k=0; k<x; k++) {
                    v++;
                    for (int d=0; d<dim; d++) {
                        input[n++] = (d+1) << 24 | v;
                    }
                }
                for (int a=0; a<y; a++) {
                    v++;
                    input[n++] = (a+1) << 24 | v;
                }
            }
        }

        return input;
    }

    std::vector<float> make_inputf() {
        std::vector<float> input(img_h*img_h*inputs, value);
        if (value)
            return input;

        float v = 0.0;
        int n = 0;
        int x = img_h/dim;
        int y = img_h%dim;

        for (int i=0; i<inputs; i++) {
            for (int j=0; j<img_h; j++) {
                for (int k=0; k<x; k++) {
                    v++;
                    for (int d=0; d<dim; d++) {
                        input[n++] = v;
                    }
                }
                for (int a=0; a<y; a++) {
                    v++;
                    input[n++] = v;
                }
            }
        }

        return input;
    }

private:
    std::string output_file;
    int dim;
    int inputs;
    int img_h;
    uint32_t same_conv = 0;
    uint32_t value = 0;
    bool use_float = false;
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
        std::make_shared<make_weight_param_t>(app),
        std::make_shared<make_bias_param_t<bias> >(app, "bias"),
        std::make_shared<make_bias_param_t<fc_bias> >(app, "fcbias"),
        std::make_shared<make_fcw_param_t<conv_fcw> >(app, "convfcw"),
        std::make_shared<make_fcw_param_t<fc_fcw> >(app, "fcfcw"),
        std::make_shared<make_img_param_t>(app),
    };

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    mkdir("src", 0755);

    for (auto &param : params) {
        if (param->ok()) {
            printf("%s %s\n", param->name().c_str(), param->run() ? "done" : "failed");
        }
    }

    return 0;
}