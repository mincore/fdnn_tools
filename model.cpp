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
        sub->add_option("--inputs", inputs_, "input count")->required();
        sub->add_option("--outputs", outputs_, "output count")->required();
    }

    bool run() {
        weight w(dim_, inputs_, outputs_);
        std::vector<float> output;
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
        std::vector<float> output;
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
        std::vector<float> output;
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
        std::vector<float> output;
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
        std::vector<float> output;
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
        feature_maps fms(dim, img_h, img_count);
        std::vector<float> output;
        return format_to_fpga(fms, output, input_file, output_file);
    }

private:
    std::string input_file;
    std::string output_file;
    int dim;
    int img_h;
    int img_count;
};

#if 0
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

        std::vector<float> conv_input(w.conv_size(), value);
        std::vector<float> output;

        printf("dim:%d inputs_:%d outputs_:%d cell:%d conv:%d value:%f\n",
                dim, inputs_, outputs_, cell, conv, value);

        if (with_index) {
            for (size_t i=0; i<conv_input.size(); i++) {
                conv_input[i] = ((int)conv_input[i] & 0xFFFFFF00) | i;
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
    float value;
    bool with_index = false;
};
#endif

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
        //std::make_shared<fill_conv_param_t>(app),
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
