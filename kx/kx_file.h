/* ===================================================
 * Copyright (C) 2018 speed-clouds All Right Reserved.
 *      Author: chenshuangping@speed-clouds.com
 *    Filename: kx_file.h
 *     Created: 2018-04-03 09:32
 * Description:
 * ===================================================
 */
#ifndef _KX_FILE_H
#define _KX_FILE_H

#include <stdio.h>
#include <stdarg.h>
#include <vector>
#include <string>

namespace kx {

class noncopyable {
protected:
    noncopyable () {}
    ~noncopyable () {}
private:
    noncopyable (const noncopyable &);
    noncopyable & operator = (const noncopyable &);
};

template<typename... Args>
std::string string_format(const char* fmt, Args... args)
{
    size_t size = snprintf(nullptr, 0, fmt, args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt, args...);
    return buf;
}

class file: private noncopyable {
public:
    ~file() { if (fp_) fclose(fp_); }

    bool open(const std::string &filename, const std::string &mode) {
        fp_ = fopen(filename.c_str(), mode.c_str());
        if (!fp_)
            return false;

        fseek(fp_, 0, SEEK_END);
        size_ = ftell(fp_);
        fseek(fp_, 0, SEEK_SET);
        return true;
    }

    size_t read(void *data, size_t size, off_t offset = 0) {
        if (offset > 0) {
            if (-1 == fseek(fp_, offset, SEEK_SET))
                return -1;
        }

        size_t ret = fread(data, 1, size, fp_);

        if (ferror(fp_))
            return -1;

        return ret;
    }

    size_t write(const void *data, size_t size, off_t offset = 0) {
        if (offset > 0) {
            if (-1 == fseek(fp_, offset, SEEK_SET))
                return -1;
        }
        return fwrite(data, 1, size, fp_);
    }

    size_t size() {
        return size_;
    }

private:
    size_t size_ = 0;
    FILE *fp_ = NULL;
};

static size_t read_file(const std::string &filename, void *data, size_t size, off_t offset = 0)
{
    file file;
    if (!file.open(filename, "r")) {
        return -1;
    }

    return file.read(data, size, offset);
}

template<typename T>
static size_t read_file(const std::string &filename, std::vector<T> &data, off_t offset = 0)
{
    file file;
    if (!file.open(filename, "r")) {
        return -1;
    }

    size_t size = (file.size() - offset) / sizeof(T);
    if (size == 0)
        return 0;

    data.resize(size);
    return file.read(&data[0], data.size()*sizeof(T), offset);
}

static size_t write_file(const std::string &filename, const void *data, size_t size, off_t offset = 0)
{
    file file;
    if (!file.open(filename, "w"))
        return -1;

    return file.write(data, size, offset);
}

template<typename T>
static int write_file(const std::string &filename, const std::vector<T> &data, off_t offset = 0)
{
    return write_file(filename, &data[0], data.size()*sizeof(T), offset);
}

}

#endif
