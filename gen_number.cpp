/* ===================================================
 * Copyright (C) 2018 speed-clouds All Right Reserved.
 *      Author: chenshuangping@speed-clouds.com
 *    Filename: 1.cpp
 *     Created: 2018-04-20 00:12
 * Description:
 * ===================================================
 */
#include "kx/kx_file.h"

int main(int argc, char *argv[])
{
    std::vector<int> a = { 0xa001, 0xa002, 0xa003, 0xa004 };

    kx::write_file("inc.bin", a);

    return 0;
}

