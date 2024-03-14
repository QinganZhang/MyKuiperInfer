/**
 * @file tick.hpp
 * @brief 
 * TICK(foo) 将创建一个名为 bench_foo 的变量，它的值为调用该宏时的当前时间。
 * 调用 TOCK(x) 将打印出以秒为单位的时间间隔，并显示时间间隔的名称。
*/

#ifndef KUIPER_INFER_INCLUDE_TICK_HPP_
#define KUIPER_INFER_INCLUDE_TICK_HPP_
#include <chrono>
#include <iostream>

#ifndef __ycm__
    #define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
    #define TOCK(x)                                                     \
    printf("%s: %lfs\n", #x,                                          \
            std::chrono::duration_cast<std::chrono::duration<double>>( \
                std::chrono::steady_clock::now() - bench_##x)          \
                .count());
#else
    #define TICK(x)
    #define TOCK(x)
#endif

#endif  // KUIPER_INFER_INCLUDE_TICK_HPP_