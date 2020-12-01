/**
 * Copyright (c) Romain Desarzens
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Gtest
#include "gtest/gtest.h"

// eigen
#include <eigen3/Eigen/Dense>

// motlib
#include <motlib/dynamic_models.hpp>

using namespace motlib;

namespace {

    struct TestTimer {
        double getElapsedTime() const { return 2.; }
    };

};// namespace


TEST(TestDynamicModel, TestConstantVelocityModelWithTimer) {
    TestTimer timer;
    ConstantVelocity2DDynamicModel dynamic_model(timer);

    auto result = dynamic_model.state_transition();

    Eigen::Matrix4d expected;
    // clang-format off
    expected << 1, 0, 2, 0,
            0, 1, 0, 2,
            0, 0, 1, 0,
            0, 0, 0, 1;
    GTEST_ASSERT_EQ(result, expected);
}
