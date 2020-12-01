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

// motlib
#include <motlib/intensity.hpp>
#include <motlib/mixture_reductions.hpp>

namespace {
    using namespace motlib;

    class TestPostProcessorFixture : public ::testing::Test {
    protected:
        TestPostProcessorFixture() {
            // clang-format off
            component_0.weight() = 1.;
            component_0.mean() << 1., 1., 0.5, 0.5;
            component_0.covariance() << 1.0,   0,   0,   0,
                                          0, 1.0,   0,   0,
                                          0,   0, 1.0,   0,
                                          0,   0,   0, 1.0;

            component_1.weight() = 1e-6;
            component_1.mean() << 0.5, 0.5, 0.2, 0.2;
            component_1.covariance() << 1.0,   0,   0,   0,
                                        0, 1.0,   0,   0,
                                        0,   0, 1.0,   0,
                                        0,   0,   0, 1.0;

            component_2.weight() = 0.1;
            component_2.mean() << 0.5, 0.5, 0.2, 0.2;
            component_2.covariance() << 1.0,   0,   0,   0,
                                        0, 1.0,   0,   0,
                                        0,   0, 1.0,   0,
                                        0,   0,   0, 1.0;

            component_3.weight() = 0.1;
            component_3.mean() << 1., 1., 1., 1.;
            component_3.covariance() << 1.0,   0,   0,   0,
                                        0, 1.0,   0,   0,
                                        0,   0, 1.0,   0,
                                        0,   0,   0, 1.0;
            component_4.weight() = 0.1;
            component_4.mean() << 2., 2., 2., 2.;
            component_4.covariance() << 1.0,   0,   0,   0,
                    0, 1.0,   0,   0,
                    0,   0, 1.0,   0,
                    0,   0,   0, 1.0;
            // clang-format on
        }

        GaussianMixtureComponent4d component_0;
        GaussianMixtureComponent4d component_1;
        GaussianMixtureComponent4d component_2;
        GaussianMixtureComponent4d component_3;
        GaussianMixtureComponent4d component_4;
    };
}// namespace

TEST_F(TestPostProcessorFixture, TestPruning) {
    Intensity4d intensity;
    intensity.components.push_back(component_2);
    intensity.components.push_back(component_1);
    intensity.components.push_back(component_1);

    pruning(intensity, 1e-5);

    GTEST_ASSERT_EQ(intensity.components.size(), 1);
    ASSERT_DOUBLE_EQ(intensity.components.front().weight(), 0.1);
}

TEST_F(TestPostProcessorFixture, TestMergeScoreOfTwoComponents) {
    const double expected_score = 0.68;

    const auto score = motlib::computeMergeScore(component_0, component_1);

    ASSERT_DOUBLE_EQ(score, expected_score);
}

TEST_F(TestPostProcessorFixture, TestMergeThreeEqualsComponents) {
    aligned_vec_t<GaussianMixtureComponent4d> components;
    components.push_back(component_2);
    components.push_back(component_2);
    components.push_back(component_2);

    const auto merged_components = motlib::mergeComponents(components);

    ASSERT_DOUBLE_EQ(merged_components.weight(), 0.3);
    GTEST_ASSERT_EQ(merged_components.mean(), component_2.mean());
    GTEST_ASSERT_EQ(merged_components.covariance(), component_2.covariance());
}

TEST_F(TestPostProcessorFixture, TestMergeTwoDifferentComponents) {
    aligned_vec_t<GaussianMixtureComponent4d> components;
    components.push_back(component_3);
    components.push_back(component_4);

    const auto merged_components = motlib::mergeComponents(components);

    GaussianMixtureComponent4d::MeanType expected_mean;
    expected_mean << 1.5, 1.5, 1.5, 1.5;
    ASSERT_DOUBLE_EQ(merged_components.mean()(0), expected_mean(0));
    ASSERT_DOUBLE_EQ(merged_components.mean()(1), expected_mean(1));
    ASSERT_DOUBLE_EQ(merged_components.mean()(2), expected_mean(2));
    ASSERT_DOUBLE_EQ(merged_components.mean()(3), expected_mean(3));
}

TEST_F(TestPostProcessorFixture, TestMergeIntensityTwoCloseComponents) {
    Intensity4d intensity;
    intensity.components.push_back(component_2);
    intensity.components.back().weight() = 0.6;
    intensity.components.push_back(component_2);
    intensity.components.back().weight() = 0.3;
    intensity.components.push_back(component_0);
    intensity.components.back().weight() = 0.1;

    motlib::merging(intensity, 0.5);

    ASSERT_DOUBLE_EQ(intensity.components.size(), 2);
    ASSERT_DOUBLE_EQ(intensity.components[0].weight(), 0.9);
}
