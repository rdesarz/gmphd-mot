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
#include <motlib/filter.hpp>
#include <motlib/intensity.hpp>

namespace {
    using namespace motlib;

    struct TestDynamicModel {
        TestDynamicModel() {
            // clang-format off
            m_state_transition_matrix << 1, 0, 1, 0,
                                         0, 1, 0, 1,
                                         0, 0, 1, 0,
                                         0, 0, 0, 1;

            m_process_noise << 0.05, 0.05, 0.05, 0.05,
                               0.05, 0.05, 0.05, 0.05,
                               0.05, 0.05, 0.05, 0.05,
                               0.05, 0.05, 0.05, 0.05;
            // clang-format on
        }

        [[nodiscard]] const Eigen::Matrix4d &state_transition() const {
            return m_state_transition_matrix;
        }

        [[nodiscard]] const Eigen::Matrix4d &process_noise() const {
            return m_process_noise;
        }

        Eigen::Matrix4d m_state_transition_matrix;
        Eigen::Matrix4d m_process_noise;
    };

    class TestMeasurementModel {
    public:
        using MeasurementType = Eigen::Matrix<double, 2, 1>;

        TestMeasurementModel() {
            // clang-format off
            m_state_transition_matrix << 1, 0, 0, 0,
                    0, 1, 0, 0;

            m_measurement_noise << 1, 0,
                    0, 1;
            // clang-format on
        }

        const Eigen::Matrix<double, 2, 4> &state_transition() {
            return m_state_transition_matrix;
        }

        const Eigen::Matrix2d &measurement_noise() {
            return m_measurement_noise;
        }

    private:
        Eigen::Matrix<double, 2, 4> m_state_transition_matrix;
        Eigen::Matrix2d m_measurement_noise;
    };

    struct TestBirthModel {
        TestBirthModel() {

            m_intensity.components.push_back(GaussianMixtureComponent4d{});
            // clang-format off
            m_intensity.components.back().weight() = 1.;
            m_intensity.components.back().mean() << 1., 1., 0.5, 0.5;
            m_intensity.components.back().covariance() << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;
            // clang-format on
        }

        const auto &intensity() { return m_intensity; }

        Intensity4d m_intensity;
    };


    class TestFilterFixture : public ::testing::Test {
    protected:
        TestFilterFixture()
            : filter{&dynamic_model, &measurement_model, &birth_model,
                     prob_survival,  prob_detection,     pruning_threshold,
                     merge_threshold} {}

        double prob_survival = 0.95;
        double prob_detection = 0.5;
        double pruning_threshold = 0.9;
        double merge_threshold = 6.0;
        TestDynamicModel dynamic_model;
        TestBirthModel birth_model;
        TestMeasurementModel measurement_model;
        GMPhdFilter<TestDynamicModel, TestMeasurementModel, TestBirthModel> filter;
    };
}// namespace


// TEST_F(TestFilterFixture,
//        TestCallsPredictTwiceAddABirthComponentAndApplyModelOnIt) {
//     filter.predict();
//     filter.predict();
//
//     double expected_weight = 0.95;
//     Eigen::Vector4d expected_mean;
//     // clang-format off
//     expected_mean << 1.5,
//             1.5,
//             0.5,
//             0.5;
//     Eigen::Matrix4d expected_covariance;
//     expected_covariance << 2.05, 0.05, 1.05, 0.05,
//             0.05, 2.05, 0.05, 1.05,
//             1.05, 0.05, 1.05, 0.05,
//             0.05, 1.05, 0.05, 1.05;
//     // clang-format on
//
//     GTEST_ASSERT_EQ(filter.currentIntensity().components.front().weight(),
//                     expected_weight);
//     GTEST_ASSERT_EQ(filter.currentIntensity().components.front().mean(),
//                     expected_mean);
//     GTEST_ASSERT_EQ(filter.currentIntensity().components.front().covariance(),
//                     expected_covariance);
//     GTEST_ASSERT_EQ(filter.currentIntensity().components.size(), 2);
// }

TEST_F(TestFilterFixture, TestCallsPredictOnceAddABirthComponent) {
    filter.predict();

    GTEST_ASSERT_EQ(filter.currentIntensity().components.front().weight(),
                    birth_model.m_intensity.components.front().weight());
    GTEST_ASSERT_EQ(filter.currentIntensity().components.front().mean(),
                    birth_model.m_intensity.components.front().mean());
    GTEST_ASSERT_EQ(filter.currentIntensity().components.front().covariance(),
                    birth_model.m_intensity.components.front().covariance());
    GTEST_ASSERT_EQ(filter.currentIntensity().components.size(), 1);
}

TEST_F(TestFilterFixture, TestCallsUpdateIntensity) {
    // clang-format off
    Intensity4d predicted_intensity;
    // Add birth component
    predicted_intensity.components.emplace_back(GaussianMixtureComponent4d{});
    predicted_intensity.components.back().weight() = 0.1;
    predicted_intensity.components.back().mean() << 2.5, 2.5, -0.1, 0.1;
    predicted_intensity.components.back().covariance() << 0.5, 0, 0, 0,
            0, 0.5, 0, 0,
            0, 0, 0.1, 0,
            0, 0, 0, 0.1;

    predicted_intensity.components.emplace_back(GaussianMixtureComponent4d{});
    predicted_intensity.components.back().weight() = 0.8;
    predicted_intensity.components.back().mean() << -1.0, 2.5, 0.2, 0.1;
    predicted_intensity.components.back().covariance() << 0.5, 0, 0, 0,
            0, 0.5, 0, 0,
            0, 0, 0.1, 0,
            0, 0, 0, 0.1;

    predicted_intensity.components.emplace_back(GaussianMixtureComponent4d{});
    predicted_intensity.components.back().weight() = 0.8;
    predicted_intensity.components.back().mean() << 1.0, 2.5, 0.2, 0.1;
    predicted_intensity.components.back().covariance() << 0.5, 0, 0, 0,
            0, 0.5, 0, 0,
            0, 0, 0.1, 0,
            0, 0, 0, 0.1;

    aligned_vec_t<Eigen::Vector2d> measurements;
    measurements.emplace_back(-1.0, 2.5);
    measurements.emplace_back(1.2, 2.6);
    measurements.emplace_back(2.0, 4.0);
    // clang-format on

    auto updated_intensity = updateIntensity(measurements, predicted_intensity,
                                             prob_detection, measurement_model);

    Eigen::Matrix2d mat;
    mat << 1.5, 0., 0., 1.5;
    auto weight_1 = sampleMultivariateNormalDistribution(
            measurements[0], Eigen::Vector2d(2.5, 2.5), mat);
    auto weight_2 = sampleMultivariateNormalDistribution(
            measurements[0], Eigen::Vector2d(-1.0, 2.5), mat);
    auto weight_3 = sampleMultivariateNormalDistribution(
            measurements[0], Eigen::Vector2d(1.0, 2.5), mat);

    weight_1 = weight_1 * 0.1 * prob_detection;
    weight_2 = weight_2 * 0.8 * prob_detection;
    weight_3 = weight_3 * 0.8 * prob_detection;

    auto total_weight = weight_1 + weight_2 + weight_3;

    weight_1 = weight_1 / total_weight;
    weight_2 = weight_2 / total_weight;
    weight_3 = weight_3 / total_weight;

    auto weight_4 = sampleMultivariateNormalDistribution(
            measurements[1], Eigen::Vector2d(2.5, 2.5), mat);
    auto weight_5 = sampleMultivariateNormalDistribution(
            measurements[1], Eigen::Vector2d(-1.0, 2.5), mat);
    auto weight_6 = sampleMultivariateNormalDistribution(
            measurements[1], Eigen::Vector2d(1.0, 2.5), mat);

    weight_4 *= 0.1 * prob_detection;
    weight_5 *= 0.8 * prob_detection;
    weight_6 *= 0.8 * prob_detection;

    total_weight = weight_4 + weight_5 + weight_6;

    weight_4 /= total_weight;
    weight_5 /= total_weight;
    weight_6 /= total_weight;

    auto weight_7 = sampleMultivariateNormalDistribution(
            measurements[2], Eigen::Vector2d(2.5, 2.5), mat);
    auto weight_8 = sampleMultivariateNormalDistribution(
            measurements[2], Eigen::Vector2d(-1.0, 2.5), mat);
    auto weight_9 = sampleMultivariateNormalDistribution(
            measurements[2], Eigen::Vector2d(1.0, 2.5), mat);

    weight_7 *= 0.1 * prob_detection;
    weight_8 *= 0.8 * prob_detection;
    weight_9 *= 0.8 * prob_detection;

    total_weight = weight_7 + weight_8 + weight_9;

    weight_7 /= total_weight;
    weight_8 /= total_weight;
    weight_9 /= total_weight;


    GTEST_ASSERT_EQ(updated_intensity.components[3].weight(), weight_1);
    GTEST_ASSERT_EQ(updated_intensity.components[4].weight(), weight_2);
    GTEST_ASSERT_EQ(updated_intensity.components[5].weight(), weight_3);
    GTEST_ASSERT_EQ(updated_intensity.components[6].weight(), weight_4);
    GTEST_ASSERT_EQ(updated_intensity.components[7].weight(), weight_5);
    GTEST_ASSERT_EQ(updated_intensity.components[8].weight(), weight_6);
    GTEST_ASSERT_EQ(updated_intensity.components[9].weight(), weight_7);
    GTEST_ASSERT_EQ(updated_intensity.components[10].weight(), weight_8);
    GTEST_ASSERT_EQ(updated_intensity.components[11].weight(), weight_9);



    GTEST_ASSERT_EQ(updated_intensity.components.size(), 12);
}

TEST_F(TestFilterFixture, TestCallsUpdateStep) {
    // clang-format off
    aligned_vec_t<Eigen::Vector2d> measurements;
    measurements.emplace_back(-1.0, 2.5);
    measurements.emplace_back(1.2, 2.6);
    measurements.emplace_back(2.0, 4.0);
    // clang-format on

    filter.update(measurements);
}