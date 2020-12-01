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
#include <motlib/intensity.hpp>
#include <motlib/update_components.hpp>

namespace {
    using namespace motlib;

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


    class TestUpdateComponentsFixture : public ::testing::Test {
    protected:
        TestUpdateComponentsFixture() {
            // clang-format off
            single_measurement << 2.,
                    2.;

            component.weight() = 1.;

            component.mean() << 1.,
                    1.,
                    0.5,
                    0.5;

            component.covariance() = Eigen::MatrixXd::Identity(4, 4);

            component_2.weight() = 1.;
            component_2.mean() << 1.,
                    1.,
                    0.5,
                    0.5;
            component_2.covariance() <<  1, 2, 3, 4,
                                       5, 6, 7, 8,
                                       9, 10,11,12,
                                      13,14,15,16;
            // clang-format on
        }

        double prob_detection = 0.5;
        Eigen::Vector2d single_measurement;
        GaussianMixtureComponent4d component;
        GaussianMixtureComponent4d component_2;
        TestMeasurementModel measurement_model;
    };
}// namespace


TEST_F(TestUpdateComponentsFixture, TestComputeInnovationCovariance) {
    auto result = computeInnovationCovariance(component, measurement_model);

    // clang-format off
    Eigen::Matrix2d expected;
    expected << 2, 0,
                0, 2;
    // clang-format on


    GTEST_ASSERT_EQ(expected, result);
}

TEST_F(TestUpdateComponentsFixture, TestComputeKalmanGain) {
    auto innovation_covariance =
            computeInnovationCovariance(component_2, measurement_model);

    //Eigen::Matrix<double, 4, 2>
    Eigen::Matrix2d test =
            (measurement_model.state_transition() * component_2.covariance() *
             measurement_model.state_transition().transpose())
                    .eval();

    Eigen::Matrix2d inv_innov_cov = innovation_covariance.inverse();
    auto result = computeKalmanGain(component_2, measurement_model,
                                    innovation_covariance);

    // clang-format off
    Eigen::Matrix<double, 4, 2> expected;
    expected << 1./2., 2./7.,
                 5./2.,6./7.,
                 9./2.,10./7.,
                13./2., 14./7.;
    // clang-format on

    // GTEST_ASSERT_EQ(expected, result);
}

TEST_F(TestUpdateComponentsFixture, TestComputeMeasurementEstimate) {
    auto result = computeMeasurementEstimate(component, measurement_model);

    // clang-format off
    Eigen::Matrix<double, 2, 1> expected;
    expected << 1.,
            1.;
    // clang-format on

    GTEST_ASSERT_EQ(expected, result);
}

TEST_F(TestUpdateComponentsFixture, TestComputeUpdateComponentCovariance) {
    auto innovation_covariance =
            computeInnovationCovariance(component, measurement_model);
    auto kalman_gain = computeKalmanGain(component, measurement_model,
                                         innovation_covariance);

    auto result = computeUpdatedGaussianCovariance(component, measurement_model,
                                                   kalman_gain);

    // clang-format off
    Eigen::Matrix4d expected;
    expected << 0.5, 0., 0., 0.,
            0., 0.5, 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.;
    // clang-format on

    GTEST_ASSERT_EQ(expected, result);
}

TEST_F(TestUpdateComponentsFixture, TestComputeUpdatedComponent) {
    auto update_components = UpdateComponents(component, measurement_model);

    auto result = computeUpdatedGaussianComponent(
            component, single_measurement, update_components, prob_detection);

    // clang-format off
    double expected_weight = 1. * prob_detection * sampleMultivariateNormalDistribution(
            single_measurement, update_components.measurement_estimate,
            update_components.innovation_covariance);
    Eigen::Vector4d expected_mean;
    expected_mean << 1.5,
                     1.5,
                     0.5,
                     0.5;
    Eigen::Matrix4d expected_covariance;
    expected_covariance << 0.5, 0., 0., 0.,
                            0., 0.5, 0., 0.,
                            0., 0., 1., 0.,
                            0., 0., 0., 1.;
    // clang-format on

    ASSERT_DOUBLE_EQ(expected_weight, result.weight());
    GTEST_ASSERT_EQ(expected_mean, result.mean());
    GTEST_ASSERT_EQ(expected_covariance, result.covariance());
}
