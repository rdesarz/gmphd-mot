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

#ifndef MOTLIB_MEASUREMENT_MODELS_H
#define MOTLIB_MEASUREMENT_MODELS_H

#include <eigen3/Eigen/Dense>

namespace motlib {

    class TwoDimensionalLinearMeasurementModel {
    public:
        TwoDimensionalLinearMeasurementModel(double noise_covariance) {
            // clang-format off
            m_state_transition_matrix << 1, 0, 0, 0,
                                         0, 1, 0, 0;

            m_measurement_noise << noise_covariance,                0,
                                                  0, noise_covariance;
            // clang-format on
        }

        const Eigen::Matrix<double, 2, 4>& state_transition() {
            return m_state_transition_matrix;
        }

        const Eigen::Matrix2d& process_noise() { return m_measurement_noise; }

    private:
        Eigen::Matrix<double, 2, 4> m_state_transition_matrix;
        Eigen::Matrix2d m_measurement_noise;
    };

}// namespace motlib

#endif// MOTLIB_MEASUREMENT_MODELS_H
