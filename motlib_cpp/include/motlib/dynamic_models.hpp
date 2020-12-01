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

#ifndef MOTLIB_DYNAMIC_MODELS_H
#define MOTLIB_DYNAMIC_MODELS_H

#include <eigen3/Eigen/Dense>

#include "timer.hpp"

namespace motlib {

    template<typename T>
    struct TwoDimensionalCVDynamicModel {
        TwoDimensionalCVDynamicModel(const motlib::Timer& timer,
                                     T process_noise)
            : m_timer{timer},
              m_process_noise_std_dev{process_noise} {
            m_state_transition_matrix = Eigen::Matrix4d::Identity();
            m_process_noise = Eigen::Matrix4d::Identity();
        }

        const Eigen::Matrix<T, 4, 4>& state_transition() {
            m_state_transition_matrix(0, 2) = m_timer.getElapsedTime();
            m_state_transition_matrix(1, 3) = m_timer.getElapsedTime();
            return m_state_transition_matrix;
        }

        const Eigen::Matrix<T, 4, 4>& process_noise() {
            const auto delta_t = m_timer.getElapsedTime();
            const double squared_noise_std =
                    std::pow(m_process_noise_std_dev, 2);

            m_process_noise(0, 0) =
                    std::pow(delta_t, 4) / 4 * squared_noise_std;
            m_process_noise(0, 2) =
                    std::pow(delta_t, 3) / 2 * squared_noise_std;
            m_process_noise(1, 1) =
                    std::pow(delta_t, 4) / 4 * squared_noise_std;
            m_process_noise(1, 3) =
                    std::pow(delta_t, 3) / 2 * squared_noise_std;
            m_process_noise(2, 0) =
                    std::pow(delta_t, 3) / 2 * squared_noise_std;
            m_process_noise(2, 2) = std::pow(delta_t, 2) * squared_noise_std;
            m_process_noise(3, 1) =
                    std::pow(delta_t, 3) / 2 * squared_noise_std;
            m_process_noise(3, 3) = std::pow(delta_t, 2) * squared_noise_std;

            return m_process_noise;
        }

    private:
        Eigen::Matrix<T, 4, 4> m_state_transition_matrix;
        Eigen::Matrix<T, 4, 4> m_process_noise;
        T m_process_noise_std_dev;
        const motlib::Timer& m_timer;
    };

}// namespace motlib

#endif// MOTLIB_DYNAMIC_MODELS_H
