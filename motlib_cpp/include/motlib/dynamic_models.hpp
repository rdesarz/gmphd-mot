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

#ifndef MOTLIB_DYNAMIC_MODEL_IMPL_H
#define MOTLIB_DYNAMIC_MODEL_IMPl_H

#include <eigen3/Eigen/Dense>

namespace motlib {

    template<typename Timer>
    class ConstantVelocity2DDynamicModel {
    public:
        explicit ConstantVelocity2DDynamicModel(const Timer &timer)
            : m_timer{timer} {
            // clang-format off
            auto delta_t = m_timer.getElapsedTime();
            m_state_transition_matrix << 1, 0, delta_t,       0,
                                         0, 1,       0, delta_t,
                                         0, 0,       1,       0,
                                         0, 0,       0,       1;

            m_process_noise <<
            std::pow(delta_t, 4)/4.,                     0., std::pow(delta_t, 3)/2,                        0,
                                 0., std::pow(delta_t, 4)/4,                     0., std::pow(delta_t, 3) / 2,
            std::pow(delta_t, 3)/2.,                     0.,   std::pow(delta_t, 2),                       0.,
                                 0., std::pow(delta_t, 3)/2,                     0.,     std::pow(delta_t, 2);
            // clang-format on
        }

        const Eigen::Matrix4d &state_transition() {
            return m_state_transition_matrix;
        }

        const Eigen::Matrix4d &process_noise() { return m_process_noise; }

    private:
        Eigen::Matrix4d m_state_transition_matrix;
        Eigen::Matrix4d m_process_noise;
        const Timer &m_timer;
    };

}// namespace motlib

#endif// MOTLIB_DYNAMIC_MODEL_IMPl_H
