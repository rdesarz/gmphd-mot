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

#ifndef MOTLIB_BIRTH_MODELS_H
#define MOTLIB_BIRTH_MODELS_H

#include <eigen3/Eigen/Dense>

#include "intensity.hpp"

namespace motlib {

    template<typename T>
    struct RectangularFovSideAppearanceBirthModel {
        RectangularFovSideAppearanceBirthModel(T x_min, T x_max, T y_min,
                                               T y_max, T vel) {
            // Create a squared field of view with possible appeareance on the side of it
            for (int i = 0; i < static_cast<std::size_t>(y_max - y_min); ++i) {
                // clang-format off
                m_intensity.components.push_back(motlib::GaussianMixtureComponent4d{});
                m_intensity.components.back().weight() = 0.1;
                m_intensity.components.back().mean() << x_min, y_min + static_cast<double>(i), vel, 0.;
                m_intensity.components.back().covariance() << 0.5, 0, 0, 0,
                                                                0, 0.5, 0, 0,
                                                                0, 0, 0.1, 0,
                                                                0, 0, 0, 0.1;

                m_intensity.components.push_back(motlib::GaussianMixtureComponent4d{});
                m_intensity.components.back().weight() = 0.1;
                m_intensity.components.back().mean() << x_max, y_min + static_cast<double>(i), -vel, 0.;
                m_intensity.components.back().covariance() << 0.5, 0, 0, 0,
                                                                0, 0.5, 0, 0,
                                                                0, 0, 0.1, 0,
                                                                0, 0, 0, 0.1;
                // clang-format on
            }
        }

        const motlib::Intensity4d& intensity() const { return m_intensity; }

        motlib::Intensity4d m_intensity;
    };
}// namespace motlib

#endif// MOTLIB_BIRTH_MODELS_H
