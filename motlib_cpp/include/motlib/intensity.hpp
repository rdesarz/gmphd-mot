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

#ifndef MOTLIB_INTENSITY_H
#define MOTLIB_INTENSITY_H

// std
#include <memory>
#include <vector>

// eigen
#include <eigen3/Eigen/Dense>

namespace motlib {

    template<typename T, std::size_t Dimension>
    struct GaussianMixtureComponent {
        using UnderlyingType = T;
        using MeanType = Eigen::Matrix<UnderlyingType, Dimension, 1>;
        using CovarianceType =
                Eigen::Matrix<UnderlyingType, Dimension, Dimension>;

        UnderlyingType& weight() { return m_weight; };
        MeanType& mean() { return m_mean; };
        CovarianceType& covariance() { return m_covariance; };

        const UnderlyingType& weight() const { return m_weight; };
        const MeanType& mean() const { return m_mean; };
        const CovarianceType& covariance() const { return m_covariance; };

    private:
        UnderlyingType m_weight;
        MeanType m_mean = MeanType::Zero();
        CovarianceType m_covariance = CovarianceType::Zero();
    };

    template<typename T>
    using aligned_vec_t = std::vector<T>;

    template<typename GaussianMixtureComponent>
    struct Intensity {
        using ComponentType = GaussianMixtureComponent;

        aligned_vec_t<ComponentType> components;
    };

    using GaussianMixtureComponent2d = GaussianMixtureComponent<double, 2>;
    using GaussianMixtureComponent3d = GaussianMixtureComponent<double, 3>;
    using GaussianMixtureComponent4d = GaussianMixtureComponent<double, 4>;
    using GaussianMixtureComponent5d = GaussianMixtureComponent<double, 5>;
    using GaussianMixtureComponent6d = GaussianMixtureComponent<double, 6>;

    using Intensity2d = Intensity<GaussianMixtureComponent2d>;
    using Intensity3d = Intensity<GaussianMixtureComponent3d>;
    using Intensity4d = Intensity<GaussianMixtureComponent4d>;
    using Intensity5d = Intensity<GaussianMixtureComponent5d>;
    using Intensity6d = Intensity<GaussianMixtureComponent6d>;

}// namespace motlib

#endif//MOTLIB_INTENSITY_H
