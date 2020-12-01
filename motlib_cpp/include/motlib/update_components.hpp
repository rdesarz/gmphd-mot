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

#ifndef MOTLIB_UPDATE_COMPONENTS_HPP
#define MOTLIB_UPDATE_COMPONENTS_HPP

#include <eigen3/Eigen/Dense>
#include <motlib/intensity.hpp>

namespace motlib {
    template<typename DerivedMean1, typename DerivedMean2,
             typename DerivedCovariance>
    auto sampleMultivariateNormalDistribution(
            const Eigen::MatrixBase<DerivedMean1>& x,
            const Eigen::MatrixBase<DerivedMean2>& mu,
            const Eigen::MatrixBase<DerivedCovariance>& sigma) {
        double n = x.rows();
        double sqrt2pi = std::sqrt(2 * M_PI);
        double quadform = (x - mu).transpose() * sigma.inverse() * (x - mu);
        double norm =
                std::pow(sqrt2pi, -n) * std::pow(sigma.determinant(), -0.5);

        return norm * exp(-0.5 * quadform);
    }

    // TODO: check if evaluation of expression before return is mandatory.
    template<typename GaussianMixtureComponent, typename MeasurementModel>
    auto computeMeasurementEstimate(const GaussianMixtureComponent& component,
                                    MeasurementModel& measurement_model) {
        return (measurement_model.state_transition() * component.mean()).eval();
    }

    template<typename GaussianMixtureComponent, typename MeasurementModel>
    auto computeInnovationCovariance(const GaussianMixtureComponent& component,
                                     MeasurementModel& measurement_model) {
        return (measurement_model.measurement_noise() +
                measurement_model.state_transition() * component.covariance() *
                        measurement_model.state_transition().transpose())
                .eval();
    }

    template<typename GaussianMixtureComponent, typename MeasurementModel,
             typename InnovationCovariance>
    auto computeKalmanGain(const GaussianMixtureComponent& component,
                           MeasurementModel& measurement_model,
                           const InnovationCovariance& innovation_covariance) {
        return (component.covariance() *
                measurement_model.state_transition().transpose() *
                innovation_covariance.inverse())
                .eval();
    }

    template<typename GaussianMixtureComponent, typename MeasurementModel,
             typename KalmanGain>
    auto computeUpdatedGaussianCovariance(
            const GaussianMixtureComponent& component,
            MeasurementModel& measurement_model,
            const KalmanGain& kalman_gain) {
        return (component.covariance() -
                kalman_gain * measurement_model.state_transition() *
                        component.covariance())
                .eval();
    }

    template<typename GaussianMixtureComponent>
    struct UpdateComponents {
        template<typename MeasurementModel>
        UpdateComponents(const GaussianMixtureComponent& component,
                         MeasurementModel& measurement_model) {
            measurement_estimate =
                    computeMeasurementEstimate(component, measurement_model);
            innovation_covariance =
                    computeInnovationCovariance(component, measurement_model);
            kalman_gain = computeKalmanGain(component, measurement_model,
                                            innovation_covariance);
            updated_covariance = computeUpdatedGaussianCovariance(
                    component, measurement_model, kalman_gain);
        }

        // TODO: deduct dimension of matrix
        Eigen::Vector2d measurement_estimate;
        Eigen::Matrix2d innovation_covariance;
        Eigen::Matrix<double, 4, 2> kalman_gain;
        typename GaussianMixtureComponent::CovarianceType updated_covariance;
    };

    template<typename GaussianMixtureComponent, typename Derived>
    auto computeUpdatedGaussianComponent(
            const GaussianMixtureComponent& component,
            const Eigen::MatrixBase<Derived>& measurement,
            const UpdateComponents<GaussianMixtureComponent>& update_components,
            double probability_detection) {
        GaussianMixtureComponent updated_component;
        updated_component.weight() =
                component.weight() * probability_detection *
                sampleMultivariateNormalDistribution(
                        measurement, update_components.measurement_estimate,
                        update_components.innovation_covariance);

        updated_component.mean() =
                component.mean() +
                update_components.kalman_gain *
                        (measurement - update_components.measurement_estimate);

        updated_component.covariance() = update_components.updated_covariance;

        return updated_component;
    }

}// namespace motlib

#endif//MOTLIB_UPDATE_COMPONENTS_HPP
