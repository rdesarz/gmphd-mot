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

#ifndef MOTLIB_FILTER_HPP
#define MOTLIB_FILTER_HPP

#include <eigen3/Eigen/Dense>
#include <utility>

#include "intensity.hpp"
#include "mixture_reductions.hpp"
#include "update_components.hpp"

namespace motlib {

    template<typename Measurement, typename IntensityType,
             typename MeasurementModel>
    IntensityType updateIntensity(
            const aligned_vec_t<Measurement> &measurements,
            const IntensityType &predicted_intensity,
            double probability_detection, MeasurementModel &measurement_model) {
        IntensityType updated_intensity;

        // Potential non detected targets
        std::transform(predicted_intensity.components.cbegin(),
                       predicted_intensity.components.cend(),
                       std::back_inserter(updated_intensity.components),
                       [&probability_detection](const auto &component) {
                           auto non_target_component = component;
                           non_target_component.weight() =
                                   non_target_component.weight() *
                                   (1 - probability_detection);
                           return non_target_component;
                       });

        // Compute updateIntensity component
        std::vector<UpdateComponents<typename IntensityType::ComponentType>>
                update_components;
        std::transform(predicted_intensity.components.cbegin(),
                       predicted_intensity.components.cend(),
                       std::back_inserter(update_components),
                       [&measurement_model](const auto &component) {
                           return UpdateComponents(component,
                                                   measurement_model);
                       });

        // Compute updated gaussian
        for (const auto &measurement : measurements) {
            IntensityType partial_intensity;
            double total_weight = 0.;
            for (std::size_t i = 0; i < predicted_intensity.components.size();
                 ++i) {
                auto updated_gaussian = computeUpdatedGaussianComponent(
                        predicted_intensity.components[i], measurement,
                        update_components[i], probability_detection);
                total_weight = total_weight + updated_gaussian.weight();
                partial_intensity.components.push_back(updated_gaussian);
            }

            for (auto &component : partial_intensity.components) {
                component.weight() = component.weight() / total_weight;
            }

            std::copy(partial_intensity.components.cbegin(),
                      partial_intensity.components.cend(),
                      std::back_inserter(updated_intensity.components));
        }

        return updated_intensity;
    }

    template<typename DynamicModel, typename MeasurementModel,
             typename BirthModel>
    class Filter {
    public:
        Filter(DynamicModel *dynamic_model, MeasurementModel *measurement_model,
               BirthModel *birth_model, double probability_survival,
               double probability_detection, double pruning_threshold,
               double merge_threshold)
            : m_dynamic_model{dynamic_model},
              m_measurement_model{measurement_model},
              m_birth_model{birth_model},
              m_probability_survival{probability_survival},
              m_probability_detection{probability_detection},
              m_pruning_threshold{pruning_threshold},
              m_merge_threshold{merge_threshold} {}

        void predict() {
            std::transform(
                    m_intensity.components.begin(),
                    m_intensity.components.end(),
                    m_intensity.components.begin(), [this](auto &component) {
                        component.weight() =
                                component.weight() * m_probability_survival;
                        component.mean() = m_dynamic_model->state_transition() *
                                           component.mean();
                        component.covariance() =
                                m_dynamic_model->state_transition() *
                                        component.covariance() *
                                        m_dynamic_model->state_transition()
                                                .transpose() +
                                m_dynamic_model->process_noise();
                        return component;
                    });

            std::copy(m_birth_model->intensity().components.begin(),
                      m_birth_model->intensity().components.end(),
                      std::back_inserter(m_intensity.components));
        }

        // TODO: Deduce type from an other type
        template<typename Measurement>
        void update(const aligned_vec_t<Measurement> &measurements) {
            m_intensity = updateIntensity(measurements, m_intensity,
                                          m_probability_detection,
                                          *m_measurement_model);

            pruning(m_intensity, m_pruning_threshold);
            merging(m_intensity, m_merge_threshold);
        }

        [[nodiscard]] const auto &currentIntensity() const {
            return m_intensity;
        }

    private:
        DynamicModel *m_dynamic_model;
        MeasurementModel *m_measurement_model;
        BirthModel *m_birth_model;

        // The type of intensity treated by the filter has to be the same as
        // the one generated by the birth model
        using IntensityType =
                typename std::remove_const_t<std::remove_reference_t<decltype(
                        std::declval<BirthModel>().intensity())>>;

        IntensityType m_intensity;
        typename IntensityType::ComponentType::UnderlyingType
                m_probability_survival;
        typename IntensityType::ComponentType::UnderlyingType
                m_probability_detection;
        typename IntensityType::ComponentType::UnderlyingType
                m_pruning_threshold;
        typename IntensityType::ComponentType::UnderlyingType m_merge_threshold;
    };

}// namespace motlib

#endif//MOTLIB_FILTER_HPP
