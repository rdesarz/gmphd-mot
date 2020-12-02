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

#ifndef MOTLIB_MIXTURE_REDUCTIONS_HPP
#define MOTLIB_MIXTURE_REDUCTIONS_HPP

#include <eigen3/Eigen/Dense>
#include <numeric>

namespace motlib {

    template<typename Intensity>
    void pruning(Intensity& intensity, double threshold) {
        intensity.components.erase(
                std::remove_if(intensity.components.begin(),
                               intensity.components.end(),
                               [threshold](const auto& component) {
                                   return component.weight() < threshold;
                               }),
                intensity.components.end());
    }

    template<typename GaussianMixtureComponent>
    class MergeReductor {
    public:
        MergeReductor(aligned_vec_t<GaussianMixtureComponent>& gaussian_mixture,
                      typename GaussianMixtureComponent::UnderlyingType
                              merge_threshold)
            : m_gaussian_mixture{gaussian_mixture},
              m_merge_threshold{merge_threshold} {}

        void reduce() {
            // Sort by weight
            std::sort(m_gaussian_mixture.begin(), m_gaussian_mixture.end(),
                      [](const auto& gauss_rhs, const auto& gauss_lhs) {
                          return gauss_rhs.weight() > gauss_lhs.weight();
                      });

            // Lookup table to check if a component was merged
            std::vector<bool> merged(m_gaussian_mixture.size(), false);

            // Compare each component with the other one to check if they can be merged together
            for (std::size_t i = 0; i < m_gaussian_mixture.size(); ++i) {
                if (!merged[i]) {
                    std::vector<component_index_t> merged_gaussians;
                    merged_gaussians.push_back(i);

                    for (std::size_t j = i + 1; j < m_gaussian_mixture.size();
                         ++j) {
                        if (!merged[j] &&
                            computeMergeScore(i, j) < m_merge_threshold) {
                            merged_gaussians.push_back(j);
                            merged[j] = true;
                        }
                    }
                    m_gaussian_mixture[i] = mergeComponents(merged_gaussians);
                }
            }

            aligned_vec_t<GaussianMixtureComponent> reduced_mixture;
            for (std::size_t i = 0; i < merged.size(); ++i) {
                if (!merged[i]) {
                    reduced_mixture.push_back(m_gaussian_mixture[i]);
                }
            }

            m_gaussian_mixture = reduced_mixture;
        }

    private:
        using component_index_t = std::size_t;

        GaussianMixtureComponent mergeComponents(
                const std::vector<component_index_t>& indexes) {
            GaussianMixtureComponent merged_component;

            merged_component.weight() = 0.;
            for (const auto index : indexes) {
                merged_component.weight() += m_gaussian_mixture[index].weight();
            }

            for (const auto index : indexes) {
                merged_component.mean() += m_gaussian_mixture[index].mean() *
                                           m_gaussian_mixture[index].weight();
            }
            merged_component.mean() /= merged_component.weight();

            for (const auto index : indexes) {
                auto mean_diff = merged_component.mean() -
                                 m_gaussian_mixture[index].mean();
                merged_component.covariance() +=
                        (m_gaussian_mixture[index].covariance() +
                         mean_diff * mean_diff.transpose()) *
                        m_gaussian_mixture[index].weight();
            }
            merged_component.covariance() /= merged_component.weight();

            return merged_component;
        }

        typename GaussianMixtureComponent::UnderlyingType computeMergeScore(
                component_index_t i, component_index_t j) {
            const typename GaussianMixtureComponent::MeanType delta_mean =
                    (m_gaussian_mixture[i].mean() -
                     m_gaussian_mixture[j].mean())
                            .eval();
            return delta_mean.transpose() *
                   m_gaussian_mixture[i].covariance().inverse() * delta_mean;
        }

    private:
        aligned_vec_t<GaussianMixtureComponent>& m_gaussian_mixture;
        typename GaussianMixtureComponent::UnderlyingType m_merge_threshold;
    };

}// namespace motlib

#endif// MOTLIB_MIXTURE_REDUCTIONS_HPP
