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
    typename GaussianMixtureComponent::UnderlyingType computeMergeScore(
            const GaussianMixtureComponent& component_i,
            const GaussianMixtureComponent& component_j) {
        const typename GaussianMixtureComponent::MeanType mean_diff =
                (component_i.mean() - component_j.mean()).eval();
        return mean_diff.transpose() * component_i.covariance().inverse() *
               mean_diff;
    }

    template<typename GaussianMixtureComponent>
    GaussianMixtureComponent mergeComponents(
            const aligned_vec_t<GaussianMixtureComponent>& components) {
        GaussianMixtureComponent merged_component;

        merged_component.weight() = 0.;
        for (const auto& component : components) {
            merged_component.weight() += component.weight();
        }

        for (const auto& component : components) {
            merged_component.mean() += component.mean() * component.weight();
        }
        merged_component.mean() /= merged_component.weight();

        for (const auto& component : components) {
            auto mean_diff = merged_component.mean() - component.mean();
            merged_component.covariance() +=
                    (component.covariance() +
                     mean_diff * mean_diff.transpose()) *
                    component.weight();
        }
        merged_component.covariance() /= merged_component.weight();

        return merged_component;
    }

    template<typename Intensity>
    void merging(Intensity& posterior_intensity, double merge_threshold) {
        // Sort by weight
        std::sort(posterior_intensity.components.begin(),
                  posterior_intensity.components.end(),
                  [](const auto& rhs, const auto& lhs) {
                      return rhs.weight() > lhs.weight();
                  });

        // Lookup table to check if a component was merged
        std::vector<bool> merged(posterior_intensity.components.size(), false);

        for (std::size_t i = 0; i < posterior_intensity.components.size();
             ++i) {
            if (!merged[i]) {
                aligned_vec_t<typename Intensity::ComponentType>
                        to_be_merged;
                to_be_merged.push_back(posterior_intensity.components[i]);

                for (std::size_t j = i + 1;
                     j < posterior_intensity.components.size(); ++j) {
                    if (!merged[j] &&
                        computeMergeScore(posterior_intensity.components[i],
                                          posterior_intensity.components[j]) <
                                merge_threshold) {
                        to_be_merged.push_back(
                                posterior_intensity.components[j]);
                        merged[j] = true;
                    }
                }
                posterior_intensity.components[i] =
                        mergeComponents(to_be_merged);
            }
        }

        Intensity modified_intensity;
        for (std::size_t i = 0; i < merged.size(); ++i) {
            if (!merged[i]) {
                modified_intensity.components.push_back(
                        posterior_intensity.components[i]);
            }
        }

        posterior_intensity = modified_intensity;
    }

}// namespace motlib

#endif// MOTLIB_MIXTURE_REDUCTIONS_HPP
