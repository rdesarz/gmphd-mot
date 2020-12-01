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

#ifndef MOTLIB_TIMER_HPP
#define MOTLIB_TIMER_HPP

#include <chrono>

namespace motlib {

    class Timer {
    public:
        Timer()
            : m_last_tick_timepoint{std::chrono::steady_clock::now()},
              m_elapsed_time{} {}

        [[nodiscard]] double getElapsedTime() const {
            return m_elapsed_time.count();
        }

        void tick() {
            using namespace std::chrono;

            const auto current_timepoint = steady_clock::now();
            m_elapsed_time = duration_cast<duration<double>>(
                    current_timepoint - m_last_tick_timepoint);
            m_last_tick_timepoint = current_timepoint;
        }

    private:
        std::chrono::steady_clock::time_point m_last_tick_timepoint;
        std::chrono::duration<double> m_elapsed_time;
    };

}// namespace motlib

#endif// MOTLIB_BIRTH_MODEL_HPP
