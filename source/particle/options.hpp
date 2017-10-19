/*
Handles options of particle part of PAMHD.

Copyright 2016, 2017 Ilja Honkonen
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of their contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef PAMHD_PARTICLE_OPTIONS_HPP
#define PAMHD_PARTICLE_OPTIONS_HPP


#include "cmath"
#include "string"

#include "rapidjson/document.h"


namespace pamhd {
namespace particle {


struct Options
{
	Options() = default;
	Options(const Options& other) = default;
	Options(Options&& other) = default;

	Options(const rapidjson::Value& object)
	{
		this->set(object);
	};

	std::string solver = "rkf78";
	double save_n = -1;

	void set(const rapidjson::Value& object) {
		using std::isnormal;

		if (not object.HasMember("save-particle-n")) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON data doesn't have a save-particle-n key."
			);
		}
		save_n = object["save-particle-n"].GetDouble();

		if (not object.HasMember("solver-particle")) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON data doesn't have a solver-particle key."
			);
		}
		solver = object["solver-particle"].GetString();
		if (
			solver != "euler"
			and solver != "midpoint"
			and solver != "rk4"
			and solver != "rkck54"
			and solver != "rkf78"
		) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "Invalid particle solver: " + solver
				+ ", should be one of euler, (modified) midpoint, rk4 (runge_kutta4), rkck54 (runge_kutta_cash_karp54), rkf78 (runge_kutta_fehlberg78), see http://www.boost.org/doc/libs/release/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html#boost_numeric_odeint.odeint_in_detail.steppers.stepper_overview"
			);
		}
	}
};

}} // namespaces


#endif // ifndef PAMHD_PARTICLE_OPTIONS_HPP
