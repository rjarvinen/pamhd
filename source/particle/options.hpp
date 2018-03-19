/*
Handles options of particle part of PAMHD.

Copyright 2016, 2017 Ilja Honkonen
Copyright 2018 Finnish Meteorological Institute
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
	double
		boltzmann = 1.38064852e-23,
		save_n = -1,
		gyroperiod_time_step_factor = 1,
		flight_time_step_factor = 1;
	size_t min_particles = 0;

	void set(const rapidjson::Value& object) {
		using std::isnormal;

		if (not object.HasMember("save-particle-n")) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON data doesn't have a save-particle-n key."
			);
		}
		const auto& save_n_json = object["save-particle-n"];
		if (not save_n_json.IsNumber()) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON item save-particle-n is not a number."
			);
		}
		save_n = save_n_json.GetDouble();

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

		if (not object.HasMember("minimum-particles")) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON data doesn't have a minimum-particles key."
			);
		}
		const auto& min_particles_json = object["minimum-particles"];
		if (not min_particles_json.IsNumber()) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON item minimum-particles is not a number."
			);
		}
		min_particles = min_particles_json.GetUint();

		if (not object.HasMember("particle-temp-nrj-ratio")) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON data doesn't have a particle-temp-nrj-ratio key."
			);
		}
		const auto& boltzmann_json = object["particle-temp-nrj-ratio"];
		if (not boltzmann_json.IsNumber()) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON item particle-temp-nrj-ratio is not a number."
			);
		}
		boltzmann = object["particle-temp-nrj-ratio"].GetDouble();
		if (not isnormal(boltzmann) or boltzmann < 0) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "Invalid particle temperature to energy ratio: " + std::to_string(boltzmann)
				+ ", should be > 0"
			);
		}

		if (not object.HasMember("gyroperiod-time-step-factor")) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON data doesn't have a gyroperiod-time-step-factor key."
			);
		}
		const auto& gyroperiod_json = object["gyroperiod-time-step-factor"];
		if (not gyroperiod_json.IsNumber()) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON item gyroperiod-time-step-factor is not a number."
			);
		}
		gyroperiod_time_step_factor = gyroperiod_json.GetDouble();

		if (not object.HasMember("flight-time-step-factor")) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON data doesn't have a flight-time-step-factor key."
			);
		}
		const auto& flight_time_json = object["flight-time-step-factor"];
		if (not flight_time_json.IsNumber()) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "JSON item flight-time-step-factor is not a number."
			);
		}
		flight_time_step_factor = flight_time_json.GetDouble();
	}
};

}} // namespaces


#endif // ifndef PAMHD_PARTICLE_OPTIONS_HPP
