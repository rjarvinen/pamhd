/*
Handles divergence of magnetic field removal options of PAMHD.

Copyright 2017 Ilja Honkonen
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

#ifndef PAMHD_DIV_OPTIONS_HPP
#define PAMHD_DIV_OPTIONS_HPP


#include "cmath"
#include "string"

#include "rapidjson/document.h"


namespace pamhd {
namespace divergence {


struct Options
{
	Options() = default;
	Options(const Options& other) = default;
	Options(Options&& other) = default;

	Options(const rapidjson::Value& object)
	{
		this->set(object);
	};


	size_t
		poisson_iterations_max = 1000,
		poisson_iterations_min = 0;
	double
		remove_n = -1,
		poisson_norm_stop = 1e-15,
		poisson_norm_increase_max = 10;

	void set(const rapidjson::Value& object) {
		using std::to_string;

		if (not object.HasMember("remove-div-B-n")) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + to_string(__LINE__) + "): "
				+ "JSON data doesn't have a remove-div-B-n key."
			);
		}
		remove_n = object["remove-div-B-n"].GetDouble();

		if (not object.HasMember("poisson-norm-stop")) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + to_string(__LINE__) + "): "
				+ "JSON data doesn't have a poisson-norm-stop key."
			);
		}
		poisson_norm_stop = object["poisson-norm-stop"].GetDouble();
		if (poisson_norm_stop <= 0) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + to_string(__LINE__) + "): "
				+ "poisson-norm-stop must be > 0 but is "
				+ to_string(poisson_norm_stop)
			);
		}

		if (not object.HasMember("poisson-norm-increase-max")) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + to_string(__LINE__) + "): "
				+ "JSON data doesn't have a poisson-norm-increase-max key."
			);
		}
		poisson_norm_increase_max = object["poisson-norm-increase-max"].GetDouble();
		if (poisson_norm_increase_max <= 1) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + to_string(__LINE__) + "): "
				+ "poisson-norm-increase-max must be > 1 but is "
				+ to_string(poisson_norm_increase_max)
			);
		}
	}
};

}} // namespaces


#endif // ifndef PAMHD_DIV_OPTIONS_HPP
