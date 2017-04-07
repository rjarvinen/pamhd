/*
Initializes particle solution of PAMHD.

Copyright 2015, 2016 Ilja Honkonen
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

#ifndef PAMHD_PARTICLE_INITIALIZE_HPP
#define PAMHD_PARTICLE_INITIALIZE_HPP


#include "algorithm"
#include "iostream"
#include "random"
#include "utility"

#include "particle/common.hpp"
#include "particle/variables.hpp"


namespace pamhd {
namespace particle {


/*!
Creates particles in given cells as defined by given initial conditions.

Returns the total number of particles created.

\param [Grid] Compatible with DCCRG (github.com/fmihpc/dccrg)
\param [Particle] Type of particles stored in a container accessed by Particles_T
\param [init_cond] Initial condition for particles obtained from user
\param [simulation_time] Time to give to init_cond when querying data
\param [cells] Cells into which to create particles
\param [random_source] Source to use when creating particle distributions
\param [particle_temp_nrj_ratio] https://en.wikipedia.org/wiki/Boltzmann_constant
\param [replace] Whether to replace/add particles in/to given cells
\param [verbose] Whether to print what is being done to stdout on MPI rank 0
*/
template<
	class Particle,
	class Particle_Mass_T,
	class Particle_Charge_Mass_Ratio_T,
	class Particle_Position_T,
	class Particle_Velocity_T,
	class Particle_ID_T,
	class Particle_Species_Mass_T,
	class Sim_Geometries,
	class Init_Cond,
	class Background_Magnetic_Field,
	class Grid,
	class Electric_Field_Getter,
	class Magnetic_Field_Getter,
	class Particles_Getter,
	class Background_Magnetic_Field_Pos_X_Face_Getter,
	class Background_Magnetic_Field_Pos_Y_Face_Getter,
	class Background_Magnetic_Field_Pos_Z_Face_Getter,
	class Boundary_Number_Density_Getter,
	class Boundary_Velocity_Getter,
	class Boundary_Temperature_Getter,
	class Boundary_Nr_Particles_Getter,
	class Boundary_Charge_To_Mass_Ratio_Getter,
	class Boundary_Species_Mass_Getter,
	class Solver_Info_Getter
> size_t initialize(
	const Sim_Geometries& geometries,
	Init_Cond& initial_conditions,
	const double simulation_time,
	const std::vector<uint64_t>& cells,
	const Background_Magnetic_Field& bg_B,
	Grid& grid,
	std::mt19937_64& random_source,
	const double particle_temp_nrj_ratio,
	const double vacuum_permeability,
	const unsigned long long int first_particle_id,
	const unsigned long long int particle_id_increase,
	const bool replace,
	const bool verbose,
	const Electric_Field_Getter Ele,
	const Magnetic_Field_Getter Mag,
	const Particles_Getter Par,
	const Background_Magnetic_Field_Pos_X_Face_Getter Bg_B_Pos_X,
	const Background_Magnetic_Field_Pos_Y_Face_Getter Bg_B_Pos_Y,
	const Background_Magnetic_Field_Pos_Z_Face_Getter Bg_B_Pos_Z,
	const Boundary_Number_Density_Getter Bdy_N,
	const Boundary_Velocity_Getter Bdy_V,
	const Boundary_Temperature_Getter Bdy_T,
	const Boundary_Nr_Particles_Getter Bdy_Nr_Par,
	const Boundary_Charge_To_Mass_Ratio_Getter Bdy_C2M,
	const Boundary_Species_Mass_Getter Bdy_SpM,
	const Solver_Info_Getter Sol_Info
) {
	if (verbose && grid.get_rank() == 0) {
		std::cout << "Setting initial particle state... ";
		std::cout.flush();
	}

	// set default state
	size_t nr_particles_created = 0;
	auto current_id_start = first_particle_id;
	for (const auto cell_id: cells) {

		const auto c = grid.geometry.get_center(cell_id);
		const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
		const auto
			lat = asin(c[2] / r),
			lon = atan2(c[1], c[0]);

		auto* const cell_data = grid[cell_id];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << "(" << __LINE__ << ") No data for cell: "
				<< cell_id
				<< std::endl;
			abort();
		}

		Bdy_N(*cell_data)
			= initial_conditions.get_default_data(
				pamhd::particle::Bdy_Number_Density(),
				simulation_time,
				c[0], c[1], c[2],
				r, lat, lon
			);
		Bdy_V(*cell_data)
			= initial_conditions.get_default_data(
				pamhd::particle::Bdy_Velocity(),
				simulation_time,
				c[0], c[1], c[2],
				r, lat, lon
			);
		Bdy_T(*cell_data)
			= initial_conditions.get_default_data(
				pamhd::particle::Bdy_Temperature(),
				simulation_time,
				c[0], c[1], c[2],
				r, lat, lon
			);
		Bdy_Nr_Par(*cell_data)
			= initial_conditions.get_default_data(
				pamhd::particle::Bdy_Nr_Particles_In_Cell(),
				simulation_time,
				c[0], c[1], c[2],
				r, lat, lon
			);
		Bdy_C2M(*cell_data)
			= initial_conditions.get_default_data(
				pamhd::particle::Charge_Mass_Ratio(),
				simulation_time,
				c[0], c[1], c[2],
				r, lat, lon
			);
		Bdy_SpM(*cell_data)
			= initial_conditions.get_default_data(
				pamhd::particle::Species_Mass(),
				simulation_time,
				c[0], c[1], c[2],
				r, lat, lon
			);

		Ele(*cell_data)
			= initial_conditions.get_default_data(
				pamhd::particle::Electric_Field(),
				simulation_time,
				c[0], c[1], c[2],
				r, lat, lon
			);
		Mag(*cell_data)
			= initial_conditions.get_default_data(
				pamhd::particle::Magnetic_Field(),
				simulation_time,
				c[0], c[1], c[2],
				r, lat, lon
			);

		const auto cell_end = grid.geometry.get_max(cell_id);
		Bg_B_Pos_X(*cell_data) = bg_B.get_background_field(
			{cell_end[0], c[1], c[2]},
			vacuum_permeability
		);
		Bg_B_Pos_Y(*cell_data) = bg_B.get_background_field(
			{c[0], cell_end[1], c[2]},
			vacuum_permeability
		);
		Bg_B_Pos_Z(*cell_data) = bg_B.get_background_field(
			{c[0], c[1], cell_end[2]},
			vacuum_permeability
		);
	}

	/*
	Set non-default initial conditions
	*/

	// electric field
	for (
		size_t i = 0;
		i < initial_conditions.get_number_of_regions(pamhd::particle::Electric_Field());
		i++
	) {
		const auto& init_cond = initial_conditions.get_initial_condition(pamhd::particle::Electric_Field(), i);
		const auto& geometry_id = init_cond.get_geometry_id();
		const auto& cells = geometries.get_cells(geometry_id);
		for (const auto& cell: cells) {
			const auto c = grid.geometry.get_center(cell);
			const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
			const auto
				lat = asin(c[2] / r),
				lon = atan2(c[1], c[0]);

			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
				abort();
			}

			Ele(*cell_data) = initial_conditions.get_data(
				pamhd::particle::Electric_Field(),
				geometry_id,
				simulation_time,
				c[0], c[1], c[2],
				r, lat, lon
			);
		}
	}

	// magnetic field
	for (
		size_t i = 0;
		i < initial_conditions.get_number_of_regions(pamhd::particle::Magnetic_Field());
		i++
	) {
		const auto& init_cond = initial_conditions.get_initial_condition(pamhd::particle::Magnetic_Field(), i);
		const auto& geometry_id = init_cond.get_geometry_id();
		const auto& cells = geometries.get_cells(geometry_id);
		for (const auto& cell: cells) {
			const auto c = grid.geometry.get_center(cell);
			const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
			const auto
				lat = asin(c[2] / r),
				lon = atan2(c[1], c[0]);

			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << ")" << std::endl;
				abort();
			}

			Mag(*cell_data) = initial_conditions.get_data(
				pamhd::particle::Magnetic_Field(),
				geometry_id,
				simulation_time,
				c[0], c[1], c[2],
				r, lat, lon
			);
		}
	}

	// number density, set boundary data variable and create particles later
	constexpr pamhd::particle::Bdy_Number_Density N{};
	for (
		size_t i = 0;
		i < initial_conditions.get_number_of_regions(N);
		i++
	) {
		const auto& init_cond = initial_conditions.get_initial_condition(N, i);
		const auto& geometry_id = init_cond.get_geometry_id();
		const auto& cells = geometries.get_cells(geometry_id);
		for (const auto& cell: cells) {
			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
				abort();
			}

			if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
				continue;
			}

			const auto c = grid.geometry.get_center(cell);
			const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
			const auto
				lat = asin(c[2] / r),
				lon = atan2(c[1], c[0]);

			Bdy_N(*cell_data)
				= initial_conditions.get_data(
						N,
						geometry_id,
						simulation_time,
						c[0], c[1], c[2],
						r, lat, lon
				);
		}
	}

	// velocity
	constexpr pamhd::particle::Bdy_Velocity V{};
	for (
		size_t i = 0;
		i < initial_conditions.get_number_of_regions(V);
		i++
	) {
		const auto& init_cond = initial_conditions.get_initial_condition(V, i);
		const auto& geometry_id = init_cond.get_geometry_id();
		const auto& cells = geometries.get_cells(geometry_id);
		for (const auto& cell: cells) {
			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
				abort();
			}

			if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
				continue;
			}

			const auto c = grid.geometry.get_center(cell);
			const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
			const auto
				lat = asin(c[2] / r),
				lon = atan2(c[1], c[0]);

			Bdy_V(*cell_data)
				= initial_conditions.get_data(
					V,
					geometry_id,
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
		}
	}

	// temperature
	constexpr pamhd::particle::Bdy_Temperature T{};
	for (
		size_t i = 0;
		i < initial_conditions.get_number_of_regions(T);
		i++
	) {
		const auto& init_cond = initial_conditions.get_initial_condition(T, i);
		const auto& geometry_id = init_cond.get_geometry_id();
		const auto& cells = geometries.get_cells(geometry_id);
		for (const auto& cell: cells) {
			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
				abort();
			}

			if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
				continue;
			}

			const auto c = grid.geometry.get_center(cell);
			const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
			const auto
				lat = asin(c[2] / r),
				lon = atan2(c[1], c[0]);

			Bdy_T(*cell_data)
				= initial_conditions.get_data(
					T,
					geometry_id,
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
		}
	}

	// number of particles
	constexpr pamhd::particle::Bdy_Nr_Particles_In_Cell Nr{};
	for (
		size_t i = 0;
		i < initial_conditions.get_number_of_regions(Nr);
		i++
	) {
		const auto& init_cond = initial_conditions.get_initial_condition(Nr, i);
		const auto& geometry_id = init_cond.get_geometry_id();
		const auto& cells = geometries.get_cells(geometry_id);
		for (const auto& cell: cells) {
			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
				abort();
			}

			if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
				continue;
			}

			const auto c = grid.geometry.get_center(cell);
			const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
			const auto
				lat = asin(c[2] / r),
				lon = atan2(c[1], c[0]);

			Bdy_Nr_Par(*cell_data)
				= initial_conditions.get_data(
					Nr,
					geometry_id,
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
		}
	}

	// charge to mass ratio
	constexpr pamhd::particle::Charge_Mass_Ratio C2M{};
	for (
		size_t i = 0;
		i < initial_conditions.get_number_of_regions(C2M);
		i++
	) {
		const auto& init_cond = initial_conditions.get_initial_condition(C2M, i);
		const auto& geometry_id = init_cond.get_geometry_id();
		const auto& cells = geometries.get_cells(geometry_id);
		for (const auto& cell: cells) {
			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
				abort();
			}

			if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
				continue;
			}

			const auto c = grid.geometry.get_center(cell);
			const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
			const auto
				lat = asin(c[2] / r),
				lon = atan2(c[1], c[0]);

			Bdy_C2M(*cell_data)
				= initial_conditions.get_data(
					C2M,
					geometry_id,
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
		}
	}

	// species mass
	constexpr pamhd::particle::Species_Mass SpM{};
	for (
		size_t i = 0;
		i < initial_conditions.get_number_of_regions(SpM);
		i++
	) {
		const auto& init_cond = initial_conditions.get_initial_condition(SpM, i);
		const auto& geometry_id = init_cond.get_geometry_id();
		const auto& cells = geometries.get_cells(geometry_id);
		for (const auto& cell: cells) {
			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
				abort();
			}

			if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
				continue;
			}

			const auto c = grid.geometry.get_center(cell);
			const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
			const auto
				lat = asin(c[2] / r),
				lon = atan2(c[1], c[0]);

			Bdy_SpM(*cell_data)
				= initial_conditions.get_data(
					SpM,
					geometry_id,
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
		}
	}

	for (const auto& cell: cells) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
			abort();
		}

		if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
			continue;
		}

		random_source.seed(cell);

		const auto
			cell_start = grid.geometry.get_min(cell),
			cell_end = grid.geometry.get_max(cell),
			cell_length = grid.geometry.get_length(cell);

		auto new_particles
			= create_particles<
				Particle,
				Particle_Mass_T,
				Particle_Charge_Mass_Ratio_T,
				Particle_Position_T,
				Particle_Velocity_T,
				Particle_ID_T,
				Particle_Species_Mass_T
			>(
				Bdy_V(*cell_data),
				Eigen::Vector3d{cell_start[0], cell_start[1], cell_start[2]},
				Eigen::Vector3d{cell_end[0], cell_end[1], cell_end[2]},
				Eigen::Vector3d{Bdy_T(*cell_data), Bdy_T(*cell_data), Bdy_T(*cell_data)},
				Bdy_Nr_Par(*cell_data),
				Bdy_C2M(*cell_data),
				Bdy_SpM(*cell_data) * Bdy_N(*cell_data) * cell_length[0] * cell_length[1] * cell_length[2],
				Bdy_SpM(*cell_data),
				particle_temp_nrj_ratio,
				random_source,
				current_id_start,
				particle_id_increase
			);
		nr_particles_created += new_particles.size();

		if (replace) {
			Par(*cell_data) = std::move(new_particles);
		} else {
			Par(*cell_data).insert(
				Par(*cell_data).end(),
				new_particles.begin(),
				new_particles.end()
			);
		}

		current_id_start += new_particles.size() * particle_id_increase;
	}

	if (verbose && grid.get_rank() == 0) {
		std::cout << "done" << std::endl;
	}

	return nr_particles_created;
}

}} // namespaces

#endif // ifndef PAMHD_PARTICLE_INITIALIZE_HPP
