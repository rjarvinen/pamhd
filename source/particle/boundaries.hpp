/*
Handles boundary logic of particle part of PAMHD.

Copyright 2015, 2016, 2017 Ilja Honkonen
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

#ifndef PAMHD_PARTICLE_BOUNDARIES_HPP
#define PAMHD_PARTICLE_BOUNDARIES_HPP


#include "algorithm"
#include "cmath"
#include "limits"
#include "set"
#include "string"
#include "utility"
#include "vector"

#include "dccrg.hpp"

#include "mhd/common.hpp"
#include "particle/common.hpp"
#include "particle/variables.hpp"


namespace pamhd {
namespace particle {


/*!
Prepares boundary information about each simulation cell needed for particle solution.

Only uses boundary info of first particle population.
*/
template<
	class Solver_Info,
	class Cell_Data,
	class Geometry,
	class Boundaries,
	class Boundary_Geometries,
	class Solver_Info_Getter
> void set_solver_info(
	dccrg::Dccrg<Cell_Data, Geometry>& grid,
	std::vector<Boundaries>& boundaries,
	const Boundary_Geometries& geometries,
	const Solver_Info_Getter Sol_Info
) {
	if (boundaries.size() == 0) {
		for (const auto& cell: grid.cells) {
			Sol_Info(*cell.data) = 0;
		}

		Cell_Data::set_transfer_all(true, pamhd::particle::Solver_Info());
		grid.update_copies_of_remote_neighbors();
		Cell_Data::set_transfer_all(false, pamhd::particle::Solver_Info());
		return;
	}

	Cell_Data::set_transfer_all(true, pamhd::particle::Solver_Info());
	boundaries[0].classify(grid, geometries, Sol_Info);

	for (const auto& cell: grid.cells) {
		Sol_Info(*cell.data) = 0;
	}

	// number of particles
	constexpr pamhd::particle::Bdy_Nr_Particles_In_Cell NPIC{};
	for (const auto& cell: boundaries[0].get_value_boundary_cells(NPIC)) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::particle_number_bdy;
	}
	for (const auto& item: boundaries[0].get_copy_boundary_cells(NPIC)) {
		auto* const cell_data = grid[item[0]];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::particle_number_bdy;
	}
	const std::set<uint64_t> dont_solve_nr_pic(
		boundaries[0].get_dont_solve_cells(NPIC).cbegin(),
		boundaries[0].get_dont_solve_cells(NPIC).cend()
	);

	// mass of particle species
	constexpr pamhd::particle::Bdy_Species_Mass SpM{};
	for (const auto& cell: boundaries[0].get_value_boundary_cells(SpM)) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::particle_species_mass_bdy;
	}
	for (const auto& item: boundaries[0].get_copy_boundary_cells(SpM)) {
		auto* const cell_data = grid[item[0]];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::particle_species_mass_bdy;
	}
	const std::set<uint64_t> dont_solve_spm(
		boundaries[0].get_dont_solve_cells(SpM).cbegin(),
		boundaries[0].get_dont_solve_cells(SpM).cend()
	);

	// charge to mass ratio
	constexpr pamhd::particle::Bdy_Charge_Mass_Ratio C2M{};
	for (const auto& cell: boundaries[0].get_value_boundary_cells(C2M)) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::particle_charge_mass_ratio_bdy;
	}
	for (const auto& item: boundaries[0].get_copy_boundary_cells(C2M)) {
		auto* const cell_data = grid[item[0]];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::particle_charge_mass_ratio_bdy;
	}
	const std::set<uint64_t> dont_solve_c2m(
		boundaries[0].get_dont_solve_cells(C2M).cbegin(),
		boundaries[0].get_dont_solve_cells(C2M).cend()
	);

	// number density
	constexpr pamhd::particle::Bdy_Number_Density N{};
	for (const auto& cell: boundaries[0].get_value_boundary_cells(N)) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::number_density_bdy;
	}
	for (const auto& item: boundaries[0].get_copy_boundary_cells(N)) {
		auto* const cell_data = grid[item[0]];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::number_density_bdy;
	}
	const std::set<uint64_t> dont_solve_mass(
		boundaries[0].get_dont_solve_cells(N).cbegin(),
		boundaries[0].get_dont_solve_cells(N).cend()
	);

	// velocity
	constexpr pamhd::particle::Bdy_Velocity V{};
	for (const auto& cell: boundaries[0].get_value_boundary_cells(V)) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::velocity_bdy;
	}
	for (const auto& item: boundaries[0].get_copy_boundary_cells(V)) {
		auto* const cell_data = grid[item[0]];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::velocity_bdy;
	}
	const std::set<uint64_t> dont_solve_velocity(
		boundaries[0].get_dont_solve_cells(V).cbegin(),
		boundaries[0].get_dont_solve_cells(V).cend()
	);

	// temperature
	constexpr pamhd::particle::Bdy_Temperature T{};
	for (const auto& cell: boundaries[0].get_value_boundary_cells(T)) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::temperature_bdy;
	}
	for (const auto& item: boundaries[0].get_copy_boundary_cells(T)) {
		auto* const cell_data = grid[item[0]];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::temperature_bdy;
	}
	const std::set<uint64_t> dont_solve_temperature(
		boundaries[0].get_dont_solve_cells(T).cbegin(),
		boundaries[0].get_dont_solve_cells(T).cend()
	);

	if (
		not std::equal(
			dont_solve_mass.cbegin(),
			dont_solve_mass.cend(),
			dont_solve_spm.cbegin()
		)
	) {
		throw std::invalid_argument(
			std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
			+ "Number density and species mass dont_solves aren't equal."
		);
	}
	if (
		not std::equal(
			dont_solve_mass.cbegin(),
			dont_solve_mass.cend(),
			dont_solve_c2m.cbegin()
		)
	) {
		throw std::invalid_argument(
			std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
			+ "Number density and charge-mass ratio dont_solves aren't equal."
		);
	}
	if (
		not std::equal(
			dont_solve_mass.cbegin(),
			dont_solve_mass.cend(),
			dont_solve_nr_pic.cbegin()
		)
	) {
		throw std::invalid_argument(
			std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
			+ "Number density and number of particles dont_solves aren't equal."
		);
	}
	if (
		not std::equal(
			dont_solve_mass.cbegin(),
			dont_solve_mass.cend(),
			dont_solve_velocity.cbegin()
		)
	) {
		throw std::invalid_argument(
			std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
			+ "Mass density and velocity dont_solves aren't equal."
		);
	}
	if (
		not std::equal(
			dont_solve_mass.cbegin(),
			dont_solve_mass.cend(),
			dont_solve_temperature.cbegin()
		)
	) {
		throw std::invalid_argument(
			std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
			+ "Mass density and temperature dont_solves aren't equal."
		);
	}

	for (size_t i = 0; i < boundaries.size(); i++) {
		const auto& npic_cpy_cells = boundaries[i].get_copy_boundary_cells(NPIC);
		const auto& spm_cpy_cells = boundaries[i].get_copy_boundary_cells(SpM);
		if (
			not std::equal(
				npic_cpy_cells.cbegin(),
				npic_cpy_cells.cend(),
				spm_cpy_cells.cbegin()
			)
		) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "Number of particles in cell and species mass copy boundaries not identical for population "
				+ std::to_string(i)
			);
		}

		const auto& c2m_cpy_cells = boundaries[i].get_copy_boundary_cells(C2M);
		if (
			not std::equal(
				npic_cpy_cells.cbegin(),
				npic_cpy_cells.cend(),
				c2m_cpy_cells.cbegin()
			)
		) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "Number of particles in cell and charge to mass copy boundaries not identical for population "
				+ std::to_string(i)
			);
		}

		const auto& n_cpy_cells = boundaries[i].get_copy_boundary_cells(N);
		if (
			not std::equal(
				npic_cpy_cells.cbegin(),
				npic_cpy_cells.cend(),
				n_cpy_cells.cbegin()
			)
		) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "Number of particles in cell and number copy boundaries not identical for population "
				+ std::to_string(i)
			);
		}

		const auto& v_cpy_cells = boundaries[i].get_copy_boundary_cells(V);
		if (
			not std::equal(
				npic_cpy_cells.cbegin(),
				npic_cpy_cells.cend(),
				v_cpy_cells.cbegin()
			)
		) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "Number of particles in cell and velocity copy boundaries not identical for population "
				+ std::to_string(i)
			);
		}

		const auto& t_cpy_cells = boundaries[i].get_copy_boundary_cells(T);
		if (
			not std::equal(
				npic_cpy_cells.cbegin(),
				npic_cpy_cells.cend(),
				t_cpy_cells.cbegin()
			)
		) {
			throw std::invalid_argument(
				std::string(__FILE__ "(") + std::to_string(__LINE__) + "): "
				+ "Number of particles in cell and temperature copy boundaries not identical for population "
				+ std::to_string(i)
			);
		}
	}

	// don't solve cells in which no variable is solved
	for (auto& cell: dont_solve_mass) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::dont_solve;
	}

	grid.update_copies_of_remote_neighbors();
	Cell_Data::set_transfer_all(false, pamhd::particle::Solver_Info());
}


//! As set_solver_info but for electric field only
template<
	class Solver_Info,
	class Cell_Data,
	class Geometry,
	class Boundaries,
	class Boundary_Geometries,
	class Solver_Info_Getter
> void set_solver_info_electric(
	dccrg::Dccrg<Cell_Data, Geometry>& grid,
	std::vector<Boundaries>& boundaries,
	const Boundary_Geometries& geometries,
	const Solver_Info_Getter Sol_Info
) {
	Cell_Data::set_transfer_all(true, pamhd::particle::Solver_Info());
	boundaries.classify(grid, geometries, Sol_Info);

	for (const auto& cell: grid.cells) {
		Sol_Info(*cell.data) = 0;
	}

	constexpr pamhd::particle::Electric_Field E{};
	for (const auto& cell: boundaries[0].get_value_boundary_cells(E)) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::electric_field_bdy;
	}
	for (const auto& item: boundaries[0].get_copy_boundary_cells(E)) {
		auto* const cell_data = grid[item[0]];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::electric_field_bdy;
	}

	for (auto& cell: boundaries[0].get_dont_solve_cells(E)) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
			abort();
		}
		Sol_Info(*cell_data) |= Solver_Info::dont_solve;
	}

	grid.update_copies_of_remote_neighbors();
	Cell_Data::set_transfer_all(false, pamhd::particle::Solver_Info());
}


/*
Copy boundaries have no effect in test particle simulation.
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
	class Boundaries,
	class Grid,
	class Solver_Info_Getter,
	class Electric_Field_Getter,
	class Magnetic_Field_Getter,
	class Particles_Getter,
	class Boundary_Number_Density_Getter,
	class Boundary_Velocity_Getter,
	class Boundary_Temperature_Getter,
	class Boundary_Nr_Particles_Getter,
	class Boundary_Charge_To_Mass_Ratio_Getter,
	class Boundary_Species_Mass_Getter
> size_t apply_massless_boundaries(
	const Sim_Geometries& bdy_geoms,
	std::vector<Boundaries>& boundaries,
	const double simulation_time,
	const size_t simulation_step,
	const std::vector<uint64_t>& cells,
	Grid& grid,
	std::mt19937_64& random_source,
	const double particle_temp_nrj_ratio,
	const double vacuum_permeability,
	const unsigned long long int first_particle_id,
	const unsigned long long int particle_id_increase,
	const bool verbose,
	const Solver_Info_Getter Sol_Info,
	const Electric_Field_Getter Ele,
	const Magnetic_Field_Getter Mag,
	const Particles_Getter Par,
	const Boundary_Number_Density_Getter Bdy_N,
	const Boundary_Velocity_Getter Bdy_V,
	const Boundary_Temperature_Getter Bdy_T,
	const Boundary_Nr_Particles_Getter Bdy_NPIC,
	const Boundary_Charge_To_Mass_Ratio_Getter Bdy_C2M,
	const Boundary_Species_Mass_Getter Bdy_SpM
) {
	size_t
		current_id_start = first_particle_id,
		nr_particles_created = 0;

	for (size_t bdy_i = 0; bdy_i < boundaries.size(); bdy_i++) {
		// magnetic field
		constexpr pamhd::Magnetic_Field B{};
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(B);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(B, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
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

				Mag(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}

		// electric field
		constexpr pamhd::particle::Electric_Field E{};
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(E);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(E, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
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

				Ele(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}

		// number density, set boundary data variable and replace particles later
		// TODO: adjust particle parameters in-place to keep particle ids
		constexpr pamhd::particle::Bdy_Number_Density N{};
		std::set<uint64_t> bdy_cells;
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(N);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(N, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_N(*cell_data) = value_bdy.get_data(
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
			i < boundaries[bdy_i].get_number_of_value_boundaries(V);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(V, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_V(*cell_data) = value_bdy.get_data(
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
			i < boundaries[bdy_i].get_number_of_value_boundaries(T);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(T, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_T(*cell_data) = value_bdy.get_data(
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
			i < boundaries[bdy_i].get_number_of_value_boundaries(Nr);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(Nr, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_NPIC(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}

		// particle species mass
		constexpr pamhd::particle::Bdy_Species_Mass SpM{};
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(SpM);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(SpM, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_SpM(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}

		// particle charge mass ratio
		constexpr pamhd::particle::Bdy_Species_Mass C2M{};
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(C2M);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(C2M, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_C2M(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}

		for (const auto& cell: bdy_cells) {
			random_source.seed(cell + 100000 * simulation_step + 10000000000 * bdy_i);

			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << ") No data for cell: "
					<< cell
					<< std::endl;
				abort();
			}

			const auto
				cell_start = grid.geometry.get_min(cell),
				cell_end = grid.geometry.get_max(cell),
				cell_length = grid.geometry.get_length(cell);

			const auto new_particles
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
					Bdy_NPIC(*cell_data),
					Bdy_C2M(*cell_data),
					Bdy_SpM(*cell_data) * Bdy_N(*cell_data) * cell_length[0] * cell_length[1] * cell_length[2],
					Bdy_SpM(*cell_data),
					particle_temp_nrj_ratio,
					random_source,
					current_id_start,
					particle_id_increase
				);
			nr_particles_created += new_particles.size();

			if (bdy_i == 0) {
				Par(*cell_data).clear();
			}
			Par(*cell_data).insert(Par(*cell_data).cend(), new_particles.cbegin(), new_particles.cend());

			current_id_start += new_particles.size() * particle_id_increase;
		}
	}

	return nr_particles_created;
}


/*
Applies particle boundary logic.

Copy boundaries copy N random particles from neighboring normal cells
where N is average number of particles in neighboring cells.
Copied particles' id is changed and position randomized.
Particle bulk data is averaged from neighboring normal cells.

Boundaries of every population overwrite existing particles.
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
	class Boundaries,
	class Grid,
	class Solver_Info_Getter,
	class Particles_Getter,
	class Boundary_Number_Density_Getter,
	class Boundary_Velocity_Getter,
	class Boundary_Temperature_Getter,
	class Boundary_Nr_Particles_Getter,
	class Boundary_Charge_To_Mass_Ratio_Getter,
	class Boundary_Species_Mass_Getter
> size_t apply_boundaries(
	const Sim_Geometries& bdy_geoms,
	std::vector<Boundaries>& boundaries,
	const double simulation_time,
	const size_t simulation_step,
	const std::vector<uint64_t>& cells,
	Grid& grid,
	std::mt19937_64& random_source,
	const double particle_temp_nrj_ratio,
	const double vacuum_permeability,
	const unsigned long long int first_particle_id,
	const unsigned long long int particle_id_increase,
	const bool verbose,
	const Solver_Info_Getter Sol_Info,
	const Particles_Getter Par,
	const Boundary_Number_Density_Getter Bdy_N,
	const Boundary_Velocity_Getter Bdy_V,
	const Boundary_Temperature_Getter Bdy_T,
	const Boundary_Nr_Particles_Getter Bdy_NPIC,
	const Boundary_Charge_To_Mass_Ratio_Getter Bdy_C2M,
	const Boundary_Species_Mass_Getter Bdy_SpM
) {
	size_t
		current_id_start = first_particle_id,
		nr_particles_created = 0;

	for (size_t bdy_i = 0; bdy_i < boundaries.size(); bdy_i++) {

		// number density, set boundary data variable and replace particles later
		// TODO?: adjust particle parameters in-place to keep particle ids
		constexpr pamhd::particle::Bdy_Number_Density N{};
		std::set<uint64_t> bdy_cells;
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(N);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(N, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_N(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}
		for (const auto& item: boundaries[bdy_i].get_copy_boundary_cells(N)) {
			if (item.size() < 2) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			// copy particles
			size_t total_number_of_particles = 0;
			for (size_t i = 1; i < item.size(); i++) {
				auto* const source_data = grid[item[i]];
				if (source_data == nullptr) {
					std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
					abort();
				}
				total_number_of_particles += Par(*source_data).size();
			}

			auto* const target_data = grid[item[0]];
			if (target_data == nullptr) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			Par(*target_data).clear(); // only do when bdy_i == 0?

			// copy particles to random position in target cell
			const auto
				cell_min = grid.geometry.get_min(item[0]),
				cell_max = grid.geometry.get_max(item[0]);
			std::uniform_real_distribution<double>
				pos_x_gen(cell_min[0], cell_max[0]),
				pos_y_gen(cell_min[1], cell_max[1]),
				pos_z_gen(cell_min[2], cell_max[2]);

			for (size_t i = 1; i < item.size(); i++) {
				auto* const source_data = grid[item[i]];
				const size_t nr_particles = Par(*source_data).size();
				if (nr_particles == 0) {
					continue;
				}

				std::uniform_int_distribution<size_t> index_generator(0, nr_particles - 1);
				const size_t nr_particles_to_copy = nr_particles * (double(nr_particles) / total_number_of_particles);

				std::set<size_t> particles_to_copy;
				while (particles_to_copy.size() < nr_particles_to_copy) {
					particles_to_copy.emplace(index_generator(random_source));
				}
				for (const auto& index: particles_to_copy) {
					auto particle = Par(*source_data)[index];

					particle[Particle_Position_T()][0] = pos_x_gen(random_source);
					particle[Particle_Position_T()][1] = pos_y_gen(random_source);
					particle[Particle_Position_T()][2] = pos_z_gen(random_source);
					particle[Particle_ID_T()] = current_id_start;
					current_id_start += particle_id_increase;

					Par(*target_data).emplace_back(particle);
				}
			}

			// copy bulk data, assume identical copy boundary cells for all variables
			pamhd::particle::Bdy_Number_Density::data_type source_value{0};
			for (size_t i = 1; i < item.size(); i++) {
				auto* const source_data = grid[item[i]];
				if (source_data == nullptr) {
					std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
					abort();
				}

				Bdy_N(*source_data) = get_bulk_nr_particles<Particle_Mass_T, Particle_Species_Mass_T>(Par(*source_data));
				source_value += Bdy_N(*source_data);
			}
			source_value /= item.size() - 1;

			Bdy_N(*target_data) = source_value;
		}

		// velocity
		constexpr pamhd::particle::Bdy_Velocity V{};
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(V);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(V, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_V(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}
		for (const auto& item: boundaries[bdy_i].get_copy_boundary_cells(V)) {
			if (item.size() < 2) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			// particles copied in number density

			pamhd::particle::Bdy_Velocity::data_type source_value{0, 0, 0};
			for (size_t i = 1; i < item.size(); i++) {
				auto* const source_data = grid[item[i]];
				if (source_data == nullptr) {
					std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
					abort();
				}

				Bdy_V(*source_data) = get_bulk_velocity<Particle_Mass_T, Particle_Velocity_T, Particle_Species_Mass_T>(Par(*source_data));
				source_value += Bdy_V(*source_data);
			}
			source_value /= item.size() - 1;

			auto* const target_data = grid[item[0]];
			if (target_data == nullptr) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			Bdy_V(*target_data) = source_value;
		}

		// temperature
		constexpr pamhd::particle::Bdy_Temperature T{};
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(T);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(T, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_T(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}
		for (const auto& item: boundaries[bdy_i].get_copy_boundary_cells(T)) {
			if (item.size() < 2) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			pamhd::particle::Bdy_Temperature::data_type source_value{0};
			for (size_t i = 1; i < item.size(); i++) {
				auto* const source_data = grid[item[i]];
				if (source_data == nullptr) {
					std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
					abort();
				}

				Bdy_T(*source_data) = get_temperature<Particle_Mass_T, Particle_Velocity_T, Particle_Species_Mass_T>(Par(*source_data), particle_temp_nrj_ratio);
				source_value += Bdy_T(*source_data);
			}
			source_value /= item.size() - 1;

			auto* const target_data = grid[item[0]];
			if (target_data == nullptr) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			Bdy_T(*target_data) = source_value;
		}

		// number of particles
		constexpr pamhd::particle::Bdy_Nr_Particles_In_Cell Nr{};
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(Nr);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(Nr, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_NPIC(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}
		for (const auto& item: boundaries[bdy_i].get_copy_boundary_cells(Nr)) {
			if (item.size() < 2) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			pamhd::particle::Bdy_Nr_Particles_In_Cell::data_type source_value{0};
			for (size_t i = 1; i < item.size(); i++) {
				auto* const source_data = grid[item[i]];
				if (source_data == nullptr) {
					std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
					abort();
				}

				Bdy_NPIC(*source_data) = Par(*source_data).size();
				source_value += Bdy_NPIC(*source_data);
			}
			source_value /= item.size() - 1;

			auto* const target_data = grid[item[0]];
			if (target_data == nullptr) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			Bdy_NPIC(*target_data) = source_value;
		}

		// particle species mass
		constexpr pamhd::particle::Bdy_Species_Mass SpM{};
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(SpM);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(SpM, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_SpM(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}
		for (const auto& item: boundaries[bdy_i].get_copy_boundary_cells(SpM)) {
			if (item.size() < 2) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			pamhd::particle::Bdy_Species_Mass::data_type source_value{0};
			for (size_t i = 1; i < item.size(); i++) {
				auto* const source_data = grid[item[i]];
				if (source_data == nullptr) {
					std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
					abort();
				}

				Bdy_SpM(*source_data) = 0;
				for (const auto& particle: Par(*source_data)) {
					Bdy_SpM(*source_data) += particle[Particle_Species_Mass_T()];
				}
				Bdy_SpM(*source_data) /= Par(*source_data).size();
				source_value += Bdy_SpM(*source_data);
			}
			source_value /= item.size() - 1;

			auto* const target_data = grid[item[0]];
			if (target_data == nullptr) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			Bdy_SpM(*target_data) = source_value;
		}

		// particle charge mass ratio
		constexpr pamhd::particle::Bdy_Charge_Mass_Ratio C2M{};
		for (
			size_t i = 0;
			i < boundaries[bdy_i].get_number_of_value_boundaries(C2M);
			i++
		) {
			auto& value_bdy = boundaries[bdy_i].get_value_boundary(C2M, i);
			const auto& geometry_id = value_bdy.get_geometry_id();
			const auto& cells = bdy_geoms.get_cells(geometry_id);
			for (const auto& cell: cells) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << std::endl;
					abort();
				}

				if ((Sol_Info(*cell_data) & Solver_Info::dont_solve) > 0) {
					continue;
				}

				bdy_cells.insert(cell);

				const auto c = grid.geometry.get_center(cell);
				const auto r = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
				const auto
					lat = asin(c[2] / r),
					lon = atan2(c[1], c[0]);

				Bdy_C2M(*cell_data) = value_bdy.get_data(
					simulation_time,
					c[0], c[1], c[2],
					r, lat, lon
				);
			}
		}
		for (const auto& item: boundaries[bdy_i].get_copy_boundary_cells(C2M)) {
			if (item.size() < 2) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			pamhd::particle::Charge_Mass_Ratio::data_type source_value{0};
			for (size_t i = 1; i < item.size(); i++) {
				auto* const source_data = grid[item[i]];
				if (source_data == nullptr) {
					std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
					abort();
				}

				Bdy_C2M(*source_data) = 0;
				for (const auto& particle: Par(*source_data)) {
					Bdy_C2M(*source_data) += particle[Particle_Charge_Mass_Ratio_T()];
				}
				Bdy_C2M(*source_data) /= Par(*source_data).size();
				source_value += Bdy_C2M(*source_data);
			}
			source_value /= item.size() - 1;

			auto* const target_data = grid[item[0]];
			if (target_data == nullptr) {
				std::cerr <<  __FILE__ << ":" << __LINE__ << std::endl;
				abort();
			}

			Bdy_C2M(*target_data) = source_value;
		}

		for (const auto& cell: bdy_cells) {
			random_source.seed(cell + 100000 * simulation_step + 10000000000 * bdy_i);

			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << ") No data for cell: "
					<< cell
					<< std::endl;
				abort();
			}

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
					Bdy_NPIC(*cell_data),
					Bdy_C2M(*cell_data),
					Bdy_SpM(*cell_data) * Bdy_N(*cell_data) * cell_length[0] * cell_length[1] * cell_length[2],
					Bdy_SpM(*cell_data),
					particle_temp_nrj_ratio,
					random_source,
					current_id_start,
					particle_id_increase
				);
			nr_particles_created += new_particles.size();
			current_id_start += new_particles.size() * particle_id_increase;

			Par(*cell_data) = std::move(new_particles);
		}
	}

	return nr_particles_created;
}


}} // namespaces


#endif // ifndef PAMHD_PARTICLE_BOUNDARIES_HPP
