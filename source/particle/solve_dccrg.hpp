/*
Particle propagator of PAMHD for DCCRG grid.

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

#ifndef PAMHD_PARTICLE_SOLVE_DCCRG_HPP
#define PAMHD_PARTICLE_SOLVE_DCCRG_HPP


#include "type_traits"
#include "utility"

#include "dccrg.hpp"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "prettyprint.hpp"

#include "common.hpp"
#include "interpolate.hpp"


namespace pamhd {
namespace particle {


/*!
State type used in boost::numeric::odeint particle propagation.
state[0] = particle position, state[1] = velocity
*/
using state_t = std::array<Eigen::Vector3d, 2>;


/*!
Object given to boost::numeric::odeint to propagate a particle.

Interpolates electric and magnetic fields from given values to
particle position at each call to operator().
*/
class Particle_Propagator
{
private:

	const double& charge_to_mass_ratio;

	const Eigen::Vector3d
		&data_start, &data_end;

	const std::array<Eigen::Vector3d, 27>
		&current_minus_velocity, &magnetic_field;

	bool E_is_derived;

	const Eigen::Vector3d& bg_B;

public:

	/*!
	Arguments except charge to mass ratio are passed to interpolate().

	If given_E_is_derived == true then E is interpolated by interpolating
	J - V and B to particle position from which E = (J-V) x B, otherwise
	E is interpolated directly by assuming E = J-V.
	*/
	Particle_Propagator(
		const double& given_charge_to_mass_ratio,
		const Eigen::Vector3d& given_data_start,
		const Eigen::Vector3d& given_data_end,
		const std::array<Eigen::Vector3d, 27>& given_current_minus_velocity,
		const std::array<Eigen::Vector3d, 27>& given_magnetic_field,
		const bool given_E_is_derived,
		const Eigen::Vector3d& given_bg_B = {0, 0, 0}
	) :
		charge_to_mass_ratio(given_charge_to_mass_ratio),
		data_start(given_data_start),
		data_end(given_data_end),
		current_minus_velocity(given_current_minus_velocity),
		magnetic_field(given_magnetic_field),
		E_is_derived(given_E_is_derived),
		bg_B(given_bg_B)
	{}

	void operator()(
		const state_t& state,
		state_t& change,
		const double
	) const {
		const auto
			B_at_pos
				= interpolate(
					state[0],
					this->data_start,
					this->data_end,
					this->magnetic_field
				),
			E_at_pos = [&](){
				if (this->E_is_derived) {
					const auto J_m_V_at_pos = interpolate(
						state[0],
						this->data_start,
						this->data_end,
						this->current_minus_velocity
					);
					return J_m_V_at_pos.cross(B_at_pos);
				} else {
					return interpolate(
						state[0],
						this->data_start,
						this->data_end,
						this->current_minus_velocity
					);
				}
			}();

		change[0] = state[1];
		change[1]
			= this->charge_to_mass_ratio
			* (E_at_pos + state[1].cross(B_at_pos + this->bg_B));
		/*std::cout
			<< "v: " << state[1][0] << ", " << state[1][1] << ", " << state[1][2]
			<< ", E: " << E_at_pos[0] << ", " << E_at_pos[1] << ", " << E_at_pos[2]
			<< ", B: " << B_at_pos[0] << ", " << B_at_pos[1] << ", " << B_at_pos[2]
			<< std::endl;*/
	};
};


/*!
Propagates particles in given cells for a given amount of time.

Returns the longest allowed time step for given cells
and their neighbors. Particles which propagate outside of the
cell in which they are stored are moved to the External_Particles_T
list of their previous cell and added to Particle_Destinations_T
information.
*/
template<
	class Stepper,
	class Cell,
	class Background_Magnetic_Field,
	class Current_Minus_Velocity_Getter,
	class Magnetic_Field_Getter,
	class Nr_Particles_External_Getter,
	class Particles_Internal_Getter,
	class Particles_External_Getter,
	class Particle_Position_Getter,
	class Particle_Velocity_Getter,
	class Particle_Charge_Mass_Ratio_Getter,
	class Particle_Mass_Getter,
	class Particle_Destination_Cell_Getter
> double solve(
	const double dt,
	const std::vector<uint64_t>& cell_ids,
	dccrg::Dccrg<Cell, dccrg::Cartesian_Geometry>& grid,
	const Background_Magnetic_Field& bg_B,
	const double vacuum_permeability,
	const bool E_is_derived_quantity,
	// if E_is_derived_quantity == true: JmV = J - V, else JmV = E
	const Current_Minus_Velocity_Getter JmV,
	const Magnetic_Field_Getter Mag,
	const Nr_Particles_External_Getter Nr_Ext,
	const Particles_Internal_Getter Part_Int,
	const Particles_External_Getter Part_Ext,
	const Particle_Position_Getter Part_Pos,
	const Particle_Velocity_Getter Part_Vel,
	const Particle_Charge_Mass_Ratio_Getter Part_C2M,
	const Particle_Mass_Getter Part_Mas,
	const Particle_Destination_Cell_Getter Part_Des
) {
	namespace odeint = boost::numeric::odeint;
	using std::isnan;
	using std::is_same;

	static_assert(
		is_same<Stepper, odeint::euler<state_t>>::value
		or is_same<Stepper, odeint::modified_midpoint<state_t>>::value
		or is_same<Stepper, odeint::runge_kutta4<state_t>>::value
		or is_same<Stepper, odeint::runge_kutta_cash_karp54<state_t>>::value
		or is_same<Stepper, odeint::runge_kutta_fehlberg78<state_t>>::value,
		"Only odeint steppers without internal state are supported."
	);


	Stepper stepper;
	double max_time_step = std::numeric_limits<double>::max();
	for (const auto& cell_id: cell_ids) {

		// get field data from neighborhood for interpolation
		std::array<Eigen::Vector3d, 27> current_minus_velocities, magnetic_fields;

		const auto* const neighbor_ids = grid.get_neighbors_of(cell_id);
		if (neighbor_ids == nullptr) {
			std::cerr << __FILE__ << "(" << __LINE__ << "): " << std::endl;
			abort();
		}

		if (neighbor_ids->size() != 26) {
			std::cerr << __FILE__ << "(" << __LINE__ << "): "
				<< "Unsupported neighborhood: " << neighbor_ids->size()
				<< std::endl;
			abort();
		}

		auto* const cell_data = grid[cell_id];
		if (cell_data == NULL) {
			std::cerr << __FILE__ << "(" << __LINE__ << "): " << std::endl;
			abort();
		}

		// put current cell's data into middle
		current_minus_velocities[14] = JmV(*cell_data);
		magnetic_fields[14] = Mag(*cell_data);

		for (size_t i = 0; i < neighbor_ids->size(); i++) {
			const auto neighbor_id = (*neighbor_ids)[i];

			// use same values in missing neighbors
			if (neighbor_id == dccrg::error_cell) {
				if (i <= 13) {
					current_minus_velocities[i] = current_minus_velocities[14];
					magnetic_fields[i] = magnetic_fields[14];
				} else {
					current_minus_velocities[i + 1] = current_minus_velocities[14];
					magnetic_fields[i + 1] = magnetic_fields[14];
				}

				continue;
			}

			auto* const neighbor_data = grid[neighbor_id];
			if (neighbor_data == nullptr) {
				std::cerr << __FILE__ << "(" << __LINE__ << "): " << std::endl;
				abort();
			}

			if (i <= 13) {
				current_minus_velocities[i] = JmV(*neighbor_data);
				magnetic_fields[i] = Mag(*neighbor_data);
			} else {
				current_minus_velocities[i + 1] = JmV(*neighbor_data);
				magnetic_fields[i + 1] = Mag(*neighbor_data);
			}
		}

		const auto
			cell_min = grid.geometry.get_min(cell_id),
			cell_max = grid.geometry.get_max(cell_id),
			cell_center = grid.geometry.get_center(cell_id),
			cell_length = grid.geometry.get_length(cell_id);

		const Eigen::Vector3d
			interpolation_start{
				cell_center[0] - cell_length[0],
				cell_center[1] - cell_length[1],
				cell_center[2] - cell_length[2]
			},
			interpolation_end{
				cell_center[0] + cell_length[0],
				cell_center[1] + cell_length[1],
				cell_center[2] + cell_length[2]
			};

		// calculate max length of time step for next step from cell-centered values
		const auto E_centered = [&](){
			if (E_is_derived_quantity) {
				return JmV(*cell_data).cross(Mag(*cell_data));
			} else {
				return JmV(*cell_data);
			}
		}();
		const decltype(E_centered) B_centered = Mag(*cell_data) + bg_B.get_background_field(
			{cell_center[0], cell_center[1], cell_center[2]},
			vacuum_permeability
		);

		for (size_t i = 0; i < Part_Int(*cell_data).size(); i++) {
			auto particle = Part_Int(*cell_data)[i]; // reference faster?

			// TODO check accurately only for most restrictive particle(s) in each cell?
			max_time_step = std::min(
				max_time_step,
				get_minmax_step(
					1.0,
					1.0 / 32.0,
					// only allow particles to propagate through half a
					// cell in order to not break field interpolation
					cell_length[0] / 2.0,
					Part_C2M(particle),
					Part_Vel(particle),
					E_centered,
					B_centered
				).second
			);

			const Particle_Propagator propagator(
				Part_C2M(particle),
				interpolation_start,
				interpolation_end,
				current_minus_velocities,
				magnetic_fields,
				E_is_derived_quantity,
				bg_B.get_background_field(Part_Pos(particle), vacuum_permeability)
			);

			state_t state{{Part_Pos(particle), Part_Vel(particle)}};
			stepper.do_step(propagator, state, 0.0, dt);
			Part_Vel(Part_Int(*cell_data)[i]) = state[1];


			// take into account periodic grid
			const std::array<double, 3> real_pos
				= grid.geometry.get_real_coordinate({{
					state[0][0],
					state[0][1],
					state[0][2]
				}});

			Part_Pos(Part_Int(*cell_data)[i]) = {
				real_pos[0],
				real_pos[1],
				real_pos[2]
			};

			// remove from simulation if particle not inside of grid
			if (
				isnan(real_pos[0])
				or isnan(real_pos[1])
				or isnan(real_pos[2])
			) {

				Part_Int(*cell_data).erase(Part_Int(*cell_data).begin() + i);
				i--;

			// move to ext list if particle outside of current cell
			} else if (
				real_pos[0] < cell_min[0]
				or real_pos[0] > cell_max[0]
				or real_pos[1] < cell_min[1]
				or real_pos[1] > cell_max[1]
				or real_pos[2] < cell_min[2]
				or real_pos[2] > cell_max[2]
			) {
				uint64_t destination = dccrg::error_cell;

				for (const auto& neighbor_id: *neighbor_ids) {
					if (neighbor_id == dccrg::error_cell) {
						continue;
					}

					const auto
						neighbor_min = grid.geometry.get_min(neighbor_id),
						neighbor_max = grid.geometry.get_max(neighbor_id);

					if (
						real_pos[0] >= neighbor_min[0]
						and real_pos[0] <= neighbor_max[0]
						and real_pos[1] >= neighbor_min[1]
						and real_pos[1] <= neighbor_max[1]
						and real_pos[2] >= neighbor_min[2]
						and real_pos[2] <= neighbor_max[2]
					) {
						destination = neighbor_id;
						break;
					}
				}

				if (destination != dccrg::error_cell) {

					const auto index = Part_Ext(*cell_data).size();

					Part_Ext(*cell_data).resize(index + 1);
					assign(
						Part_Ext(*cell_data)[index],
						Part_Int(*cell_data)[i]
					);
					Part_Des(Part_Ext(*cell_data)[index]) = destination;

					Part_Int(*cell_data).erase(Part_Int(*cell_data).begin() + i);
					i--;

				} else {

					std::cerr << __FILE__ << "(" << __LINE__ << "): "
						<< " No destination found for particle at " << real_pos
						<< " propagated from " << Part_Pos(particle)
						<< " with dt " << dt
						<< " in cell " << cell_id
						<< " of length " << cell_length
						<< " at " << cell_center
						<< " with E " << JmV(*cell_data).cross(Mag(*cell_data))
						<< " and B " << Mag(*cell_data)
						<< " from neighbors ";
					for (const auto& neighbor_id: *neighbor_ids) {
						std::cerr << neighbor_id << " ";
					}
					std::cerr << std::endl;
					abort();
				}
			}
		}

		Nr_Ext(*cell_data) = Part_Ext(*cell_data).size();
	}

	return max_time_step;
}


template<
	class Nr_Particles_External_T,
	class Particles_External_T,
	class Cell
> void resize_receiving_containers(
	const std::vector<uint64_t>& cell_ids,
	dccrg::Dccrg<Cell, dccrg::Cartesian_Geometry>& grid
) {
	for (const auto& cell_id: cell_ids) {

		auto* const cell_ptr = grid[cell_id];
		if (cell_ptr == NULL) {
			std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}
		auto& cell_data = *cell_ptr;

		cell_data[Particles_External_T()]
			.resize(cell_data[Nr_Particles_External_T()]);
	}
}


template<
	class Nr_Particles_Internal_T,
	class Particles_Internal_T,
	class Particles_External_T,
	class Particle_Destination_T,
	class Cell
> void incorporate_external_particles(
	const std::vector<uint64_t>& cell_ids,
	dccrg::Dccrg<Cell, dccrg::Cartesian_Geometry>& grid
) {
	constexpr Nr_Particles_Internal_T Nr_Int{};
	constexpr Particles_Internal_T Part_Int{};
	constexpr Particles_External_T Part_Ext{};
	constexpr Particle_Destination_T Dest{};

	for (const auto& cell_id: cell_ids) {

		auto* const cell_ptr = grid[cell_id];
		if (cell_ptr == nullptr) {
			std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}
		auto& cell_data = *cell_ptr;

		// assign particles from neighbors' external list to this cell
		const auto* const neighbor_ids = grid.get_neighbors_of(cell_id);
		if (neighbor_ids == nullptr) {
			std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}

		for (const auto& neighbor_id: *neighbor_ids) {
			if (neighbor_id == dccrg::error_cell) {
				continue;
			}

			auto* const neighbor_ptr = grid[neighbor_id];
			if (neighbor_ptr == nullptr) {
				std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
				abort();
			}
			auto& neighbor_data = *neighbor_ptr;

			for (auto& particle: neighbor_data[Part_Ext]) {
				if (particle[Dest] == dccrg::error_cell) {
					continue;
				}

				if (particle[Dest] == cell_id) {
					particle[Dest] = dccrg::error_cell;

					const auto index = cell_data[Part_Int].size();

					cell_data[Part_Int].resize(index + 1);

					assign(cell_data[Part_Int][index], particle);
				}
			}
		}

		cell_data[Nr_Int] = cell_data[Part_Int].size();
	}
}


template<
	class Nr_Particles_External_T,
	class Particles_External_T,
	class Cell
> void remove_external_particles(
	const std::vector<uint64_t>& cell_ids,
	dccrg::Dccrg<Cell, dccrg::Cartesian_Geometry>& grid
) {
	for (auto cell_id: cell_ids) {

		auto* const cell_data = grid[cell_id];
		if (cell_data == nullptr) {
			std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}

		(*cell_data)[Nr_Particles_External_T()] = 0;
		(*cell_data)[Particles_External_T()].clear();
	}
}


}} // namespaces

#endif // ifndef PAMHD_PARTICLE_SOLVE_DCCRG_HPP
