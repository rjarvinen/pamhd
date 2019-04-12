/*
Particle data accumulator of PAMHD built on top of DCCRG.

Copyright 2015, 2016, 2017 Ilja Honkonen
Copyright 2019 Finnish Meteorological Institute
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

#ifndef PAMHD_PARTICLE_ACCUMULATE_DCCRG_HPP
#define PAMHD_PARTICLE_ACCUMULATE_DCCRG_HPP


#include "algorithm"
#include "iostream"
#include "utility"
#include "vector"

#include "dccrg.hpp"
#include "dccrg_cartesian_geometry.hpp"
#include "Eigen/Core"

#include "mhd/common.hpp"
#include "particle/accumulate.hpp"
#include "particle/common.hpp"
#include "particle/variables.hpp"


namespace pamhd {
namespace particle {


/*! Assumes geometry API is compatible with cartesian dccrg geometry */
template<class Geometry> std::tuple<
	Eigen::Vector3d,
	Eigen::Vector3d,
	Eigen::Vector3d,
	Eigen::Vector3d
> get_cell_geometry(
	const uint64_t cell_id,
	const Geometry& geometry
) {
	const auto
		cell_min_tmp = geometry.get_min(cell_id),
		cell_max_tmp = geometry.get_max(cell_id),
		cell_length_tmp = geometry.get_length(cell_id),
		cell_center_tmp = geometry.get_center(cell_id);
	const Eigen::Vector3d
		cell_min{cell_min_tmp[0], cell_min_tmp[1], cell_min_tmp[2]},
		cell_max{cell_max_tmp[0], cell_max_tmp[1], cell_max_tmp[2]},
		cell_length{cell_length_tmp[0], cell_length_tmp[1], cell_length_tmp[2]},
		cell_center{cell_center_tmp[0], cell_center_tmp[1], cell_center_tmp[2]};
	return std::make_tuple(cell_min, cell_max, cell_length, cell_center);
}


template<class Cell> std::tuple<
	std::vector<bool>,
	std::vector<uint64_t>,
	std::vector<Cell*>,
	std::vector<Eigen::Vector3d>,
	std::vector<Eigen::Vector3d>
> cache_neighbor_data(
	const uint64_t cell_id,
	dccrg::Dccrg<Cell, dccrg::Cartesian_Geometry>& grid
) {
	std::vector<bool> is_locals;
	std::vector<uint64_t> neighbor_ids;
	std::vector<Cell*> neighbor_datas;
	std::vector<Eigen::Vector3d> neighbor_mins, neighbor_maxs;

	Eigen::Vector3d cell_min, cell_max, cell_length, cell_center;
	std::tie(cell_min, cell_max, cell_length, cell_center) = get_cell_geometry(cell_id, grid.geometry);

	for (const auto& x_offset: {-1, 0, 1}) {
	for (const auto& y_offset: {-1, 0, 1}) {
	for (const auto& z_offset: {-1, 0, 1}) {
		const auto neighbors
			= grid.get_neighbors_of_at_offset(cell_id, x_offset, y_offset, z_offset);
		if (
			neighbors.size() == 0
			and (x_offset != 0 or y_offset != 0 or z_offset != 0)
		) {
			std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}

		for (const auto& neighbor_id: neighbors) {
			if (neighbor_id == dccrg::error_cell) {
				continue;
			}

			if (
				grid.get_refinement_level(neighbor_id) != grid.get_refinement_level(cell_id)
			) {
				std::cerr << __FILE__ << "(" << __LINE__ << ") "
					<< "Different cell refinement levels not supported"
					<< std::endl;
				abort();
			}

			neighbor_ids.emplace_back(neighbor_id);

			auto* const neighbor_data = grid[neighbor_id];
			if (neighbor_data == nullptr) {
				std::cerr << __FILE__ << "(" << __LINE__ << "): "
					<< "No data for neighbor " << neighbor_id
					<< " of cell " << cell_id
					<< std::endl;
				abort();
			}
			neighbor_datas.emplace_back(neighbor_data);

			if (grid.is_local(neighbor_id)) {
				is_locals.push_back(true);
			} else {
				is_locals.push_back(false);
			}

			Eigen::Vector3d neighbor_min, neighbor_max, neighbor_length, neighbor_center;
			std::tie(neighbor_min, neighbor_max, neighbor_length, neighbor_center)
				= get_cell_geometry(neighbor_id, grid.geometry);

			// handle periodic grid geometry
			if (x_offset < 0 and neighbor_center[0] > cell_min[0]) {
				neighbor_max[0] = cell_min[0];
				neighbor_min[0] = cell_min[0] - neighbor_length[0];
			} else if (x_offset > 0 and neighbor_center[0] < cell_max[0]) {
				neighbor_min[0] = cell_max[0];
				neighbor_max[0] = cell_max[0] + neighbor_length[0];
			}

			if (y_offset < 0 and neighbor_center[1] > cell_min[1]) {
				neighbor_max[1] = cell_min[1];
				neighbor_min[1] = cell_min[1] - neighbor_length[1];
			} else if (y_offset > 0 and neighbor_center[1] < cell_max[1]) {
				neighbor_min[1] = cell_max[1];
				neighbor_max[1] = cell_max[1] + neighbor_length[1];
			}

			if (z_offset < 0 and neighbor_center[2] > cell_min[2]) {
				neighbor_max[2] = cell_min[2];
				neighbor_min[2] = cell_min[2] - neighbor_length[2];
			} else if (z_offset > 0 and neighbor_center[2] < cell_max[2]) {
				neighbor_min[2] = cell_max[2];
				neighbor_max[2] = cell_max[2] + neighbor_length[2];
			}

			neighbor_mins.push_back(neighbor_min);
			neighbor_maxs.push_back(neighbor_max);
		}
	}}}

	return std::make_tuple(is_locals, neighbor_ids, neighbor_datas, neighbor_mins, neighbor_maxs);
}


/*!
Accumulates particle data in given cells to those cells and their neighbors.

Accumulated data from particles in the same cell is not zeroed before accumulating
unless clear_at_start == true;

Accumulated data is written to local cells directly, accumulated data to
remote neighbors is stored in local cells' accumulation list.

Updates the number of remote accumulated values.

Particle data to/from dont_solve cells isn't accumulated.
*/
template<
	class Particles_Getter,
	class Particle_Position_Getter,
	class Particle_Value_Getter,
	class Bulk_Value_Getter,
	class Bulk_Value_In_List_Getter,
	class Target_In_List_Getter,
	class Accumulation_List_Length_Getter,
	class Accumulation_List_Getter,
	class Solver_Info_Getter,
	class Cell_Iterator,
	class Cell_Data
> void accumulate(
	const Cell_Iterator& cells,
	dccrg::Dccrg<Cell_Data, dccrg::Cartesian_Geometry>& grid,
	Particles_Getter Part,
	Particle_Position_Getter Part_Pos,
	Particle_Value_Getter Part_Val,
	Bulk_Value_Getter Bulk_Val,
	Bulk_Value_In_List_Getter List_Bulk_Val,
	Target_In_List_Getter List_Target,
	Accumulation_List_Length_Getter List_Len,
	Accumulation_List_Getter Accu_List,
	Solver_Info_Getter Sol_Info,
	const bool clear_at_start = true
) {
	for (const auto& cell: cells) {
		if ((Sol_Info(*cell.data) & pamhd::particle::Solver_Info::dont_solve) > 0) {
			Bulk_Val(*cell.data) = {};
			continue;
		}

		Eigen::Vector3d cell_min, cell_max, cell_length, cell_center;
		std::tie(cell_min, cell_max, cell_length, cell_center) = get_cell_geometry(cell.id, grid.geometry);

		if (clear_at_start) {
			Accu_List(*cell.data).clear();
		}

		// TODO: faster to iterate over neighbors first?
		for (auto& particle: Part(*cell.data)) {
			auto& position = Part_Pos(particle);
			const Eigen::Vector3d
				value_box_min{
					position[0] - cell_length[0] / 2,
					position[1] - cell_length[1] / 2,
					position[2] - cell_length[2] / 2
				},
				value_box_max{
					position[0] + cell_length[0] / 2,
					position[1] + cell_length[1] / 2,
					position[2] + cell_length[2] / 2
				};

			// accumulate to current cell
			Bulk_Val(*cell.data)
				+= get_accumulated_value(
					Part_Val(*cell.data, particle),
					value_box_min,
					value_box_max,
					cell_min,
					cell_max
				);

			// accumulate to neighbors
			for (const auto& neighbor: cell.neighbors_of) {

				Eigen::Vector3d neigh_min, neigh_max, neigh_length, neigh_center;
				std::tie(neigh_min, neigh_max, neigh_length, neigh_center) = get_cell_geometry(neighbor.id, grid.geometry);
				if (
					value_box_min[0] > neigh_max[0]
					or value_box_min[1] > neigh_max[1]
					or value_box_min[2] > neigh_max[2]
					or value_box_max[0] < neigh_min[0]
					or value_box_max[1] < neigh_min[1]
					or value_box_max[2] < neigh_min[2]
				) {
					continue;
				}

				// don't accumulate into dont_solve cells
				if ((Sol_Info(*neighbor.data) & pamhd::particle::Solver_Info::dont_solve) > 0) {
					continue;
				}

				const auto accumulated_value
					= get_accumulated_value(
						Part_Val(*neighbor.data, particle),
						value_box_min,
						value_box_max,
						neigh_min,
						neigh_max
					);

				// same as for current cell above
				if (neighbor.is_local) {
					Bulk_Val(*neighbor.data) += accumulated_value;
				// accumulate values to a list in current cell
				} else {
					// use this index in the list
					size_t accumulation_index = 0;

					// find the index of target neighbor
					auto iter
						= std::find_if(
							Accu_List(*cell.data).begin(),
							Accu_List(*cell.data).end(),
							[&neighbor, &List_Target](
								const decltype(*Accu_List(*cell.data).begin()) candidate_item
							) {
								if (List_Target(candidate_item) == neighbor.id) {
									return true;
								} else {
									return false;
								}
							}
						);

					// found
					if (iter != Accu_List(*cell.data).end()) {
						List_Bulk_Val(*iter) += accumulated_value;
					// create the item
					} else {
						const auto old_size = Accu_List(*cell.data).size();
						Accu_List(*cell.data).resize(old_size + 1);
						auto& new_item = Accu_List(*cell.data)[old_size];
						List_Target(new_item) = neighbor.id;
						List_Bulk_Val(new_item) = accumulated_value;
					}
				}
			}
		}

		List_Len(*cell.data) = Accu_List(*cell.data).size();
	}
}


/*!
Same as accumulate() but uses get_accumulated_value_weighted() instead of
get_accumulated_value() and also records the total weight from all particles.
*/
template<
	class Particles_Getter,
	class Particle_Position_Getter,
	class Particle_Value_Getter,
	class Particle_Weight_Getter,
	class Bulk_Value_Getter,
	class Bulk_Value_In_List_Getter,
	class Target_In_List_Getter,
	class Accumulation_List_Length_Getter,
	class Accumulation_List_Getter,
	class Solver_Info_Getter,
	class Cell
> void accumulate_weighted(
	const std::vector<uint64_t>& cell_ids,
	dccrg::Dccrg<Cell, dccrg::Cartesian_Geometry>& grid,
	Particles_Getter Part,
	Particle_Position_Getter Part_Pos,
	Particle_Value_Getter Part_Val,
	Particle_Weight_Getter Part_Wei,
	Bulk_Value_Getter Bulk_Val,
	Bulk_Value_In_List_Getter List_Bulk_Val,
	Target_In_List_Getter List_Target,
	Accumulation_List_Length_Getter List_Len,
	Accumulation_List_Getter Accu_List,
	Solver_Info_Getter Sol_Info,
	const bool clear_at_start = true
) {
	for (const auto& cell_id: cell_ids) {
		auto* const cell_data = grid[cell_id];
		if (cell_data == nullptr) {
			std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}

		if ((Sol_Info(*cell_data) & pamhd::particle::Solver_Info::dont_solve) > 0) {
			Bulk_Val(*cell_data) = {};
			continue;
		}

		Eigen::Vector3d cell_min, cell_max, cell_length, cell_center;
		std::tie(cell_min, cell_max, cell_length, cell_center) = get_cell_geometry(cell_id, grid.geometry);

		if (clear_at_start) {
			Accu_List(*cell_data).clear();
		}

		std::vector<bool> is_locals;
		std::vector<uint64_t> neighbor_ids;
		std::vector<Cell*> neighbor_datas;
		std::vector<Eigen::Vector3d> neighbor_mins, neighbor_maxs;
		std::tie(
			is_locals,
			neighbor_ids,
			neighbor_datas,
			neighbor_mins,
			neighbor_maxs
		) = cache_neighbor_data(cell_id, grid);

		for (auto& particle: Part(*cell_data)) {
			auto& position = Part_Pos(particle);
			const Eigen::Vector3d
				value_box_min{
					position[0] - cell_length[0] / 2,
					position[1] - cell_length[1] / 2,
					position[2] - cell_length[2] / 2
				},
				value_box_max{
					position[0] + cell_length[0] / 2,
					position[1] + cell_length[1] / 2,
					position[2] + cell_length[2] / 2
				};

			// accumulate to current cell
			const auto accu_to_cell
				= get_accumulated_value_weighted(
					Part_Val(*cell_data, particle),
					Part_Wei(particle),
					value_box_min,
					value_box_max,
					cell_min,
					cell_max
				);
			Bulk_Val(*cell_data).first += accu_to_cell.first;
			// final weight of this particle's data in this cell
			Bulk_Val(*cell_data).second += accu_to_cell.second;

			// accumulate to neighbors
			for (size_t i = 0; i < is_locals.size(); i++) {

				if (
					value_box_min[0] > neighbor_maxs[i][0]
					or value_box_min[1] > neighbor_maxs[i][1]
					or value_box_min[2] > neighbor_maxs[i][2]
					or value_box_max[0] < neighbor_mins[i][0]
					or value_box_max[1] < neighbor_mins[i][1]
					or value_box_max[2] < neighbor_mins[i][2]
				) {
					continue;
				}

				// don't accumulate into dont_solve cells
				if ((Sol_Info(*(neighbor_datas[i])) & pamhd::particle::Solver_Info::dont_solve) > 0) {
					continue;
				}

				const auto accumulated_value
					= get_accumulated_value_weighted(
						Part_Val(*(neighbor_datas[i]), particle),
						Part_Wei(particle),
						value_box_min,
						value_box_max,
						neighbor_mins[i],
						neighbor_maxs[i]
					);

				// same as for current cell above
				if (is_locals[i]) {
					Bulk_Val(*(neighbor_datas[i])).first += accumulated_value.first;
					Bulk_Val(*(neighbor_datas[i])).second += accumulated_value.second;
				// accumulate values to a list in current cell
				} else {
					// use this index in the list
					size_t accumulation_index = 0;

					// find the index of target neighbor
					const auto neighbor_id = neighbor_ids[i];
					auto iter
						= std::find_if(
							Accu_List(*cell_data).begin(),
							Accu_List(*cell_data).end(),
							[&neighbor_id, &List_Target](
								const decltype(*Accu_List(*cell_data).begin()) candidate_item
							) {
								if (List_Target(candidate_item) == neighbor_id) {
									return true;
								} else {
									return false;
								}
							}
						);

					// found
					if (iter != Accu_List(*cell_data).end()) {
						List_Bulk_Val(*iter).first += accumulated_value.first;
						List_Bulk_Val(*iter).second += accumulated_value.second;
					// create the item
					} else {
						const auto old_size = Accu_List(*cell_data).size();
						Accu_List(*cell_data).resize(old_size + 1);
						auto& new_item = Accu_List(*cell_data)[old_size];
						List_Target(new_item) = neighbor_id;
						List_Bulk_Val(new_item).first = accumulated_value.first;
						List_Bulk_Val(new_item).second = accumulated_value.second;
					}
				}
			}
		}

		List_Len(*cell_data) = Accu_List(*cell_data).size();
	}
}


/*!
Allocates memory for lists of accumulated particle data in copies of remote neighbors of local cells.
*/
template<
	class Grid,
	class Accumulation_List_Getter,
	class Accumulation_List_Length_Getter
> void allocate_accumulation_lists(
	Grid& grid,
	Accumulation_List_Getter Accu_List,
	Accumulation_List_Length_Getter List_Len
) {
	for (const auto& remote_cell_id: grid.get_remote_cells_on_process_boundary()) {
		auto* const cell_data = grid[remote_cell_id];
		if (cell_data == nullptr) {
			std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}

		Accu_List(*cell_data).resize(List_Len(*cell_data));
	}
}


/*!
Adds accumulated particle data from remote neighbors' to local cells.
*/
template<
	class Grid,
	class Bulk_Value_Getter,
	class Value_In_List_Getter,
	class Target_In_List_Getter,
	class Accumulation_List_Getter,
	class Solver_Info_Getter
> void accumulate_from_remote_neighbors(
	Grid& grid,
	Bulk_Value_Getter Bulk_Val,
	Value_In_List_Getter Value_In_List,
	Target_In_List_Getter Target_In_List,
	Accumulation_List_Getter Accu_List,
	Solver_Info_Getter Sol_Info
) {
	for (const auto& remote_cell_id: grid.get_remote_cells_on_process_boundary()) {
		auto* const source_data = grid[remote_cell_id];
		if (source_data == nullptr) {
			std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}

		if ((Sol_Info(*source_data) & pamhd::particle::Solver_Info::dont_solve) > 0) {
			continue;
		}

		for (auto& item: Accu_List(*source_data)) {
			if (not grid.is_local(Target_In_List(item))) {
				continue;
			}

			auto* const target_data = grid[Target_In_List(item)];
			if (target_data == nullptr) {
				std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
				abort();
			}

			Bulk_Val(*target_data) += Value_In_List(item);
		}
	}
}

template<
	class Grid,
	class Bulk_Value_Getter,
	class Value_In_List_Getter,
	class Target_In_List_Getter,
	class Accumulation_List_Getter,
	class Solver_Info_Getter
> void accumulate_weighted_from_remote_neighbors(
	Grid& grid,
	Bulk_Value_Getter Bulk_Val,
	Value_In_List_Getter Value_In_List,
	Target_In_List_Getter Target_In_List,
	Accumulation_List_Getter Accu_List,
	Solver_Info_Getter Sol_Info
) {
	for (const auto& remote_cell_id: grid.get_remote_cells_on_process_boundary()) {
		auto* const source_data = grid[remote_cell_id];
		if (source_data == nullptr) {
			std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}

		if ((Sol_Info(*source_data) & pamhd::particle::Solver_Info::dont_solve) > 0) {
			continue;
		}

		for (auto& item: Accu_List(*source_data)) {
			if (not grid.is_local(Target_In_List(item))) {
				continue;
			}

			auto* const target_data = grid[Target_In_List(item)];
			if (target_data == nullptr) {
				std::cerr << __FILE__ << "(" << __LINE__ << ")" << std::endl;
				abort();
			}

			Bulk_Val(*target_data).first += Value_In_List(item).first;
			Bulk_Val(*target_data).second += Value_In_List(item).second;
		}
	}
}


/*!
Updates Bulk_Momentum.
*/
template<
	class Cell,
	class Particles_Getter,
	class Particle_Position_Getter,
	class Particle_Mass_Getter,
	class Particle_Species_Mass_Getter,
	class Particle_Velocity_Getter,
	class Particle_Relative_Kinetic_Energy_Getter,
	class Number_Of_Particles_Getter,
	class Particle_Weight_Getter,
	class Bulk_Mass_Getter,
	class Bulk_Momentum_Getter,
	class Bulk_Relative_Kinetic_Energy_Getter,
	class Number_Of_Particles_In_List_Getter,
	class Bulk_Mass_In_List_Getter,
	class Bulk_Momentum_In_List_Getter,
	class Bulk_Relative_Kinetic_Energy_In_List_Getter,
	class Bulk_Velocity_Getter,
	class Target_In_List_Getter,
	class Accumulation_List_Length_Getter,
	class Accumulation_List_Getter,
	class Accumulation_List_Length_Variable,
	class Accumulation_List_Variable,
	class Bulk_Velocity_Variable,
	class Solver_Info_Getter
> void accumulate_mhd_data(
	const std::vector<uint64_t>& inner_cell_ids,
	const std::vector<uint64_t>& outer_cell_ids,
	dccrg::Dccrg<Cell, dccrg::Cartesian_Geometry>& grid,
	Particles_Getter Particles,
	Particle_Position_Getter Particle_Position,
	Particle_Mass_Getter Particle_Mass,
	Particle_Species_Mass_Getter Particle_Species_Mass,
	Particle_Velocity_Getter Particle_Velocity,
	Particle_Relative_Kinetic_Energy_Getter Particle_Relative_Kinetic_Energy,
	Number_Of_Particles_Getter Number_Of_Particles,
	Particle_Weight_Getter Particle_Weight,
	Bulk_Mass_Getter Bulk_Mass,
	Bulk_Momentum_Getter Bulk_Momentum,
	Bulk_Relative_Kinetic_Energy_Getter Bulk_Relative_Kinetic_Energy,
	Bulk_Velocity_Getter Bulk_Velocity,
	Number_Of_Particles_In_List_Getter Accu_List_Number_Of_Particles,
	Bulk_Mass_In_List_Getter Accu_List_Bulk_Mass,
	Bulk_Momentum_In_List_Getter Accu_List_Bulk_Velocity,
	Bulk_Relative_Kinetic_Energy_In_List_Getter Accu_List_Bulk_Relative_Kinetic_Energy,
	Target_In_List_Getter Accu_List_Target,
	Accumulation_List_Length_Getter Accu_List_Length,
	Accumulation_List_Getter Accu_List,
	Accumulation_List_Length_Variable accu_list_len_var,
	Accumulation_List_Variable accu_list_var,
	Bulk_Velocity_Variable bulk_vel_var,
	Solver_Info_Getter Sol_Info
) {
	auto cell_ids = inner_cell_ids;
	cell_ids.insert(cell_ids.end(), outer_cell_ids.cbegin(), outer_cell_ids.cend());

	for (const auto& cell: cell_ids) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}
		Bulk_Mass(*cell_data) = 0;
		Bulk_Momentum(*cell_data) = {0, 0, 0};
		Bulk_Velocity(*cell_data).first = {0, 0, 0};
		Bulk_Velocity(*cell_data).second = 0;
	}

	accumulate(
		outer_cell_ids,
		grid,
		Particles,
		Particle_Position,
		Particle_Mass,
		Bulk_Mass,
		Accu_List_Bulk_Mass,
		Accu_List_Target,
		Accu_List_Length,
		Accu_List,
		Sol_Info
	);
	accumulate_weighted(
		outer_cell_ids,
		grid,
		Particles,
		Particle_Position,
		Particle_Velocity,
		Particle_Weight,
		Bulk_Velocity,
		Accu_List_Bulk_Velocity,
		Accu_List_Target,
		Accu_List_Length,
		Accu_List,
		Sol_Info,
		false // keep previous data in accumulation lists
	);

	Cell::set_transfer_all(true, accu_list_len_var);
	grid.start_remote_neighbor_copy_updates();

	accumulate(
		inner_cell_ids,
		grid,
		Particles,
		Particle_Position,
		Particle_Mass,
		Bulk_Mass,
		Accu_List_Bulk_Mass,
		Accu_List_Target,
		Accu_List_Length,
		Accu_List,
		Sol_Info,
		false
	);
	accumulate_weighted(
		inner_cell_ids,
		grid,
		Particles,
		Particle_Position,
		Particle_Velocity,
		Particle_Weight,
		Bulk_Velocity,
		Accu_List_Bulk_Velocity,
		Accu_List_Target,
		Accu_List_Length,
		Accu_List,
		Sol_Info,
		false
	);

	grid.wait_remote_neighbor_copy_update_receives();

	allocate_accumulation_lists(
		grid,
		Accu_List,
		Accu_List_Length
	);

	grid.wait_remote_neighbor_copy_update_sends();
	Cell::set_transfer_all(false, accu_list_len_var);

	Cell::set_transfer_all(true, accu_list_var);
	grid.start_remote_neighbor_copy_updates();
	grid.wait_remote_neighbor_copy_update_receives();

	accumulate_from_remote_neighbors(
		grid,
		Bulk_Mass,
		Accu_List_Bulk_Mass,
		Accu_List_Target,
		Accu_List,
		Sol_Info
	);
	accumulate_weighted_from_remote_neighbors(
		grid,
		Bulk_Velocity,
		Accu_List_Bulk_Velocity,
		Accu_List_Target,
		Accu_List,
		Sol_Info
	);

	grid.wait_remote_neighbor_copy_update_sends();
	Cell::set_transfer_all(false, accu_list_var);

	// scale velocities relative to total weights
	for (const auto& cell: cell_ids) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}
		if (Bulk_Velocity(*cell_data).second <= 0) {
			Bulk_Velocity(*cell_data).first = {0, 0, 0};
		} else {
			Bulk_Velocity(*cell_data).first /= Bulk_Velocity(*cell_data).second;
		}
		Bulk_Momentum(*cell_data) = Bulk_Velocity(*cell_data).first * Bulk_Mass(*cell_data);
	}


	/*
	Accumulate particle data required for pressure
	*/

	// needs remote neighbors' bulk velocity
	Cell::set_transfer_all(true, bulk_vel_var);
	grid.start_remote_neighbor_copy_updates();

	for (const auto& cell: inner_cell_ids) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}
		Number_Of_Particles(*cell_data)     =
		Bulk_Relative_Kinetic_Energy(*cell_data) = 0;
	}
	for (const auto& cell: outer_cell_ids) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}
		Number_Of_Particles(*cell_data)     =
		Bulk_Relative_Kinetic_Energy(*cell_data) = 0;
	}

	grid.wait_remote_neighbor_copy_update_receives();

	// returns number of particles of mass species mass
	const auto particle_counter
		= [
			&Particle_Mass,
			&Particle_Species_Mass
		](
			Cell& cell,
			// TODO:switch to auto in c++17
			decltype(*Particles(*(grid[0])).begin())& particle
		) ->double {
			return Particle_Mass(cell, particle) / Particle_Species_Mass(cell, particle);
		};
	// TODO: merge with bulk mass accumulation
	accumulate(
		outer_cell_ids,
		grid,
		Particles,
		Particle_Position,
		particle_counter,
		Number_Of_Particles,
		Accu_List_Number_Of_Particles,
		Accu_List_Target,
		Accu_List_Length,
		Accu_List,
		Sol_Info
	);
	accumulate(
		outer_cell_ids,
		grid,
		Particles,
		Particle_Position,
		Particle_Relative_Kinetic_Energy,
		Bulk_Relative_Kinetic_Energy,
		Accu_List_Bulk_Relative_Kinetic_Energy,
		Accu_List_Target,
		Accu_List_Length,
		Accu_List,
		Sol_Info,
		false
	);

	grid.wait_remote_neighbor_copy_update_sends();
	Cell::set_transfer_all(false, bulk_vel_var);


	Cell::set_transfer_all(true, accu_list_len_var);
	grid.start_remote_neighbor_copy_updates();

	accumulate(
		inner_cell_ids,
		grid,
		Particles,
		Particle_Position,
		particle_counter,
		Number_Of_Particles,
		Accu_List_Number_Of_Particles,
		Accu_List_Target,
		Accu_List_Length,
		Accu_List,
		Sol_Info,
		false
	);
	accumulate(
		inner_cell_ids,
		grid,
		Particles,
		Particle_Position,
		Particle_Relative_Kinetic_Energy,
		Bulk_Relative_Kinetic_Energy,
		Accu_List_Bulk_Relative_Kinetic_Energy,
		Accu_List_Target,
		Accu_List_Length,
		Accu_List,
		Sol_Info,
		false
	);

	grid.wait_remote_neighbor_copy_update_receives();

	allocate_accumulation_lists(
		grid,
		Accu_List,
		Accu_List_Length
	);

	grid.wait_remote_neighbor_copy_update_sends();
	Cell::set_transfer_all(false, accu_list_len_var);

	Cell::set_transfer_all(true, accu_list_var);
	grid.start_remote_neighbor_copy_updates();
	grid.wait_remote_neighbor_copy_update_receives();

	accumulate_from_remote_neighbors(
		grid,
		Number_Of_Particles,
		Accu_List_Number_Of_Particles,
		Accu_List_Target,
		Accu_List,
		Sol_Info
	);
	accumulate_from_remote_neighbors(
		grid,
		Bulk_Relative_Kinetic_Energy,
		Accu_List_Bulk_Relative_Kinetic_Energy,
		Accu_List_Target,
		Accu_List,
		Sol_Info
	);

	grid.wait_remote_neighbor_copy_update_sends();
	Cell::set_transfer_all(false, accu_list_var);
}


/*!
Converts bulk particle data to conservative MHD form.

Skips a cell if it doesn't have particles and no_particles_allowed == true.
*/
template<
	class Cell,
	class Geometry,
	class Number_Of_Particles_Getter,
	class Particle_Bulk_Mass_Getter,
	class Particle_Bulk_Momentum_Getter,
	class Particle_Bulk_Relative_Kinetic_Energy_Getter,
	class Particle_List_Getter,
	class MHD_Mass_Getter,
	class MHD_Momentum_Getter,
	class MHD_Energy_Getter,
	class MHD_Magnetic_Field_Getter,
	class Solver_Info_Getter
> void fill_mhd_fluid_values(
	const std::vector<uint64_t>& cell_ids,
	dccrg::Dccrg<Cell, Geometry>& grid,
	const double adiabatic_index,
	const double vacuum_permeability,
	const double particle_temp_nrj_ratio,
	const double minimum_pressure,
	Number_Of_Particles_Getter Number_Of_Particles,
	Particle_Bulk_Mass_Getter Particle_Bulk_Mass,
	Particle_Bulk_Momentum_Getter Particle_Bulk_Momentum,
	Particle_Bulk_Relative_Kinetic_Energy_Getter Particle_Bulk_Relative_Kinetic_Energy,
	Particle_List_Getter Particle_List,
	MHD_Mass_Getter MHD_Mass,
	MHD_Momentum_Getter MHD_Momentum,
	MHD_Energy_Getter MHD_Energy,
	MHD_Magnetic_Field_Getter MHD_Magnetic_Field,
	Solver_Info_Getter Sol_Info
) {
	for (const auto& cell: cell_ids) {
		auto* const cell_data = grid[cell];
		if (cell_data == nullptr) {
			std::cerr <<  __FILE__ << "(" << __LINE__ << ")" << std::endl;
			abort();
		}

		if ((Sol_Info(*cell_data) & pamhd::particle::Solver_Info::dont_solve) > 0) {
			MHD_Mass(*cell_data) = 0;
			MHD_Momentum(*cell_data) = {0, 0, 0};
			MHD_Energy(*cell_data) = 0;
			continue;
		}

		if (Particle_Bulk_Mass(*cell_data) <= 0) {
			MHD_Mass(*cell_data) = 0;
			MHD_Momentum(*cell_data) = {0, 0, 0};
			MHD_Energy(*cell_data) = 0;
			continue;
		}

		const auto length = grid.geometry.get_length(cell);
		const auto volume = length[0] * length[1] * length[2];

		MHD_Mass(*cell_data) = Particle_Bulk_Mass(*cell_data) / volume;
		MHD_Momentum(*cell_data) = Particle_Bulk_Momentum(*cell_data) / volume;

		const double pressure = [&](){
			if (Particle_Bulk_Mass(*cell_data) <= 0) {
				return 0.0;
			} else {
				return std::max(
					minimum_pressure,
					2 * Particle_Bulk_Relative_Kinetic_Energy(*cell_data) / 3 / volume
				);
			}
		}();

		if (MHD_Mass(*cell_data) <= 0) {
			MHD_Energy(*cell_data)
				= pressure / (adiabatic_index - 1)
				+ MHD_Magnetic_Field(*cell_data).squaredNorm() / vacuum_permeability / 2;
		} else {
			MHD_Energy(*cell_data)
				= pressure / (adiabatic_index - 1)
				+ MHD_Momentum(*cell_data).squaredNorm() / MHD_Mass(*cell_data) / 2
				+ MHD_Magnetic_Field(*cell_data).squaredNorm() / vacuum_permeability / 2;
		}
	}
}


}} // namespaces

#endif // ifndef PAMHD_PARTICLE_ACCUMULATE_DCCRG_HPP
