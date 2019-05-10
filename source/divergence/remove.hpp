/*
Functions for working with divergence of vector field.

Copyright 2014, 2015, 2016, 2017 Ilja Honkonen
Copyright 2018, 2019 Finnish Meteorological Institute
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

#ifndef PAMHD_DIVERGENCE_REMOVE_HPP
#define PAMHD_DIVERGENCE_REMOVE_HPP

#include "cmath"
#include "limits"
#include "vector"

#include "dccrg.hpp"
#include "dccrg_cartesian_geometry.hpp"
#include "mpi.h"
#include "tests/poisson/poisson_solve.hpp" // part of dccrg


namespace pamhd {
namespace divergence {


/*!
Calculates divergence of vector variable.

Divergence is calculated in all @cells for
which @Cell_Type returns 0 or 1.

Returns the total divergence i.e. sum of absolute
divergence in given cells of all processes divided
by number of cells on all processes in which divergence
was calculated.

Variable returned by @Vector must have a defined value
in face neighbors of cells where divergence is calculated.
Dimension(s) where at least one neighbor is missing
doesn't contribute to divergence.

@Cell_Type must be an object that, when given a reference
to data of one grid cell, returns 0 or 1 if divergence
should be calculated for that cell and -1 otherwise.
@Vector must be an object that, when given a reference
to the data of one grid cell, returns a *reference* to
the data from which divergence should be calculated.
Similarly @Divergence should return a *reference* to
data in which calculated divergence should be stored.

Example:

struct Cell_Data {
	std::array<double, 3> vector_data;
	double scalar_data;
	int calculate_div;
	std::tuple<...> get_mpi_datatype() {...}
};

dccrg::Dccrg<Cell_Data, ...> grid;

pamhd::divergence::get_divergence(
	grid.local_cells(),
	grid,
	[](Cell_Data& cell_data)->std::array<double, 3>& {
		return cell_data.vector_data;
	},
	[](Cell_Data& cell_data)->double& {
		return cell_data.scalar_data;
	},
	[](Cell_Data& cell_data)->int& {
		return cell_data.calculate_div;
	}
);
*/
template <
	class Cell_Iterator,
	class Grid,
	class Vector_Getter,
	class Divergence_Getter,
	class Cell_Type_Getter
> double get_divergence(
	const Cell_Iterator& cells,
	Grid& grid,
	Vector_Getter Vector,
	Divergence_Getter Divergence,
	Cell_Type_Getter Cell_Type
) {
	double local_divergence = 0, global_divergence = 0;
	uint64_t local_calculated_cells = 0, global_calculated_cells = 0;
	for (const auto& cell: cells) {
		if (Cell_Type(*cell.data) < 0) {
			continue;
		}

		// get distance between neighbors in same dimension
		std::array<double, 3>
			// distance from current cell on neg and pos side (> 0)
			neigh_neg_dist{{0, 0, 0}},
			neigh_pos_dist{{0, 0, 0}};

		// number of neighbors in each dimension
		std::array<size_t, 3> nr_neighbors{{0, 0, 0}};

		const auto cell_length = grid.geometry.get_length(cell.id);
		const int cell_size = grid.mapping.get_cell_length_in_indices(cell.id);
		for (const auto& neighbor: cell.neighbors_of) {
			if (Cell_Type(*cell.data) < 0) {
				continue;
			}

			// only calculate between face neighbors
			int overlaps = 0, neighbor_dir = 0;

			const int neigh_size = grid.mapping.get_cell_length_in_indices(neighbor.id);
			if (neighbor.x < cell_size and neighbor.x > -neigh_size) {
				overlaps++;
			} else if (neighbor.x == cell_size) {
				neighbor_dir = 1;
			} else if (neighbor.x == -neigh_size) {
				neighbor_dir = -1;
			}

			if (neighbor.y < cell_size and neighbor.y > -neigh_size) {
				overlaps++;
			} else if (neighbor.y == cell_size) {
				neighbor_dir = 2;
			} else if (neighbor.y == -neigh_size) {
				neighbor_dir = -2;
			}

			if (neighbor.z < cell_size and neighbor.z > -neigh_size) {
				overlaps++;
			} else if (neighbor.z == cell_size) {
				neighbor_dir = 3;
			} else if (neighbor.z == -neigh_size) {
				neighbor_dir = -3;
			}

			if (overlaps < 2) {
				continue;
			}
			if (neighbor_dir == 0) {
				continue;
			}

			const size_t dim = std::abs(neighbor_dir) - 1;

			nr_neighbors[dim]++;

			const auto neighbor_length
				= grid.geometry.get_length(neighbor.id);

			const double distance
				= (cell_length[dim] + neighbor_length[dim]) / 2.0;

			if (neighbor_dir < 0) {
				neigh_neg_dist[dim] = distance;
			} else {
				neigh_pos_dist[dim] = distance;
			}
		}

		bool have_enough_neighbors = false;
		for (auto dim = 0; dim < 3; dim++) {
			if (
				nr_neighbors[dim] == 1 + 1
				or nr_neighbors[dim] == 1 + 4
				or nr_neighbors[dim] == 4 + 4
			) {
				have_enough_neighbors = true;
				break;
			}
		}

		if (not have_enough_neighbors) {
			continue;
		}
		local_calculated_cells++;

		auto& div = Divergence(*cell.data);
		div = 0;

		const auto cell_ref_lvl = grid.get_refinement_level(cell.id);

		for (const auto& neighbor: cell.neighbors_of) {

			int overlaps = 0, neighbor_dir = 0;

			const int neigh_size = grid.mapping.get_cell_length_in_indices(neighbor.id);
			if (neighbor.x < cell_size and neighbor.x > -neigh_size) {
				overlaps++;
			} else if (neighbor.x == cell_size) {
				neighbor_dir = 1;
			} else if (neighbor.x == -neigh_size) {
				neighbor_dir = -1;
			}

			if (neighbor.y < cell_size and neighbor.y > -neigh_size) {
				overlaps++;
			} else if (neighbor.y == cell_size) {
				neighbor_dir = 2;
			} else if (neighbor.y == -neigh_size) {
				neighbor_dir = -2;
			}

			if (neighbor.z < cell_size and neighbor.z > -neigh_size) {
				overlaps++;
			} else if (neighbor.z == cell_size) {
				neighbor_dir = 3;
			} else if (neighbor.z == -neigh_size) {
				neighbor_dir = -3;
			}

			if (overlaps < 2) {
				continue;
			}
			if (neighbor_dir == 0) {
				continue;
			}

			const size_t dim = std::abs(neighbor_dir) - 1;

			if (
				nr_neighbors[dim] != 2
				and nr_neighbors[dim] != 5
				and nr_neighbors[dim] != 8
			) {
				continue;
			}

			double multiplier = 1 / (neigh_pos_dist[dim] + neigh_neg_dist[dim]);
			if (neighbor_dir < 0) {
				multiplier *= -1;
			}

			const auto neigh_ref_lvl = grid.get_refinement_level(neighbor.id);
			if (neigh_ref_lvl > cell_ref_lvl) {
				multiplier /= 4;
			}

			div += multiplier * Vector(*neighbor.data)[dim];
		}

		local_divergence += std::fabs(div);
	}

	MPI_Comm comm = grid.get_communicator();
	MPI_Allreduce(
		&local_divergence,
		&global_divergence,
		1,
		MPI_DOUBLE,
		MPI_SUM,
		comm
	);
	MPI_Allreduce(
		&local_calculated_cells,
		&global_calculated_cells,
		1,
		MPI_UINT64_T,
		MPI_SUM,
		comm
	);
	MPI_Comm_free(&comm);

	return global_divergence / global_calculated_cells;
}


/*!
Calculates gradient of scalar variable.

See get_divergence() for more info on arguments, etc.
*/
template <
	class Cell_Iterator,
	class Grid,
	class Scalar_Getter,
	class Gradient_Getter,
	class Cell_Type_Getter
> void get_gradient(
	const Cell_Iterator& cells,
	Grid& grid,
	Scalar_Getter Scalar,
	Gradient_Getter Gradient,
	Cell_Type_Getter Cell_Type
) {
	for (const auto& cell: cells) {
		if (Cell_Type(*cell.data) < 0) {
			continue;
		}

		// get distance between neighbors in same dimension
		std::array<double, 3>
			// distance from current cell on neg and pos side (> 0)
			neigh_neg_dist{{0, 0, 0}},
			neigh_pos_dist{{0, 0, 0}};

		// number of neighbors in each dimension
		std::array<size_t, 3> nr_neighbors{{0, 0, 0}};

		const auto cell_length = grid.geometry.get_length(cell.id);
		const auto face_neighbors_of = grid.get_face_neighbors_of(cell.id);
		for (const auto& item: face_neighbors_of) {
			const auto neighbor = item.first;
			const auto direction = item.second;
			if (direction == 0 or std::abs(direction) > 3) {
				std::cerr << __FILE__ << "(" << __LINE__<< ")" << std::endl;
				abort();
			}
			const size_t dim = std::abs(direction) - 1;

			auto* const neighbor_data = grid[neighbor];
			if (neighbor_data == nullptr) {
				std::cerr << __FILE__ << "(" << __LINE__<< ")" << std::endl;
				abort();
			}

			if (Cell_Type(*neighbor_data) < 0) {
				continue;
			}

			nr_neighbors[dim]++;

			const auto neighbor_length
				= grid.geometry.get_length(neighbor);

			const double distance
				= (cell_length[dim] + neighbor_length[dim]) / 2.0;

			if (direction < 0) {
				neigh_neg_dist[dim] = distance;
			} else {
				neigh_pos_dist[dim] = distance;
			}
		}

		bool have_enough_neighbors = false;
		for (auto dim = 0; dim < 3; dim++) {
			if (
				nr_neighbors[dim] == 1 + 1
				or nr_neighbors[dim] == 1 + 4
				or nr_neighbors[dim] == 4 + 4
			) {
				have_enough_neighbors = true;
				break;
			}
		}

		auto& gradient = Gradient(*cell.data);
		gradient[0] =
		gradient[1] =
		gradient[2] = 0;

		if (not have_enough_neighbors) {
			continue;
		}

		const auto cell_ref_lvl = grid.get_refinement_level(cell.id);

		for (const auto& item: face_neighbors_of) {
			const auto neighbor = item.first;
			const auto direction = item.second;
			const size_t dim = std::abs(direction) - 1;

			if (
				nr_neighbors[dim] != 2
				and nr_neighbors[dim] != 5
				and nr_neighbors[dim] != 8
			) {
				continue;
			}

			double multiplier = 1 / (neigh_pos_dist[dim] + neigh_neg_dist[dim]);
			if (direction < 0) {
				multiplier *= -1;
			}

			const auto neigh_ref_lvl = grid.get_refinement_level(neighbor);
			if (neigh_ref_lvl > cell_ref_lvl) {
				multiplier /= 4;
			}

			auto* const neighbor_data = grid[neighbor];
			if (neighbor_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__<< "): "
					<< "No data for neighbor " << neighbor
					<< " of cell " << cell.id
					<< std::endl;
				abort();
			}

			gradient[dim] += multiplier * Scalar(*neighbor_data);
		}
	}
}


/*!
Calculates curl of vector variable.

See get_divergence() for more info on arguments, etc.
*/
template <
	class Cell_Iterator,
	class Grid,
	class Vector_Getter,
	class Curl_Getter,
	class Cell_Type_Getter
> void get_curl(
	const Cell_Iterator& cells,
	Grid& grid,
	Vector_Getter Vector,
	Curl_Getter Curl,
	Cell_Type_Getter Cell_Type
) {
	for (const auto& cell: cells) {
		if (Cell_Type(*cell.data) < 0) {
			continue;
		}

		if (grid.get_refinement_level(cell.id) != 0) {
			std::cerr <<  __FILE__ << "(" << __LINE__<< "): "
				<< "Adaptive mesh refinement not supported"
				<< std::endl;
			abort();
		}

		const auto cell_length = grid.geometry.get_length(cell.id);

		const auto face_neighbors_of = grid.get_face_neighbors_of(cell.id);

		// get distance between neighbors in same dimension
		std::array<double, 3>
			// distance from current cell on neg and pos side (> 0)
			neigh_neg_dist{{0, 0, 0}},
			neigh_pos_dist{{0, 0, 0}};

		// number of neighbors in each dimension
		std::array<size_t, 3> nr_neighbors{{0, 0, 0}};

		for (const auto& item: face_neighbors_of) {
			const auto neighbor = item.first;
			const auto direction = item.second;
			if (direction == 0 or std::abs(direction) > 3) {
				std::cerr << __FILE__ << "(" << __LINE__<< ")" << std::endl;
				abort();
			}
			const size_t dim = std::abs(direction) - 1;

			auto* const neighbor_data = grid[neighbor];
			if (neighbor_data == nullptr) {
				std::cerr << __FILE__ << "(" << __LINE__<< ")" << std::endl;
				abort();
			}

			if (Cell_Type(*neighbor_data) < 0) {
				continue;
			}

			nr_neighbors[dim]++;

			const auto neighbor_length
				= grid.geometry.get_length(neighbor);

			const double distance
				= (cell_length[dim] + neighbor_length[dim]) / 2.0;

			if (direction < 0) {
				neigh_neg_dist[dim] = distance;
			} else {
				neigh_pos_dist[dim] = distance;
			}
		}

		bool have_enough_neighbors = false;
		for (auto dim = 0; dim < 3; dim++) {
			if (nr_neighbors[dim] == 2) {
				have_enough_neighbors = true;
			}
		}

		auto
			&vec = Vector(*cell.data),
			&curl = Curl(*cell.data);

		curl[0] =
		curl[1] =
		curl[2] = 0;

		if (not have_enough_neighbors) {
			continue;
		}

		/*
		curl_0 = diff_vec_2 / diff_pos_1 - diff_vec_1 / diff_pos_2
		curl_1 = diff_vec_0 / diff_pos_2 - diff_vec_2 / diff_pos_0
		curl_2 = diff_vec_1 / diff_pos_0 - diff_vec_0 / diff_pos_1
		*/
		for (auto dim0 = 0; dim0 < 3; dim0++) {

			const auto
				dim1 = (dim0 + 1) % 3,
				dim2 = (dim0 + 2) % 3;

			// zero in dimensions with missing neighbor(s)
			if (nr_neighbors[dim1] == 2) {
				curl[dim0] += vec[dim2] * (neigh_pos_dist[dim1] - neigh_neg_dist[dim1]);
			}
			if (nr_neighbors[dim2] == 2) {
				curl[dim0] -= vec[dim1] * (neigh_pos_dist[dim2] - neigh_neg_dist[dim2]);
			}
		}

		for (const auto& item: face_neighbors_of) {
			const auto neighbor = item.first;
			const auto direction = item.second;
			const size_t dim0 = std::abs(direction) - 1;

			if (nr_neighbors[dim0] != 2) {
				continue;
			}

			const auto
				dim1 = (dim0 + 1) % 3,
				dim2 = (dim0 + 2) % 3;

			double multiplier = 0;
			if (direction < 0) {
				multiplier = -neigh_pos_dist[dim0] / neigh_neg_dist[dim0];
			} else {
				multiplier = neigh_neg_dist[dim0] / neigh_pos_dist[dim0];
			}
			multiplier /= (neigh_pos_dist[dim0] + neigh_neg_dist[dim0]);

			auto* const neighbor_data = grid[neighbor];
			if (neighbor_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__<< "): "
					<< "No data for neighbor " << neighbor
					<< " of cell " << cell.id
					<< std::endl;
				abort();
			}
			auto& neigh_vec = Vector(*neighbor_data);

			curl[dim2] += multiplier * neigh_vec[dim1];
			curl[dim1] -= multiplier * neigh_vec[dim2];
		}
	}
}


/*!
Removes divergence of a vector variable.

Returns total divergence of vector variable
(from get_divergence()) before removing divergence.

Solves phi from div(grad(phi)) = div(vector) and assigns
vector = vector - grad(phi) after which div(vector) -> 0.

Vector variable must have been updated between processes
before calling this function.

Vector, Divergence and Gradient should return a reference
to the variable's data.

See get_divergence() for more info on arguments, etc.

The transfer of Vector and Divergence variables' data
must have been switched on in cells of simulation grid
before calling this function.

The arguments max_iterations to verbose are given
directly to the constructor of dccrg Poisson equation
solver class:
https://gitorious.org/dccrg/dccrg/source/master:tests/poisson/poisson_solve.hpp

Poisson solver is applied to previous results retries
number of times in a row, i.e. solver is used once
if retries == 0.
*/
template <
	class Cell_Iterator,
	class Grid,
	class Vector_Getter,
	class Divergence_Getter,
	class Gradient_Getter,
	class Cell_Type_Getter
	// TODO: add possibility to reuse solution from previous call
> double remove(
	const Cell_Iterator& cells,
	Grid& grid,
	Vector_Getter Vector,
	Divergence_Getter Divergence,
	Gradient_Getter Gradient,
	Cell_Type_Getter Cell_Type,
	const unsigned int max_iterations = 1000,
	const unsigned int min_iterations = 0,
	const double stop_residual = 1e-15,
	const double p_of_norm = 2,
	const double stop_after_residual_increase = 10,
	const unsigned int retries = 0,
	const bool use_failsafe = false,
	const bool verbose = false
) {
	/*
	Prepare solution grid and source term
	*/

	grid.update_copies_of_remote_neighbors();

	// rhs for poisson solver = div(vec)
	const double div_before = get_divergence(
		cells,
		grid,
		Vector,
		Divergence,
		Cell_Type
	);
	// zero divergence in boundary cells
	for (const auto& cell: cells) {
		if (Cell_Type(*cell.data) == 0) {
			Divergence(*cell.data) = 0;
		}
	}

	dccrg::Dccrg<Poisson_Cell, typename Grid::geometry_type> poisson_grid(grid);

	// transfer rhs to poisson grid
	for (const auto& cell: cells) {
		auto* const poisson_data = poisson_grid[cell.id];
		if (poisson_data == nullptr) {
			std::cerr <<  __FILE__ << "(" << __LINE__<< "): "
				<< "No data for poisson cell " << cell.id
				<< std::endl;
			abort();
		}

		if (Cell_Type(*cell.data) != 1) {
			poisson_data->cell_type = DCCRG_POISSON_BOUNDARY_CELL;
			continue;
		}

		poisson_data->solution = 0;
		poisson_data->rhs = Divergence(*cell.data);
		poisson_data->cell_type = DCCRG_POISSON_SOLVE_CELL;
	}

	Poisson_Solve solver(
		max_iterations,
		min_iterations,
		stop_residual,
		p_of_norm,
		stop_after_residual_increase,
		verbose
	);

	// solve phi in div(grad(phi)) = rhs
	std::vector<uint64_t> solve_cells, skip_cells;
	for (const auto& cell: cells) {
		const auto type = Cell_Type(*cell.data);
		switch (type) {
		case 1:
			solve_cells.push_back(cell.id);
			break;
		case -1:
			skip_cells.push_back(cell.id);
			break;
		default:
			break;
		}
	}
	if (use_failsafe) {
		solver.solve_failsafe(solve_cells, poisson_grid, skip_cells);
	} else {
		size_t iters = 0;
		while (iters <= retries) {
			solver.solve(
				solve_cells,
				poisson_grid,
				skip_cells,
				[iters](){
					if (iters == 0) {
						return false;
					} else {
						return true;
					}
				}()
			);
			iters++;
		}
	}

	/*
	Remove divergence with Vec = Vec - grad(phi)
	*/

	// store phi (solution) in divergence variable
	for (const auto& cell: cells) {
		if (Cell_Type(*cell.data) != 1) {
			continue;
		}

		auto* const poisson_data = poisson_grid[cell.id];
		if (poisson_data == nullptr) {
			std::cerr <<  __FILE__ << "(" << __LINE__<< "): "
				<< "No data for poisson cell " << cell.id
				<< std::endl;
			abort();
		}

		Divergence(*cell.data) = poisson_data->solution;
	}

	grid.update_copies_of_remote_neighbors();

	get_gradient(
		cells,
		grid,
		Divergence,
		Gradient,
		Cell_Type
	);

	for (const auto& cell: cells) {
		if (Cell_Type(*cell.data) != 1) {
			continue;
		}
		for (size_t dim = 0; dim < 3; dim++) {
			Vector(*cell.data)[dim] -= Gradient(*cell.data)[dim];
		}
	}

	// clean up divergence variable
	for (const auto& cell: cells) {
		if (Cell_Type(*cell.data) == 0) {
			Divergence(*cell.data) = 0;
		}
	}

	return div_before;
}


/*!
Calculates curl(curl(vector)) of vector variable.

See get_divergence() for more info on arguments, etc.
*/
template <
	class Cell_Iterator,
	class Grid,
	class Vector_Getter,
	class Result_Getter,
	class Cell_Type_Getter
> void get_curl_curl(
	const Cell_Iterator& cells,
	Grid& grid,
	Vector_Getter Vector,
	Result_Getter Result,
	Cell_Type_Getter Cell_Type
) {
	if (grid.get_maximum_refinement_level() > 0) {
		std::cerr <<  __FILE__ << "(" << __LINE__<< "): "
			<< "Adaptive mesh refinement not supported"
			<< std::endl;
		abort();
	}

	for (const auto& cell: cells) {
		if (Cell_Type(*cell.data) < 0) {
			continue;
		}

		const auto cell_length = grid.geometry.get_length(cell.id);

		const auto face_neighbors_of = grid.get_face_neighbors_of(cell.id);

		// get distance between neighbors in same dimension
		std::array<double, 3>
			// distance from current cell on neg and pos side (> 0)
			neigh_neg_dist{{0, 0, 0}},
			neigh_pos_dist{{0, 0, 0}};

		// number of neighbors in each dimension
		std::array<size_t, 3> nr_neighbors{{0, 0, 0}};
		// pointers to neighbor data, -x, +x, -y, +y, -z, +z
		std::array<typename Grid::cell_data_type*, 6> neighbor_datas{nullptr};

		for (const auto& item: face_neighbors_of) {
			const auto neighbor = item.first;
			const auto direction = item.second;
			if (direction == 0 or std::abs(direction) > 3) {
				std::cerr << __FILE__ << "(" << __LINE__<< ")" << std::endl;
				abort();
			}
			const size_t dim = std::abs(direction) - 1;

			auto* const neighbor_data = grid[neighbor];
			if (neighbor_data == nullptr) {
				std::cerr << __FILE__ << "(" << __LINE__<< ")" << std::endl;
				abort();
			}

			if (Cell_Type(*neighbor_data) < 0) {
				continue;
			}

			nr_neighbors[dim]++;

			const auto neighbor_length
				= grid.geometry.get_length(neighbor);

			const double distance
				= (cell_length[dim] + neighbor_length[dim]) / 2.0;

			if (direction < 0) {
				neigh_neg_dist[dim] = distance;
			} else {
				neigh_pos_dist[dim] = distance;
			}

			size_t index = dim * 2;
			if (direction > 0) {
				index++;
			}
			neighbor_datas[index] = neighbor_data;
		}

		bool have_enough_neighbors = false;
		for (auto dim = 0; dim < 3; dim++) {
			if (nr_neighbors[dim] == 2) {
				have_enough_neighbors = true;
			}
		}

		auto
			&vec = Vector(*cell.data),
			&result = Result(*cell.data);

		result[0] =
		result[1] =
		result[2] = 0;

		if (not have_enough_neighbors) {
			continue;
		}

		/*
		result_x = dydxVy - dydyVx - dzdzVx + dzdxVz
		result_y = dzdyVz - dzdzVy - dxdxVy + dxdyVx
		result_z = dxdzVx - dxdxVz - dydyVz + dydzVy

		dxdyA = dydxA = [A(0, 1) + A(1, 0) - A(-1, 0) - A(0, -1)] / 2 / sqrt(dx*dx+dy*dy)
		dxdxA = 2*[-A(0, 0) + A(1, 0)/2 + A(-1, 0)/2]/dx/dx
		...
		*/

		// result_x
		if (nr_neighbors[1] == 2) {
			// -dydyVx
			result[0] -= 2 * (
				-vec[0] / neigh_neg_dist[1] / neigh_pos_dist[1]
				+ Vector(*neighbor_datas[3])[0] / neigh_pos_dist[1] / (neigh_pos_dist[1] + neigh_neg_dist[1])
				+ Vector(*neighbor_datas[2])[0] / neigh_neg_dist[1] / (neigh_pos_dist[1] + neigh_neg_dist[1])
			);
			// dydxVy
			if (nr_neighbors[0] == 2) {
				result[0] += 0.5 * (
					Vector(*neighbor_datas[3])[1]
					+ Vector(*neighbor_datas[1])[1]
					- Vector(*neighbor_datas[0])[1]
					- Vector(*neighbor_datas[2])[1]
				) / std::sqrt(
					neigh_pos_dist[0]*neigh_pos_dist[0]
					+ neigh_pos_dist[1]*neigh_pos_dist[1]
				);
			}
		}
		if (nr_neighbors[2] == 2) {
			// -dzdzVx
			result[0] -= 2 * (
				-vec[0] / neigh_neg_dist[2] / neigh_pos_dist[2]
				+ Vector(*neighbor_datas[5])[0] / neigh_pos_dist[2] / (neigh_pos_dist[2] + neigh_neg_dist[2])
				+ Vector(*neighbor_datas[4])[0] / neigh_neg_dist[2] / (neigh_pos_dist[2] + neigh_neg_dist[2])
			);
			// dzdxVz
			if (nr_neighbors[0] == 2) {
				result[0] += 0.5 * (
					Vector(*neighbor_datas[5])[2]
					+ Vector(*neighbor_datas[1])[2]
					- Vector(*neighbor_datas[0])[2]
					- Vector(*neighbor_datas[4])[2]
				) / std::sqrt(
					neigh_pos_dist[0]*neigh_pos_dist[0]
					+ neigh_pos_dist[2]*neigh_pos_dist[2]
				);
			}
		}
		// result_y
		if (nr_neighbors[2] == 2) {
			// -dzdzVy
			result[1] -= 2 * (
				-vec[1] / neigh_neg_dist[2] / neigh_pos_dist[2]
				+ Vector(*neighbor_datas[5])[1] / neigh_pos_dist[2] / (neigh_pos_dist[2] + neigh_neg_dist[2])
				+ Vector(*neighbor_datas[4])[1] / neigh_neg_dist[2] / (neigh_pos_dist[2] + neigh_neg_dist[2])
			);
			// dzdyVz
			if (nr_neighbors[1] == 2) {
				result[1] += 0.5 * (
					Vector(*neighbor_datas[5])[2]
					+ Vector(*neighbor_datas[3])[2]
					- Vector(*neighbor_datas[2])[2]
					- Vector(*neighbor_datas[4])[2]
				) / std::sqrt(
					neigh_pos_dist[1]*neigh_pos_dist[1]
					+ neigh_pos_dist[2]*neigh_pos_dist[2]
				);
			}
		}
		if (nr_neighbors[0] == 2) {
			// -dxdxVy
			result[1] -= 2 * (
				-vec[1] / neigh_neg_dist[0] / neigh_pos_dist[0]
				+ Vector(*neighbor_datas[1])[1] / neigh_pos_dist[0] / (neigh_pos_dist[0] + neigh_neg_dist[0])
				+ Vector(*neighbor_datas[0])[1] / neigh_neg_dist[0] / (neigh_pos_dist[0] + neigh_neg_dist[0])
			);
			// dydxVx
			if (nr_neighbors[1] == 2) {
				result[1] += 0.5 * (
					Vector(*neighbor_datas[3])[0]
					+ Vector(*neighbor_datas[1])[0]
					- Vector(*neighbor_datas[0])[0]
					- Vector(*neighbor_datas[2])[0]
				) / std::sqrt(
					neigh_pos_dist[0]*neigh_pos_dist[0]
					+ neigh_pos_dist[1]*neigh_pos_dist[1]
				);
			}
		}
		// result_z
		if (nr_neighbors[0] == 2) {
			// -dxdxVz
			result[2] -= 2 * (
				-vec[2] / neigh_neg_dist[0] / neigh_pos_dist[0]
				+ Vector(*neighbor_datas[1])[2] / neigh_pos_dist[0] / (neigh_pos_dist[0] + neigh_neg_dist[0])
				+ Vector(*neighbor_datas[0])[2] / neigh_neg_dist[0] / (neigh_pos_dist[0] + neigh_neg_dist[0])
			);
			// dxdzVx
			if (nr_neighbors[2] == 2) {
				result[2] += 0.5 * (
					Vector(*neighbor_datas[5])[0]
					+ Vector(*neighbor_datas[1])[0]
					- Vector(*neighbor_datas[0])[0]
					- Vector(*neighbor_datas[4])[0]
				) / std::sqrt(
					neigh_pos_dist[0]*neigh_pos_dist[0]
					+ neigh_pos_dist[2]*neigh_pos_dist[2]
				);
			}
		}
		if (nr_neighbors[1] == 2) {
			// -dydyVz
			result[2] -= 2 * (
				-vec[2] / neigh_neg_dist[1] / neigh_pos_dist[1]
				+ Vector(*neighbor_datas[3])[2] / neigh_pos_dist[1] / (neigh_pos_dist[1] + neigh_neg_dist[1])
				+ Vector(*neighbor_datas[2])[2] / neigh_neg_dist[1] / (neigh_pos_dist[1] + neigh_neg_dist[1])
			);
			// dydzVy
			if (nr_neighbors[2] == 2) {
				result[2] += 0.5 * (
					Vector(*neighbor_datas[5])[1]
					+ Vector(*neighbor_datas[3])[1]
					- Vector(*neighbor_datas[2])[1]
					- Vector(*neighbor_datas[4])[1]
				) / std::sqrt(
					neigh_pos_dist[1]*neigh_pos_dist[1]
					+ neigh_pos_dist[2]*neigh_pos_dist[2]
				);
			}
		}
	}
}


}} // namespaces

#endif // ifndef PAMHD_DIVERGENCE_REMOVE_HPP
