/*
Tests vector field divergence removal of PAMHD in 1d.

Copyright 2014, 2015, 2016, 2017 Ilja Honkonen
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


#include "array"
#include "cstdlib"
#include "iostream"
#include "limits"
#include "vector"

#include "dccrg.hpp"
#include "dccrg_cartesian_geometry.hpp"
#include "gensimcell.hpp"

#include "divergence/remove.hpp"


int Poisson_Cell::transfer_switch = Poisson_Cell::INIT;


double function(const double x)
{
	return 10 + std::sin(x);
}

double div_removed_function()
{
	return 10;
}


struct Vector {
	using data_type = std::array<double, 3>;
};

struct Divergence {
	using data_type = double;
};

struct Gradient {
	using data_type = std::array<double, 3>;
};

struct Type {
	using data_type = int;
};

using Cell = gensimcell::Cell<
	gensimcell::Always_Transfer,
	Vector,
	Divergence,
	Gradient,
	Type
>;


/*!
Use p == 1 to get maximum norm.
*/
template<class Grid_T> double get_diff_lp_norm(
	const Grid_T& grid,
	const double p,
	const double cell_volume,
	const size_t dimension
) {
	double local_norm = 0, global_norm = 0;
	for (const auto& cell: grid.local_cells()) {
		if ((*cell.data)[Type()] != 1) {
			continue;
		}

		local_norm += std::pow(
			std::fabs((*cell.data)[Vector()][dimension] - div_removed_function()),
			p
		);
	}
	local_norm *= cell_volume;

	if (p == 1) {
		MPI_Comm comm = grid.get_communicator();
		MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_MAX, comm);
		MPI_Comm_free(&comm);
		return global_norm;
	} else {
		MPI_Comm comm = grid.get_communicator();
		MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Comm_free(&comm);
		return std::pow(global_norm, 1.0 / p);
	}
}


int main(int argc, char* argv[])
{
	if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
		std::cerr << "Couldn't initialize MPI." << std::endl;
		abort();
	}

	MPI_Comm comm = MPI_COMM_WORLD;

	int rank = 0, comm_size = 0;
	if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) {
		std::cerr << "Couldn't obtain MPI rank." << std::endl;
		abort();
	}
	if (MPI_Comm_size(comm, &comm_size) != MPI_SUCCESS) {
		std::cerr << "Couldn't obtain size of MPI communicator." << std::endl;
		abort();
	}


	// intialize Zoltan
	float zoltan_version;
	if (Zoltan_Initialize(argc, argv, &zoltan_version) != ZOLTAN_OK) {
		std::cerr << "Zoltan_Initialize failed." << std::endl;
		abort();
	}

	dccrg::Dccrg<Cell, dccrg::Cartesian_Geometry> grid_x;

	constexpr auto nr_of_cells = 64;
	const std::array<uint64_t, 3> grid_len{{nr_of_cells + 2, 3, 3}};

	grid_x
		.set_load_balancing_method("RANDOM")
		.set_initial_length(grid_len)
		.set_maximum_refinement_level(0)
		.set_neighborhood_length(1)
		.initialize(comm)
		.balance_load();

	const std::array<double, 3>
		cell_length_x{{4 * M_PI / (grid_len[0] - 2), 1, 1}},
		grid_start_x{{-cell_length_x[0], -cell_length_x[1], -cell_length_x[2]}};

	dccrg::Cartesian_Geometry::Parameters geom_params_x;
	geom_params_x.start = grid_start_x;
	geom_params_x.level_0_cell_length = cell_length_x;

	grid_x.set_geometry(geom_params_x);

	std::ofstream before("before"), after("after");
	for (const auto& cell: grid_x.local_cells()) {
		const auto x = grid_x.geometry.get_center(cell.id)[0];
		auto& vec_x = (*cell.data)[Vector()];
		vec_x[0] = function(x);
		vec_x[1] = vec_x[2] = 0;

		if (
			cell.id >= grid_len[0]*(grid_len[1]*grid_len[2]/2) + 2
			and cell.id < grid_len[0]*(grid_len[1]*grid_len[2]/2) + grid_len[0]
		) {
			(*cell.data)[Type()] = 1;
			before << x << " " << vec_x[0] << "\n";
		} else {
			(*cell.data)[Type()] = 0;
		}
	}
	grid_x.update_copies_of_remote_neighbors();

	auto Vector_Getter = [](Cell& cell_data) -> Vector::data_type& {
		return cell_data[Vector()];
	};
	auto Divergence_Getter = [](Cell& cell_data) -> Divergence::data_type& {
		return cell_data[Divergence()];
	};
	auto Gradient_Getter = [](Cell& cell_data) -> Gradient::data_type& {
		return cell_data[Gradient()];
	};
	auto Type_Getter = [](Cell& cell_data) -> Type::data_type& {
		return cell_data[Type()];
	};
	const auto div_before = pamhd::divergence::get_divergence(
		grid_x.local_cells(),
		grid_x,
		Vector_Getter,
		Divergence_Getter,
		Type_Getter
	);
	std::cout << "\nDiv before: " << div_before << std::endl;
	pamhd::divergence::remove(
		grid_x.local_cells(),
		grid_x,
		Vector_Getter,
		Divergence_Getter,
		Gradient_Getter,
		Type_Getter,
		100, 0, 1e-15, 2, 100, 0, false, true
	);
	const auto div_after = pamhd::divergence::get_divergence(
		grid_x.local_cells(),
		grid_x,
		Vector_Getter,
		Divergence_Getter,
		Type_Getter
	);
	std::cout << "Div after: " << div_after << std::endl;

	for (const auto& cell: grid_x.local_cells()) {
		const auto x = grid_x.geometry.get_center(cell.id)[0];
		auto& vec_x = (*cell.data)[Vector()];
		if (
			cell.id >= grid_len[0]*(grid_len[1]*grid_len[2]/2) + 2
			and cell.id < grid_len[0]*(grid_len[1]*grid_len[2]/2) + grid_len[0]
		) {
			(*cell.data)[Type()] = 1;
			after << x << " " << vec_x[0] << " " << cell.id << "\n";
		} else {
			(*cell.data)[Type()] = 0;
		}
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
