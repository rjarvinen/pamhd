/*
Propagates test particles (0 mass) in prescribed electric and magnetic fields.

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


Copy boundaries have no effect in this program and shouldn't be used in config file.
*/

#include "array"
#include "cmath"
#include "cstdlib"
#include "fstream"
#include "functional"
#include "iostream"
#include "random"
#include "string"

#include "boost/filesystem.hpp"
#include "boost/numeric/odeint.hpp"
#include "dccrg.hpp"
#include "dccrg_cartesian_geometry.hpp"
#include "Eigen/Core" // must be included before gensimcell
#include "Eigen/Geometry"
#include "mpi.h" // must be included before gensimcell
#include "gensimcell.hpp"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "boundaries/geometries.hpp"
#include "boundaries/multivariable_boundaries.hpp"
#include "boundaries/multivariable_initial_conditions.hpp"
#include "grid_options.hpp"
#include "mhd/background_magnetic_field.hpp"
#include "mhd/options.hpp"
#include "mhd/variables.hpp"
#include "particle/boundaries.hpp"
#include "particle/initialize.hpp"
#include "particle/options.hpp"
#include "particle/save.hpp"
#include "particle/solve_dccrg.hpp"
#include "particle/variables.hpp"


using namespace std;
namespace odeint = boost::numeric::odeint;

using Cell = pamhd::particle::Cell_test_particle;
using Grid = dccrg::Dccrg<Cell, dccrg::Cartesian_Geometry>;


// unique particle id to assign to next particle
unsigned long long int next_particle_id;


// returns reference to magnetic field for propagating particles
const auto Mag
	= [](Cell& cell_data)->typename pamhd::particle::Magnetic_Field::data_type&{
		return cell_data[pamhd::particle::Magnetic_Field()];
	};
// electric field for propagating particles
const auto Ele
	= [](Cell& cell_data)->typename pamhd::particle::Electric_Field::data_type&{
		return cell_data[pamhd::particle::Electric_Field()];
	};
// list of particles in cell not moving to another cell
const auto Part_Int
	= [](Cell& cell_data)->typename pamhd::particle::Particles_Internal::data_type&{
		return cell_data[pamhd::particle::Particles_Internal()];
	};
// particles moving to another cell
const auto Part_Ext
	= [](Cell& cell_data)->typename pamhd::particle::Particles_External::data_type&{
		return cell_data[pamhd::particle::Particles_External()];
	};
// number of particles in above list, for allocating memory for arriving particles
const auto Nr_Ext
	= [](Cell& cell_data)->typename pamhd::particle::Nr_Particles_External::data_type&{
		return cell_data[pamhd::particle::Nr_Particles_External()];
	};

// solver info variable for boundary logic
const auto Sol_Info
	= [](Cell& cell_data)->typename pamhd::particle::Solver_Info::data_type&{
		return cell_data[pamhd::particle::Solver_Info()];
	};

// references to background magnetic fields
pamhd::mhd::Bg_Magnetic_Field_Pos_X::data_type no_bg_field_in_cells_of_this_program{};
const auto Bg_B_Pos_X
	= [](Cell& /*cell_data*/)->typename pamhd::mhd::Bg_Magnetic_Field_Pos_X::data_type&{
		return no_bg_field_in_cells_of_this_program;
	};
const auto Bg_B_Pos_Y
	= [](Cell& /*cell_data*/)->typename pamhd::mhd::Bg_Magnetic_Field_Pos_Y::data_type&{
		return no_bg_field_in_cells_of_this_program;
	};
const auto Bg_B_Pos_Z
	= [](Cell& /*cell_data*/)->typename pamhd::mhd::Bg_Magnetic_Field_Pos_Z::data_type&{
		return no_bg_field_in_cells_of_this_program;
	};

// references to initial condition & boundary data of cell
const auto Bdy_N
	= [](Cell& cell_data)->typename pamhd::particle::Bdy_Number_Density::data_type&{
		return cell_data[pamhd::particle::Bdy_Number_Density()];
	};
const auto Bdy_V
	= [](Cell& cell_data)->typename pamhd::particle::Bdy_Velocity::data_type&{
		return cell_data[pamhd::particle::Bdy_Velocity()];
	};
const auto Bdy_T
	= [](Cell& cell_data)->typename pamhd::particle::Bdy_Temperature::data_type&{
		return cell_data[pamhd::particle::Bdy_Temperature()];
	};
const auto Bdy_Nr_Par
	= [](Cell& cell_data)->typename pamhd::particle::Bdy_Nr_Particles_In_Cell::data_type&{
		return cell_data[pamhd::particle::Bdy_Nr_Particles_In_Cell()];
	};
const auto Bdy_SpM
	= [](Cell& cell_data)->typename pamhd::particle::Bdy_Species_Mass::data_type&{
		return cell_data[pamhd::particle::Bdy_Species_Mass()];
	};
const auto Bdy_C2M
	= [](Cell& cell_data)->typename pamhd::particle::Bdy_Charge_Mass_Ratio::data_type&{
		return cell_data[pamhd::particle::Bdy_Charge_Mass_Ratio()];
	};

// given a particle these return references to particle's parameters
const auto Part_Pos
	= [](pamhd::particle::Particle_Internal& particle)->typename pamhd::particle::Position::data_type&{
		return particle[pamhd::particle::Position()];
	};
const auto Part_Vel
	= [](pamhd::particle::Particle_Internal& particle)->typename pamhd::particle::Velocity::data_type&{
		return particle[pamhd::particle::Velocity()];
	};
const auto Part_C2M
	= [](pamhd::particle::Particle_Internal& particle)->typename pamhd::particle::Charge_Mass_Ratio::data_type&{
		return particle[pamhd::particle::Charge_Mass_Ratio()];
	};
const auto Part_Mas
	= [](pamhd::particle::Particle_Internal& particle)->typename pamhd::particle::Mass::data_type&{
		return particle[pamhd::particle::Mass()];
	};
const auto Part_Des
	= [](pamhd::particle::Particle_External& particle)->typename pamhd::particle::Destination_Cell::data_type&{
		return particle[pamhd::particle::Destination_Cell()];
	};


int main(int argc, char* argv[])
{
	constexpr double Re = 6.371e6; // radius of earth

	/*
	Initialize MPI
	*/

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

	next_particle_id = 1 + rank;

	// intialize Zoltan
	float zoltan_version;
	if (Zoltan_Initialize(argc, argv, &zoltan_version) != ZOLTAN_OK) {
		std::cerr << "Zoltan_Initialize failed." << std::endl;
		abort();
	}


	// read and parse json data from configuration file
	if (argc != 2) {
		if (argc < 2 and rank == 0) {
			std::cerr
				<< "Name of configuration file required."
				<< std::endl;
		}
		if (argc > 2 and rank == 0) {
			std::cerr
				<< "Too many arguments given to " << argv[0]
				<< ": " << argc - 1 << ", should be 1"
				<< std::endl;
		}
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	std::ifstream json_file(argv[1]);
	if (not json_file.good()) {
		if (rank == 0) {
			std::cerr << "Couldn't open configuration file " << argv[1] << std::endl;
		}
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	std::string json{
		std::istreambuf_iterator<char>(json_file),
		std::istreambuf_iterator<char>()
	};

	rapidjson::Document document;
	document.Parse(json.c_str());
	if (document.HasParseError()) {
		std::cerr << "Couldn't parse json data in file " << argv[1]
			<< " at character position " << document.GetErrorOffset()
			<< ": " << rapidjson::GetParseError_En(document.GetParseError())
			<< std::endl;
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// particle options
	pamhd::particle::Options options_particle{document};

	if (rank == 0 and options_particle.output_directory != "") {
		try {
			boost::filesystem::create_directories(options_particle.output_directory);
		} catch (const boost::filesystem::filesystem_error& e) {
			std::cerr <<  __FILE__ << "(" << __LINE__ << ") "
				"Couldn't create output directory "
				<< options_particle.output_directory << ": "
				<< e.what()
				<< std::endl;
			abort();
		}
	}

	int particle_stepper = -1;
	if (options_particle.solver == "euler") {
		particle_stepper = 0;
	} else if (options_particle.solver == "midpoint") {
		particle_stepper = 1;
	} else if (options_particle.solver == "rk4") {
		particle_stepper = 2;
	} else if (options_particle.solver == "rkck54") {
		particle_stepper = 3;
	} else if (options_particle.solver == "rkf78") {
		particle_stepper = 4;
	} else {
		std::cerr <<  __FILE__ << "(" << __LINE__ << "): "
			<< "Unsupported solver: " << options_particle.solver
			<< ", should be one of: euler, (modified) midpoint, rk4 (runge_kutta4), rkck54 (runge_kutta_cash_karp54), rkf78 (runge_kutta_fehlberg78), see http://www.boost.org/doc/libs/release/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html#boost_numeric_odeint.odeint_in_detail.steppers.stepper_overview"
			<< std::endl;
		abort();
	}


	using geometry_id_t = unsigned int;

	pamhd::boundaries::Geometries<
		geometry_id_t,
		std::array<double, 3>,
		double,
		uint64_t
	> geometries;
	geometries.set(document);

	pamhd::boundaries::Multivariable_Initial_Conditions<
		geometry_id_t,
		pamhd::particle::Bdy_Number_Density,
		pamhd::particle::Bdy_Temperature,
		pamhd::particle::Bdy_Velocity,
		pamhd::particle::Bdy_Nr_Particles_In_Cell,
		pamhd::particle::Charge_Mass_Ratio,
		pamhd::particle::Species_Mass,
		pamhd::particle::Electric_Field,
		pamhd::particle::Magnetic_Field
	> initial_conditions;
	initial_conditions.set(document);

	pamhd::boundaries::Multivariable_Boundaries<
		uint64_t,
		geometry_id_t,
		pamhd::particle::Bdy_Number_Density,
		pamhd::particle::Bdy_Temperature,
		pamhd::particle::Bdy_Velocity,
		pamhd::particle::Bdy_Nr_Particles_In_Cell,
		pamhd::particle::Bdy_Charge_Mass_Ratio,
		pamhd::particle::Bdy_Species_Mass,
		pamhd::particle::Electric_Field,
		pamhd::particle::Magnetic_Field
	> boundaries;
	boundaries.set(document);

	pamhd::mhd::Background_Magnetic_Field<
		pamhd::mhd::Magnetic_Field::data_type
	> background_B;
	background_B.set(document);


	/*
	Initialize simulation grid
	*/
	Grid grid;

	pamhd::grid::Options grid_options;
	grid_options.set(document);

	const unsigned int neighborhood_size = 1;
	const auto& number_of_cells = grid_options.get_number_of_cells();
	const auto& periodic = grid_options.get_periodic();
	if (not grid.initialize(
		number_of_cells,
		comm,
		options_particle.lb_name.c_str(),
		neighborhood_size,
		0,
		periodic[0],
		periodic[1],
		periodic[2]
	)) {
		std::cerr << __FILE__ << ":" << __LINE__
			<< ": Couldn't initialize grid."
			<< std::endl;
		abort();
	}

	// set grid geometry
	const std::array<double, 3>
		simulation_volume
			= grid_options.get_volume(),
		cell_volume{
			simulation_volume[0] / number_of_cells[0],
			simulation_volume[1] / number_of_cells[1],
			simulation_volume[2] / number_of_cells[2]
		};

	dccrg::Cartesian_Geometry::Parameters geom_params;
	geom_params.start = grid_options.get_start();
	geom_params.level_0_cell_length = cell_volume;

	if (not grid.set_geometry(geom_params)) {
		std::cerr << __FILE__ << ":" << __LINE__
			<< ": Couldn't set grid geometry."
			<< std::endl;
		abort();
	}

	grid.balance_load();

	// update owner process of cells for saving into file
	for (auto& cell: grid.cells) {
		(*cell.data)[pamhd::mhd::MPI_Rank()] = rank;
	}

	// assign cells into boundary geometries
	for (const auto& cell: grid.cells) {
		const auto
			start = grid.geometry.get_min(cell.id),
			end = grid.geometry.get_max(cell.id);
		geometries.overlaps(start, end, cell.id);
	}

	// pointer to data of every local cell and its neighbor(s)
	const auto& cell_data_pointers = grid.get_cell_data_pointers();

	// index of first outer cell in dccrg's cell data pointer cache
	size_t outer_cell_start_i = 0;
	for (const auto& item: cell_data_pointers) {
		outer_cell_start_i++;
		if (get<0>(item) == dccrg::error_cell) {
			break;
		}
	}


	/*
	Simulate
	*/

	const double time_end = options_particle.time_start + options_particle.time_length;
	double
		max_dt = 0,
		simulation_time = options_particle.time_start,
		next_particle_save = options_particle.save_n;

	std::vector<uint64_t>
		cells = grid.get_cells(),
		inner_cells = grid.get_local_cells_not_on_process_boundary(),
		outer_cells = grid.get_local_cells_on_process_boundary(),
		remote_cells = grid.get_remote_cells_on_process_boundary();

	// set initial condition
	std::mt19937_64 random_source;

	auto nr_particles_created
		= pamhd::particle::initialize<
			pamhd::particle::Particle_Internal,
			pamhd::particle::Mass,
			pamhd::particle::Charge_Mass_Ratio,
			pamhd::particle::Position,
			pamhd::particle::Velocity,
			pamhd::particle::Particle_ID,
			pamhd::particle::Species_Mass
		>(
			geometries,
			initial_conditions,
			simulation_time,
			cells,
			background_B,
			grid,
			random_source,
			options_particle.boltzmann,
			options_particle.vacuum_permeability,
			next_particle_id,
			grid.get_comm_size(),
			true,
			true,
			Ele,
			Mag,
			Part_Int,
			Bg_B_Pos_X,
			Bg_B_Pos_Y,
			Bg_B_Pos_Z,
			Bdy_N,
			Bdy_V,
			Bdy_T,
			Bdy_Nr_Par,
			Bdy_SpM,
			Bdy_C2M,
			Sol_Info
		);
	next_particle_id += nr_particles_created * grid.get_comm_size();

	nr_particles_created
		+= pamhd::particle::apply_massless_boundaries<
			pamhd::particle::Particle_Internal,
			pamhd::particle::Mass,
			pamhd::particle::Charge_Mass_Ratio,
			pamhd::particle::Position,
			pamhd::particle::Velocity,
			pamhd::particle::Particle_ID,
			pamhd::particle::Species_Mass
		>(
			geometries,
			boundaries,
			simulation_time,
			0,
			cells,
			grid,
			random_source,
			options_particle.boltzmann,
			options_particle.vacuum_permeability,
			next_particle_id,
			grid.get_comm_size(),
			true,
			Sol_Info,
			Ele,
			Mag,
			Part_Int,
			Bdy_N,
			Bdy_V,
			Bdy_T,
			Bdy_Nr_Par,
			Bdy_SpM,
			Bdy_C2M
		);
	next_particle_id += nr_particles_created * grid.get_comm_size();

	if (options_particle.verbose and rank == 0) {
		cout << "Done initializing particles" << endl;
	}

	/*
	Classify cells into normal, boundary and dont_solve
	*/

	pamhd::particle::set_solver_info<pamhd::particle::Solver_Info>(
		grid, boundaries, geometries, Sol_Info
	);
	// make lists from above for divergence removal functions
	std::vector<uint64_t> solve_cells, bdy_cells, skip_cells;
	for (const auto& cell: grid.cells) {
		if ((Sol_Info(*cell.data) & pamhd::particle::Solver_Info::dont_solve) > 0) {
			skip_cells.push_back(cell.id);
		} else if (Sol_Info(*cell.data) > 0) {
			bdy_cells.push_back(cell.id);
		} else {
			solve_cells.push_back(cell.id);
		}
	}


	size_t simulated_steps = 0;
	while (simulation_time < time_end) {
		simulated_steps++;

		double
			// don't step over the final simulation time
			until_end = time_end - simulation_time,
			local_time_step = std::min(std::min(options_particle.time_step_factor * max_dt, until_end), max_dt),
			time_step = -1;

		if (
			MPI_Allreduce(
				&local_time_step,
				&time_step,
				1,
				MPI_DOUBLE,
				MPI_MIN,
				comm
			) != MPI_SUCCESS
		) {
			std::cerr << __FILE__ << ":" << __LINE__
				<< ": Couldn't reduce time step."
				<< std::endl;
			abort();
		}

		/*
		Solve
		*/

		if (options_particle.verbose && grid.get_rank() == 0) {
			cout << "Solving particles at time " << simulation_time
				<< " s with time step " << time_step << " s" << endl;
		}

		max_dt = std::numeric_limits<double>::max();


		Cell::set_transfer_all(
			true,
			pamhd::particle::Electric_Field(),
			pamhd::particle::Magnetic_Field()
		);
		grid.update_copies_of_remote_neighbors();
		Cell::set_transfer_all(
			false,
			pamhd::particle::Electric_Field(),
			pamhd::particle::Magnetic_Field()
		);

		// E is given directly to particle propagator
		// TODO: don't use preprocessor
		#define SOLVE_WITH_STEPPER(given_type, given_cells) \
			pamhd::particle::solve<\
				given_type\
			>(\
				time_step,\
				given_cells,\
				grid,\
				background_B,\
				options_particle.vacuum_permeability,\
				false,\
				Ele,\
				Mag,\
				Nr_Ext,\
				Part_Int,\
				Part_Ext,\
				Part_Pos,\
				Part_Vel,\
				Part_C2M,\
				Part_Mas,\
				Part_Des\
			)

		switch (particle_stepper) {
		case 0:
			max_dt = std::min(
				max_dt,
				SOLVE_WITH_STEPPER(odeint::euler<pamhd::particle::state_t>, outer_cells)
			);
			break;
		case 1:
			max_dt = std::min(
				max_dt,
				SOLVE_WITH_STEPPER(odeint::modified_midpoint<pamhd::particle::state_t>, outer_cells)
			);
			break;
		case 2:
			max_dt = std::min(
				max_dt,
				SOLVE_WITH_STEPPER(odeint::runge_kutta4<pamhd::particle::state_t>, outer_cells)
			);
			break;
		case 3:
			max_dt = std::min(
				max_dt,
				SOLVE_WITH_STEPPER(odeint::runge_kutta_cash_karp54<pamhd::particle::state_t>, outer_cells)
			);
			break;
		case 4:
			max_dt = std::min(
				max_dt,
				SOLVE_WITH_STEPPER(odeint::runge_kutta_fehlberg78<pamhd::particle::state_t>, outer_cells)
			);
			break;
		default:
			std::cerr <<  __FILE__ << "(" << __LINE__ << "): " << particle_stepper << std::endl;
			abort();
		}

		Cell::set_transfer_all(true, pamhd::particle::Nr_Particles_External());
		grid.start_remote_neighbor_copy_updates();

		switch (particle_stepper) {
		case 0:
			max_dt = std::min(
				max_dt,
				SOLVE_WITH_STEPPER(odeint::euler<pamhd::particle::state_t>, inner_cells)
			);
			break;
		case 1:
			max_dt = std::min(
				max_dt,
				SOLVE_WITH_STEPPER(odeint::modified_midpoint<pamhd::particle::state_t>, inner_cells)
			);
			break;
		case 2:
			max_dt = std::min(
				max_dt,
				SOLVE_WITH_STEPPER(odeint::runge_kutta4<pamhd::particle::state_t>, inner_cells)
			);
			break;
		case 3:
			max_dt = std::min(
				max_dt,
				SOLVE_WITH_STEPPER(odeint::runge_kutta_cash_karp54<pamhd::particle::state_t>, inner_cells)
			);
			break;
		case 4:
			max_dt = std::min(
				max_dt,
				SOLVE_WITH_STEPPER(odeint::runge_kutta_fehlberg78<pamhd::particle::state_t>, inner_cells)
			);
			break;
		default:
			std::cerr <<  __FILE__ << "(" << __LINE__ << "): " << particle_stepper << std::endl;
			abort();
		}
		#undef SOLVE_WITH_STEPPER

		simulation_time += time_step;

		grid.wait_remote_neighbor_copy_update_receives();
		pamhd::particle::resize_receiving_containers<
			pamhd::particle::Nr_Particles_External,
			pamhd::particle::Particles_External
		>(remote_cells, grid);

		grid.wait_remote_neighbor_copy_update_sends();

		Cell::set_transfer_all(false, pamhd::particle::Nr_Particles_External());
		Cell::set_transfer_all(true, pamhd::particle::Particles_External());

		grid.start_remote_neighbor_copy_updates();

		pamhd::particle::incorporate_external_particles<
			pamhd::particle::Nr_Particles_External,
			pamhd::particle::Particles_Internal,
			pamhd::particle::Particles_External,
			pamhd::particle::Destination_Cell
		>(inner_cells, grid);

		grid.wait_remote_neighbor_copy_update_receives();

		pamhd::particle::incorporate_external_particles<
			pamhd::particle::Nr_Particles_External,
			pamhd::particle::Particles_Internal,
			pamhd::particle::Particles_External,
			pamhd::particle::Destination_Cell
		>(outer_cells, grid);

		pamhd::particle::remove_external_particles<
			pamhd::particle::Nr_Particles_External,
			pamhd::particle::Particles_External
		>(inner_cells, grid);

		grid.wait_remote_neighbor_copy_update_sends();
		Cell::set_transfer_all(false, pamhd::particle::Particles_External());

		pamhd::particle::remove_external_particles<
			pamhd::particle::Nr_Particles_External,
			pamhd::particle::Particles_External
		>(outer_cells, grid);


		nr_particles_created
			+= pamhd::particle::apply_massless_boundaries<
				pamhd::particle::Particle_Internal,
				pamhd::particle::Mass,
				pamhd::particle::Charge_Mass_Ratio,
				pamhd::particle::Position,
				pamhd::particle::Velocity,
				pamhd::particle::Particle_ID,
				pamhd::particle::Species_Mass
			>(
				geometries,
				boundaries,
				simulation_time,
				simulated_steps,
				cells,
				grid,
				random_source,
				options_particle.boltzmann,
				options_particle.vacuum_permeability,
				next_particle_id,
				grid.get_comm_size(),
				true,
				Sol_Info,
				Ele,
				Mag,
				Part_Int,
				Bdy_N,
				Bdy_V,
				Bdy_T,
				Bdy_Nr_Par,
				Bdy_SpM,
				Bdy_C2M
			);
		next_particle_id += nr_particles_created * grid.get_comm_size();


		/*
		Save simulation to disk
		*/

		if (
			(options_particle.save_n >= 0 and (simulation_time == 0 or simulation_time >= time_end))
			or (options_particle.save_n > 0 and simulation_time >= next_particle_save)
		) {
			if (next_particle_save <= simulation_time) {
				next_particle_save += options_particle.save_n;
			}

			if (options_particle.verbose && rank == 0) {
				cout << "Saving particles at time " << simulation_time << endl;
			}

			// TODO: add version info
			if (
				not pamhd::particle::save<
					pamhd::particle::Electric_Field,
					pamhd::particle::Magnetic_Field,
					pamhd::mhd::Electric_Current_Density,
					pamhd::particle::Nr_Particles_Internal,
					pamhd::particle::Particles_Internal
				>(
					boost::filesystem::canonical(
						boost::filesystem::path(options_particle.output_directory)
					).append("particle_").generic_string(),
					grid,
					simulation_time,
					0,
					0,
					options_particle.boltzmann
				)
			) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << "): Couldn't save particle result." << std::endl;
				MPI_Finalize();
				return EXIT_FAILURE;
			}
		}
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
