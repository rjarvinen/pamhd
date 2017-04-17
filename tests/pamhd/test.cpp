/*
Particle-assisted magnetohydrodynamics.

Copyright 2015, 2016, 2017 Ilja Honkonen
All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.


Combination of ../particle/test.cpp and ../mhd/two_test.cpp where
particles represent one of the fluids.
*/

#include "array"
#include "cmath"
#include "cstdlib"
#include "fstream"
#include "iostream"
#include "random"
#include "string"

#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
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
#include "divergence/remove.hpp"
#include "grid_options.hpp"
#include "mhd/background_magnetic_field.hpp"
#include "mhd/common.hpp"
#include "mhd/options.hpp"
#include "mhd/save.hpp"
//#include "mhd/N_boundaries.hpp"
#include "mhd/N_solve.hpp"
#include "mhd/N_hll_athena.hpp"
//#include "mhd/N_initialize.hpp"
#include "mhd/initialize.hpp"
#include "mhd/N_rusanov.hpp"
#include "mhd/variables.hpp"
#include "particle/accumulate_dccrg.hpp"
#include "particle/boundaries.hpp"
#include "particle/common.hpp"
#include "particle/initialize.hpp"
#include "particle/options.hpp"
#include "particle/save.hpp"
#include "particle/solve_dccrg.hpp"
#include "particle/variables.hpp"
//#include "pamhd/initialize.hpp"


using namespace std;

/*
See comments in ../mhd/two_test.cpp and ../particle/test.cpp
for explanation of items identical to ones in those files
*/

unsigned long long int next_particle_id;

int Poisson_Cell::transfer_switch = Poisson_Cell::INIT;


// data stored in every cell of simulation grid
using Cell = gensimcell::Cell<
	gensimcell::Optional_Transfer,
	pamhd::mhd::HD1_State, // fluid
	pamhd::mhd::HD2_State, // particles
	pamhd::mhd::Magnetic_Field,
	pamhd::mhd::Electric_Current_Density,
	pamhd::particle::Solver_Info,
	pamhd::mhd::MPI_Rank,
	pamhd::mhd::Resistivity,
	pamhd::mhd::Bg_Magnetic_Field_Pos_X,
	pamhd::mhd::Bg_Magnetic_Field_Pos_Y,
	pamhd::mhd::Bg_Magnetic_Field_Pos_Z,
	pamhd::mhd::Magnetic_Field_Resistive,
	pamhd::mhd::Magnetic_Field_Temp,
	pamhd::mhd::Magnetic_Field_Divergence,
	pamhd::mhd::Scalar_Potential_Gradient,
	pamhd::mhd::HD1_Flux,
	pamhd::mhd::HD2_Flux,
	pamhd::mhd::Magnetic_Field_Flux,
	pamhd::particle::Electric_Field,
	pamhd::particle::Number_Of_Particles,
	pamhd::particle::Bdy_Number_Density,
	pamhd::particle::Bdy_Velocity,
	pamhd::particle::Bdy_Temperature,
	pamhd::particle::Bdy_Species_Mass,
	pamhd::particle::Bdy_Charge_Mass_Ratio,
	pamhd::particle::Bdy_Nr_Particles_In_Cell,
	pamhd::particle::Bulk_Mass,
	pamhd::particle::Bulk_Momentum,
	pamhd::particle::Bulk_Velocity,
	pamhd::particle::Current_Minus_Velocity,
	pamhd::particle::Bulk_Relative_Velocity2,
	pamhd::particle::Nr_Particles_Internal,
	pamhd::particle::Nr_Particles_External,
	pamhd::particle::Nr_Accumulated_To_Cells,
	pamhd::particle::Particles_Internal,
	pamhd::particle::Particles_External,
	pamhd::particle::Accumulated_To_Cells
>;
// simulation data, see doi:10.1016/j.cpc.2012.12.017 or arxiv.org/abs/1212.3496
using Grid = dccrg::Dccrg<Cell, dccrg::Cartesian_Geometry>;


// returns a reference to cell's list of particles not moving to another cell
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
const auto Sol_Info_P
	= [](Cell& cell_data)->typename pamhd::particle::Solver_Info::data_type&{
		return cell_data[pamhd::particle::Solver_Info()];
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
// as above but for caller that also provides cell's data
const auto Part_Mas_Cell
	= [](Cell&, pamhd::particle::Particle_Internal& particle)->typename pamhd::particle::Mass::data_type&{
		return particle[pamhd::particle::Mass()];
	};
const auto Part_Des
	= [](pamhd::particle::Particle_External& particle)->typename pamhd::particle::Destination_Cell::data_type&{
		return particle[pamhd::particle::Destination_Cell()];
	};
// reference to mass of given particle's species
const auto Part_SpM
	= [](pamhd::particle::Particle_Internal& particle)->typename pamhd::particle::Species_Mass::data_type&{
		return particle[pamhd::particle::Species_Mass()];
	};
// as above but for caller that also provides cell's data
const auto Part_SpM_Cell
	= [](Cell&, pamhd::particle::Particle_Internal& particle)->typename pamhd::particle::Species_Mass::data_type&{
		return particle[pamhd::particle::Species_Mass()];
	};
// returns a copy of given particle's momentun
const auto Part_Mom
	= [](Cell&, pamhd::particle::Particle_Internal& particle)->typename pamhd::particle::Velocity::data_type{
		return particle[pamhd::particle::Mass()] * particle[pamhd::particle::Velocity()];
	};
// copy of particle's velocity squared relative to pamhd::particle::Bulk_Velocity
const auto Part_RV2
	= [](Cell& cell_data, pamhd::particle::Particle_Internal& particle)->typename pamhd::particle::Mass::data_type{
		return
			particle[pamhd::particle::Species_Mass()]
			* particle[pamhd::particle::Mass()]
			* (
				particle[pamhd::particle::Velocity()]
				- cell_data[pamhd::particle::Bulk_Velocity()]
			).squaredNorm();
	};

// reference to accumulated number of particles in given cell
const auto Nr_Particles
	= [](Cell& cell_data)->typename pamhd::particle::Number_Of_Particles::data_type&{
		return cell_data[pamhd::particle::Number_Of_Particles()];
	};

const auto Bulk_Mass_Getter
	= [](Cell& cell_data)->typename pamhd::particle::Bulk_Mass::data_type&{
		return cell_data[pamhd::particle::Bulk_Mass()];
	};

const auto Bulk_Momentum_Getter
	= [](Cell& cell_data)->typename pamhd::particle::Bulk_Momentum::data_type&{
		return cell_data[pamhd::particle::Bulk_Momentum()];
	};

const auto Bulk_Relative_Velocity2_Getter
	= [](Cell& cell_data)->typename pamhd::particle::Bulk_Relative_Velocity2::data_type&{
		return cell_data[pamhd::particle::Bulk_Relative_Velocity2()];
	};

// accumulated momentum / accumulated velocity of particles in given cell
const auto Bulk_Velocity_Getter
	= [](Cell& cell_data)->typename pamhd::particle::Bulk_Velocity::data_type&{
		return cell_data[pamhd::particle::Bulk_Velocity()];
	};

// list of items (variables above) accumulated from particles in given cell
const auto Accu_List_Getter
	= [](Cell& cell_data)->typename pamhd::particle::Accumulated_To_Cells::data_type&{
		return cell_data[pamhd::particle::Accumulated_To_Cells()];
	};

// length of above list (for transferring between processes)
const auto Accu_List_Length_Getter
	= [](Cell& cell_data)->typename pamhd::particle::Nr_Accumulated_To_Cells::data_type&{
		return cell_data[pamhd::particle::Nr_Accumulated_To_Cells()];
	};

// target cell of accumulated particle values in an accumulation list item
const auto Accu_List_Target_Getter
	= [](pamhd::particle::Accumulated_To_Cell& accu_item)
		->typename pamhd::particle::Target::data_type&
	{
		return accu_item[pamhd::particle::Target()];
	};

// accumulated number of particles in an accumulation list item
const auto Accu_List_Number_Of_Particles_Getter
	= [](pamhd::particle::Accumulated_To_Cell& accu_item)
		->typename pamhd::particle::Number_Of_Particles::data_type&
	{
		return accu_item[pamhd::particle::Number_Of_Particles()];
	};

const auto Accu_List_Bulk_Mass_Getter
	= [](pamhd::particle::Accumulated_To_Cell& accu_item)
		->typename pamhd::particle::Bulk_Mass::data_type&
	{
		return accu_item[pamhd::particle::Bulk_Mass()];
	};

const auto Accu_List_Bulk_Momentum_Getter
	= [](pamhd::particle::Accumulated_To_Cell& accu_item)
		->typename pamhd::particle::Bulk_Momentum::data_type&
	{
		return accu_item[pamhd::particle::Bulk_Momentum()];
	};

const auto Accu_List_Bulk_Relative_Velocity2_Getter
	= [](pamhd::particle::Accumulated_To_Cell& accu_item)
		->typename pamhd::particle::Bulk_Relative_Velocity2::data_type&
	{
		return accu_item[pamhd::particle::Bulk_Relative_Velocity2()];
	};

// reference to total mass density of all fluids in given cell
/*const auto Mas
	= [](Cell& cell_data)->typename pamhd::mhd::Mass_Density::data_type&{
		return cell_data[pamhd::mhd::MHD_State_Conservative()][pamhd::mhd::Mass_Density()];
	};
const auto Mom
	= [](Cell& cell_data)->typename pamhd::mhd::Momentum_Density::data_type&{
		return cell_data[pamhd::mhd::MHD_State_Conservative()][pamhd::mhd::Momentum_Density()];
	};
const auto Nrj
	= [](Cell& cell_data)->typename pamhd::mhd::Total_Energy_Density::data_type&{
		return cell_data[pamhd::mhd::MHD_State_Conservative()][pamhd::mhd::Total_Energy_Density()];
	};*/
const auto Mag
	= [](Cell& cell_data)->typename pamhd::mhd::Magnetic_Field::data_type&{
		return cell_data[pamhd::mhd::Magnetic_Field()];
	};

// field before divergence removal in case removal fails
const auto Mag_tmp
	= [](Cell& cell_data)->typename pamhd::mhd::Magnetic_Field_Temp::data_type&{
		return cell_data[pamhd::mhd::Magnetic_Field_Temp()];
	};
// divergence of magnetic field
const auto Mag_div
	= [](Cell& cell_data)->typename pamhd::mhd::Magnetic_Field_Divergence::data_type&{
		return cell_data[pamhd::mhd::Magnetic_Field_Divergence()];
	};
// electrical resistivity
const auto Res
	= [](Cell& cell_data)->typename pamhd::mhd::Resistivity::data_type&{
		return cell_data[pamhd::mhd::Resistivity()];
	};
// adjustment to magnetic field due to resistivity
const auto Mag_res
	= [](Cell& cell_data)->typename pamhd::mhd::Magnetic_Field_Resistive::data_type&{
		return cell_data[pamhd::mhd::Magnetic_Field_Resistive()];
	};
// magnetic field for propagating particles
/*const auto Mag_part
	= [](Cell& cell_data)->typename pamhd::particle::Magnetic_Field::data_type&{
		return cell_data[pamhd::particle::Magnetic_Field()];
	};*/
// curl of magnetic field
const auto Cur
	= [](Cell& cell_data)->typename pamhd::mhd::Electric_Current_Density::data_type&{
		return cell_data[pamhd::mhd::Electric_Current_Density()];
	};
// electric current minus bulk velocity
const auto J_m_V
	= [](Cell& cell_data)->typename pamhd::particle::Current_Minus_Velocity::data_type&{
		return cell_data[pamhd::particle::Current_Minus_Velocity()];
	};
// electric field for propagating particles
const auto Ele
	= [](Cell& cell_data)->typename pamhd::particle::Electric_Field::data_type&{
		return cell_data[pamhd::particle::Electric_Field()];
	};

// doesn't affect result
/*const auto Mas_f
	= [](Cell& cell_data)->typename pamhd::mhd::Mass_Density::data_type&{
		return cell_data[pamhd::mhd::MHD_Flux_Conservative()][pamhd::mhd::Mass_Density()];
	};
// doesn't affect result
const auto Mom_f
	= [](Cell& cell_data)->typename pamhd::mhd::Momentum_Density::data_type&{
		return cell_data[pamhd::mhd::MHD_Flux_Conservative()][pamhd::mhd::Momentum_Density()];
	};
// doesn't affect result
const auto Nrj_f
	= [](Cell& cell_data)->typename pamhd::mhd::Total_Energy_Density::data_type&{
		return cell_data[pamhd::mhd::MHD_Flux_Conservative()][pamhd::mhd::Total_Energy_Density()];
	};
// flux / total change of magnetic field over one time step
const auto Mag_f
	= [](Cell& cell_data)->typename pamhd::mhd::Magnetic_Field::data_type&{
		return cell_data[pamhd::mhd::MHD_Flux_Conservative()][pamhd::mhd::Magnetic_Field()];
	};*/



// flux / total change of magnetic field over one time step
const auto Mag_f
	= [](Cell& cell_data)->typename pamhd::mhd::Magnetic_Field_Flux::data_type&{
		return cell_data[pamhd::mhd::Magnetic_Field_Flux()];
	};

// reference to mass density of fluid 1 in given cell
const auto Mas1
	= [](Cell& cell_data)->typename pamhd::mhd::Mass_Density::data_type&{
		return cell_data[pamhd::mhd::HD1_State()][pamhd::mhd::Mass_Density()];
	};
const auto Mom1
	= [](Cell& cell_data)->typename pamhd::mhd::Momentum_Density::data_type&{
		return cell_data[pamhd::mhd::HD1_State()][pamhd::mhd::Momentum_Density()];
	};
const auto Nrj1
	= [](Cell& cell_data)->typename pamhd::mhd::Total_Energy_Density::data_type&{
		return cell_data[pamhd::mhd::HD1_State()][pamhd::mhd::Total_Energy_Density()];
	};
// reference to mass density of fluid 2 in given cell
const auto Mas2
	= [](Cell& cell_data)->typename pamhd::mhd::Mass_Density::data_type&{
		return cell_data[pamhd::mhd::HD2_State()][pamhd::mhd::Mass_Density()];
	};
const auto Mom2
	= [](Cell& cell_data)->typename pamhd::mhd::Momentum_Density::data_type&{
		return cell_data[pamhd::mhd::HD2_State()][pamhd::mhd::Momentum_Density()];
	};
const auto Nrj2
	= [](Cell& cell_data)->typename pamhd::mhd::Total_Energy_Density::data_type&{
		return cell_data[pamhd::mhd::HD2_State()][pamhd::mhd::Total_Energy_Density()];
	};

// flux of mass density of fluid 1 over one time step
const auto Mas1_f
	= [](Cell& cell_data)->typename pamhd::mhd::Mass_Density::data_type&{
		return cell_data[pamhd::mhd::HD1_Flux()][pamhd::mhd::Mass_Density()];
	};
const auto Mom1_f
	= [](Cell& cell_data)->typename pamhd::mhd::Momentum_Density::data_type&{
		return cell_data[pamhd::mhd::HD1_Flux()][pamhd::mhd::Momentum_Density()];
	};
const auto Nrj1_f
	= [](Cell& cell_data)->typename pamhd::mhd::Total_Energy_Density::data_type&{
		return cell_data[pamhd::mhd::HD1_Flux()][pamhd::mhd::Total_Energy_Density()];
	};
// flux of mass density of fluid 2 over one time step
const auto Mas2_f
	= [](Cell& cell_data)->typename pamhd::mhd::Mass_Density::data_type&{
		return cell_data[pamhd::mhd::HD2_Flux()][pamhd::mhd::Mass_Density()];
	};
const auto Mom2_f
	= [](Cell& cell_data)->typename pamhd::mhd::Momentum_Density::data_type&{
		return cell_data[pamhd::mhd::HD2_Flux()][pamhd::mhd::Momentum_Density()];
	};
const auto Nrj2_f
	= [](Cell& cell_data)->typename pamhd::mhd::Total_Energy_Density::data_type&{
		return cell_data[pamhd::mhd::HD2_Flux()][pamhd::mhd::Total_Energy_Density()];
	};


int main(int argc, char* argv[])
{
	using std::min;

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

	// options
	pamhd::particle::Options options_particle{document};
	pamhd::mhd::Options options_mhd{document};

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
	/* TODO if (rank == 0 and options_mhd.output_directory != "") {
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
	}*/

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

	/*
	Initial and (TODO) boundary conditions for fluid 1 and magnetic field
	*/
	pamhd::boundaries::Multivariable_Initial_Conditions<
		geometry_id_t,
		pamhd::mhd::Number_Density,
		pamhd::mhd::Velocity,
		pamhd::mhd::Pressure,
		pamhd::mhd::Magnetic_Field
	> initial_conditions_fluid;
	initial_conditions_fluid.set(document);

	/*
	Initial and (TODO) boundary conditions for fluid 2 represented by particles
	*/
	std::vector<
		pamhd::boundaries::Multivariable_Initial_Conditions<
			geometry_id_t,
			pamhd::particle::Bdy_Number_Density,
			pamhd::particle::Bdy_Temperature,
			pamhd::particle::Bdy_Velocity,
			pamhd::particle::Bdy_Nr_Particles_In_Cell,
			pamhd::particle::Charge_Mass_Ratio,
			pamhd::particle::Species_Mass
		>
	> initial_conditions_particles;
	for (size_t population_id = 0; population_id < 99; population_id++) {
		const auto& obj_population_i
			= document.FindMember(
				(
					"particle-population-"
					+ boost::lexical_cast<std::string>(population_id)
				).c_str()
			);
		if (obj_population_i == document.MemberEnd()) {
			if (population_id == 0) {
				continue; // allow population ids to start from 0 and 1
			} else {
				break;
			}
		}
		const auto& obj_population = obj_population_i->value;

		const auto old_size = initial_conditions_particles.size();
		initial_conditions_particles.resize(old_size + 1);
		initial_conditions_particles[old_size].set(obj_population);
	}

	pamhd::mhd::Background_Magnetic_Field<
		pamhd::mhd::Magnetic_Field::data_type
	> background_B;
	background_B.set(document);


	const auto mhd_solver
		= [&options_mhd, &background_B, &rank](){
			if (options_mhd.solver == "rusanov") {

				return pamhd::mhd::get_flux_N_rusanov<
					pamhd::mhd::MHD_Conservative,
					pamhd::mhd::Magnetic_Field::data_type,
					pamhd::mhd::Mass_Density,
					pamhd::mhd::Momentum_Density,
					pamhd::mhd::Total_Energy_Density,
					pamhd::mhd::Magnetic_Field
				>;

			} else if (options_mhd.solver == "hll-athena") {

				return pamhd::mhd::athena::get_flux_N_hll<
					pamhd::mhd::MHD_Conservative,
					pamhd::mhd::Magnetic_Field::data_type,
					pamhd::mhd::Mass_Density,
					pamhd::mhd::Momentum_Density,
					pamhd::mhd::Total_Energy_Density,
					pamhd::mhd::Magnetic_Field
				>;

			} /* TODO else if (options_mhd.solver == "hlld-athena") {

				if (background_B.exists() and rank == 0) {
					std::cout << "NOTE: background magnetic field ignored by hlld-athena solver." << std::endl;
				}

				return pamhd::mhd::athena::get_flux_hlld<
					pamhd::mhd::MHD_Conservative,
					pamhd::mhd::Magnetic_Field::data_type,
					pamhd::mhd::Mass_Density,
					pamhd::mhd::Momentum_Density,
					pamhd::mhd::Total_Energy_Density,
					pamhd::mhd::Magnetic_Field
				>;

			} else if (options_mhd.solver == "roe-athena") {

				if (background_B.exists() and rank == 0) {
					std::cout << "NOTE: background magnetic field ignored by roe-athena solver." << std::endl;
				}

				return pamhd::mhd::athena::get_flux_roe<
					pamhd::mhd::MHD_Conservative,
					pamhd::mhd::Magnetic_Field::data_type,
					pamhd::mhd::Mass_Density,
					pamhd::mhd::Momentum_Density,
					pamhd::mhd::Total_Energy_Density,
					pamhd::mhd::Magnetic_Field
				>;
			} */else {
				std::cerr <<  __FILE__ << "(" << __LINE__ << "): "
					<< "Unsupported solver: " << options_mhd.solver
					<< std::endl;
				abort();
			}
		}();


	/*
	Prepare resistivity
	*/

	pamhd::boundaries::Math_Expression<pamhd::mhd::Resistivity> resistivity;
	mup::Value J_val;
	mup::Variable J_var(&J_val);
	resistivity.add_expression_variable("J", J_var);

	const auto resistivity_name = pamhd::mhd::Resistivity::get_option_name();
	if (not document.HasMember(resistivity_name.c_str())) {
		if (rank == 0) {
			std::cerr << __FILE__ "(" << __LINE__
				<< "): Configuration file doesn't have a "
				<< resistivity_name << " key."
				<< std::endl;
		};
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	const auto& json_resistivity = document[resistivity_name.c_str()];
	if (not json_resistivity.IsString()) {
		if (rank == 0) {
			std::cerr << __FILE__ "(" << __LINE__
				<< "): Resistivity option is not of type string."
				<< std::endl;
		};
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	resistivity.set_expression(json_resistivity.GetString());


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
		next_particle_save = options_particle.save_n,
		next_mhd_save = options_mhd.save_mhd_n,
		next_rem_div_B = options_mhd.remove_div_B_n;

	std::vector<uint64_t>
		cells = grid.get_cells(),
		inner_cells = grid.get_local_cells_not_on_process_boundary(),
		outer_cells = grid.get_local_cells_on_process_boundary(),
		remote_cells = grid.get_remote_cells_on_process_boundary();


	if (rank == 0) {
		cout << "Initializing simulation... " << endl;
	}

	// resistivity
	for (auto& cell: grid.cells) {
		Res(*cell.data) = 0;
	}

	// set B to 0 before initializing fluids to get correct total energy
	pamhd::mhd::initialize_fluid(
		geometries,
		initial_conditions,
		grid,
		cells,
		simulation_time,
		options_mhd.adiabatic_index,
		options_mhd.vacuum_permeability,
		options_mhd.proton_mass,
		verbose,
		Mas1, Mom1, Nrj1, Mag,
		Mas_f, Mom_f, Nrj_f
	);

	// particles
	std::mt19937_64 random_source;

	unsigned long long int nr_particles_created = 0;
	for (auto& init_cond_part: initial_conditions_particles) {
		nr_particles_created = pamhd::particle::initialize_particles<
			pamhd::particle::Particle_Internal,
			pamhd::particle::Mass,
			pamhd::particle::Charge_Mass_Ratio,
			pamhd::particle::Position,
			pamhd::particle::Velocity,
			pamhd::particle::Particle_ID,
			pamhd::particle::Species_Mass
		>(
			geometries,
			init_cond_part,
			simulation_time,
			cells,
			grid,
			random_source,
			options_particle.boltzmann,
			next_particle_id,
			grid.get_comm_size(),
			false,
			true,
			Part_Int,
			Bdy_N,
			Bdy_V,
			Bdy_T,
			Bdy_Nr_Par,
			Bdy_SpM,
			Bdy_C2M,
			Sol_Info_P
		);
		next_particle_id += nr_particles_created * grid.get_comm_size();
	}

	// accumulate particle data to fluid 2

	// magnetic field
	pamhd::mhd::initialize_magnetic_field<pamhd::mhd::Magnetic_Field>(
		geometries,
		initial_conditions,
		background_B,
		grid,
		cells,
		simulation_time,
		options_mhd.vacuum_permeability,
		Mag, Mag_f,
		Bg_B_Pos_X, Bg_B_Pos_Y, Bg_B_Pos_Z
	);
	// update background field between processes
	Cell::set_transfer_all(
		true,
		pamhd::mhd::Bg_Magnetic_Field_Pos_X(),
		pamhd::mhd::Bg_Magnetic_Field_Pos_Y(),
		pamhd::mhd::Bg_Magnetic_Field_Pos_Z()
	);
	grid.update_copies_of_remote_neighbors();
	Cell::set_transfer_all(
		false,
		pamhd::mhd::Bg_Magnetic_Field_Pos_X(),
		pamhd::mhd::Bg_Magnetic_Field_Pos_Y(),
		pamhd::mhd::Bg_Magnetic_Field_Pos_Z()
	);

	/*
	Add contribution of magnetic field energy to total energy of fluid 1
	*/

	// accumulate particle mass to Mas2 for fluid total energy adjustment
	for (const auto& cell: grid.cells) {
		// accumulate below doesn't zero the variable
		Bulk_Mass_Getter(*cell.data) = 0;
	}
	pamhd::particle::accumulate(
		cell_ids,
		grid,
		Part_Int,
		Particle_Position_Getter,
		Particle_Mass_Getter,
		Bulk_Mass_Getter,
		Accu_List_Bulk_Mass_Getter,
		Accu_List_Target_Getter,
		Accu_List_Length_Getter,
		Accu_List_Getter
	);

	Cell::set_transfer_all(true, pamhd::particle::Nr_Accumulated_To_Cells());
	grid.update_copies_of_remote_neighbors();
	Cell::set_transfer_all(false, pamhd::particle::Nr_Accumulated_To_Cells());

	pamhd::particle::allocate_accumulation_lists(
		grid,
		Accu_List_Getter,
		Accu_List_Length_Getter
	);

	Cell::set_transfer_all(true, pamhd::particle::Accumulated_To_Cells());
	grid.update_copies_of_remote_neighbors();
	Cell::set_transfer_all(false, pamhd::particle::Accumulated_To_Cells());

	pamhd::particle::accumulate_from_remote_neighbors(
		grid,
		Bulk_Mass_Getter,
		Accu_List_Bulk_Mass_Getter,
		Accu_List_Target_Getter,
		Accu_List_Getter
	);
	for (const auto& cell: grid.cells) {
		const auto length = grid.geometry.get_length(cell.id);
		const auto volume = length[0] * length[1] * length[2];
		Mas2(*cell.data) = Bulk_Mass_Getter(*cell.data) / volume;
	}

	// add magnetic field contribution to total energy density
	for (const auto& cell: grid.cells) {
		const auto
			total_mass = Mas1(*cell.data) + Mas2(*cell.data),
			mass_frac1 = Mas1(*cell.data) / total_mass;
		Nrj1(*cell.data) += mass_frac1 * 0.5 * Mag(*cell.data).squaredNorm() / options_mhd.vacuum_permeability;
	}

	for (const auto& cell: grid.cells) {
		const auto
			total_mass = Mas1(*cell.data) + Mas2(*cell.data),
			mass_frac1 = Mas1(*cell.data) / total_mass,
			mass_frac2 = Mas2(*cell.data) / total_mass;
		Nrj1(*cell_data) += mass_frac1 * 0.5 * Mag(*cell_data).squaredNorm() / vacuum_permeability;
		Nrj2(*cell_data) += mass_frac2 * 0.5 * Mag(*cell_data).squaredNorm() / vacuum_permeability;
	}


	/*
	TODO Classify cells into normal, boundary and dont_solve
	*/
	/*pamhd::particle::set_solver_info<pamhd::particle::Solver_Info>(
		grid, boundaries, geometries, Sol_Info_P
	);*/
	for (const auto& cell: grid.cells) {
		Sol_Info_P(*cell.data) = pamhd::particle::Solver_Info::normal;
	}
	// make lists from above for divergence removal functions
	std::vector<uint64_t> solve_cells, bdy_cells, skip_cells;
	for (const auto& cell: grid.cells) {
		if ((Sol_Info_P(*cell.data) & pamhd::particle::Solver_Info::dont_solve) > 0) {
			skip_cells.push_back(cell.id);
		} else if (Sol_Info_P(*cell.data) > 0) {
			bdy_cells.push_back(cell.id);
		} else {
			solve_cells.push_back(cell.id);
		}
	}

	// TODO: apply boundaries

	if (rank == 0) {
		cout << "done." << endl;
	}


	double
		max_dt = 0,
		simulation_time = start_time,
		next_particle_save = simulation_time + save_particle_n,
		next_mhd_save = simulation_time + save_mhd_n,
		next_rem_div_B = simulation_time + remove_div_B_n;

	size_t simulated_steps = 0;
	while (simulation_time < end_time) {
		simulated_steps++;

		double
			// don't step over the final simulation time
			until_end = end_time - simulation_time,
			local_time_step = min(min(time_step_factor * max_dt, until_end), max_time_step),
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

		accumulate_mhd_data(
			inner_cells,
			outer_cells,
			grid,
			Part_Int,
			Particle_Position_Getter,
			Particle_Mass_Getter,
			Particle_Species_Mass_Getter,
			Particle_Momentum_Getter,
			Particle_Relative_Velocity2_Getter,
			Number_Of_Particles_Getter,
			Bulk_Mass_Getter,
			Bulk_Momentum_Getter,
			Bulk_Relative_Velocity2_Getter,
			Bulk_Velocity_Getter,
			Accu_List_Number_Of_Particles_Getter,
			Accu_List_Bulk_Mass_Getter,
			Accu_List_Bulk_Momentum_Getter,
			Accu_List_Bulk_Relative_Velocity2_Getter,
			Accu_List_Target_Getter,
			Accu_List_Length_Getter,
			Accu_List_Getter,
			pamhd::particle::Nr_Accumulated_To_Cells(),
			pamhd::particle::Accumulated_To_Cells(),
			pamhd::particle::Bulk_Velocity()
		);

		// B required for E calculation
		Cell::set_transfer_all(true, pamhd::mhd::Magnetic_Field());
		grid.start_remote_neighbor_copy_updates();

		pamhd::particle::fill_mhd_fluid_values(
			cell_ids,
			grid,
			options_mhd.adiabatic_index,
			options_mhd.vacuum_permeability,
			options_particle.boltzmann,
			Number_Of_Particles_Getter,
			Bulk_Mass_Getter,
			Bulk_Momentum_Getter,
			Bulk_Relative_Velocity2_Getter,
			Part_Int,
			Mas2, Mom2, Nrj2, Mag
		);

		// inner: J for E = (J - V) x B
		pamhd::divergence::get_curl(
			inner_cells,
			grid,
			Mag,
			Cur
		);
		// not included in get_curl above
		for (const auto& cell: inner_cells) {
			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << ")" << std::endl;
				abort();
			}
			Cur(*cell_data) /= options_mhd.vacuum_permeability;
		}

		grid.wait_remote_neighbor_copy_update_receives();

		// outer: J for E = (J - V) x B
		pamhd::divergence::get_curl(
			outer_cells,
			grid,
			Mag,
			Cur
		);
		for (const auto& cell: outer_cells) {
			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << ")" << std::endl;
				abort();
			}
			Cur(*cell_data) /= options_mhd.vacuum_permeability;
		}

		grid.wait_remote_neighbor_copy_update_sends();
		Cell::set_transfer_all(false, pamhd::mhd::Magnetic_Field());

		// inner: E = (J - V) x B
		for (const auto& cell: inner_cells) {
			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << ")" << std::endl;
				abort();
			}

			Bulk_Velocity_Getter(*cell_data)
				= Bulk_Momentum_Getter(*cell_data)
				/ Bulk_Mass_Getter(*cell_data);

			J_m_V(*cell_data)
				= Cur(*cell_data)
				- (Mom1(*cell_data) + Mom2(*cell_data))
					/ (Mas1(*cell_data) + Mas2(*cell_data));
			Ele(*cell_data) = J_m_V(*cell_data).cross(Mag(*cell_data));
		}

		// outer: E = (J - V) x B
		for (const auto& cell: outer_cells) {
			auto* const cell_data = grid[cell];
			if (cell_data == nullptr) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << ")" << std::endl;
				abort();
			}

			Bulk_Velocity_Getter(*cell_data)
				= Bulk_Momentum_Getter(*cell_data)
				/ Bulk_Mass_Getter(*cell_data);

			J_m_V(*cell_data)
				= Cur(*cell_data)
				- (Mom1(*cell_data) + Mom2(*cell_data))
					/ (Mas1(*cell_data) + Mas2(*cell_data));
			Ele(*cell_data) = J_m_V(*cell_data).cross(Mag(*cell_data));
		}


		Cell::set_transfer_all(true, pamhd::particle::Current_Minus_Velocity());
		grid.update_copies_of_remote_neighbors();
		Cell::set_transfer_all(false, pamhd::particle::Current_Minus_Velocity());


		/*
		Solve
		*/

		if (rank == 0) {
			cout << "Solving at time " << simulation_time
				<< " s with time step " << time_step << " s" << endl;
		}

		max_dt = std::numeric_limits<double>::max();

		// outer particles
		max_dt = min(
			max_dt,
			pamhd::particle::solve<
				pamhd::particle::Current_Minus_Velocity,
				pamhd::mhd::Magnetic_Field,
				pamhd::particle::Nr_Particles_External,
				pamhd::particle::Particles_Internal,
				pamhd::particle::Particles_External,
				pamhd::particle::Position,
				pamhd::particle::Velocity,
				pamhd::particle::Charge_Mass_Ratio,
				pamhd::particle::Mass,
				pamhd::particle::Destination_Cell,
				boost::numeric::odeint::runge_kutta_fehlberg78<pamhd::particle::state_t>
			>(time_step, outer_cells, grid)
		);


		Cell::set_transfer_all(
			true,
			pamhd::mhd::Magnetic_Field(),
			pamhd::mhd::HD1_State(),
			pamhd::mhd::HD2_State(),
			//pamhd::mhd::Cell_Type_Field(),
			pamhd::particle::Nr_Particles_External()
		);
		grid.start_remote_neighbor_copy_updates();

		// inner MHD
		double solve_max_dt = -1;
		size_t solve_index = 0;

		std::tie(
			solve_max_dt,
			solve_index
		) = pamhd::mhd::N_solve(
			mhd_solver,
			0,
			grid,
			time_step,
			options_mhd.adiabatic_index,
			options_mhd.vacuum_permeability,
			std::make_pair(Mas1, Mas2),
			std::make_pair(Mom1, Mom2),
			std::make_pair(Nrj1, Nrj2),
			Mag,
			std::make_pair(Mas1_f, Mas2_f),
			std::make_pair(Mom1_f, Mom2_f),
			std::make_pair(Nrj1_f, Nrj2_f),
			Mag_f,
			Cell_t,
			bdy_classifier_fluid1.normal_cell,
			bdy_classifier_fluid1.dont_solve_cell
		);
		max_dt = min(
			max_dt,
			solve_max_dt
		);

		// inner particles
		max_dt = min(
			max_dt,
			pamhd::particle::solve<
				pamhd::particle::Current_Minus_Velocity,
				pamhd::mhd::Magnetic_Field,
				pamhd::particle::Nr_Particles_External,
				pamhd::particle::Particles_Internal,
				pamhd::particle::Particles_External,
				pamhd::particle::Position,
				pamhd::particle::Velocity,
				pamhd::particle::Charge_Mass_Ratio,
				pamhd::particle::Mass,
				pamhd::particle::Destination_Cell,
				boost::numeric::odeint::runge_kutta_fehlberg78<pamhd::particle::state_t>
			>(time_step, inner_cells, grid)
		);

		grid.wait_remote_neighbor_copy_update_receives();

		// outer MHD
		std::tie(
			solve_max_dt,
			std::ignore
		) = pamhd::mhd::N_solve(
			mhd_solver,
			solve_index + 1,
			grid,
			time_step,
			options_mhd.adiabatic_index,
			options_mhd.vacuum_permeability,
			std::make_pair(Mas1, Mas2),
			std::make_pair(Mom1, Mom2),
			std::make_pair(Nrj1, Nrj2),
			Mag,
			std::make_pair(Mas1_f, Mas2_f),
			std::make_pair(Mom1_f, Mom2_f),
			std::make_pair(Nrj1_f, Nrj2_f),
			Mag_f,
			Cell_t,
			bdy_classifier_fluid1.normal_cell,
			bdy_classifier_fluid1.dont_solve_cell
		);
		max_dt = min(
			max_dt,
			solve_max_dt
		);

		pamhd::particle::resize_receiving_containers<
			pamhd::particle::Nr_Particles_External,
			pamhd::particle::Particles_External
		>(remote_cell_ids, grid);

		grid.wait_remote_neighbor_copy_update_sends();
		Cell::set_transfer_all(
			false,
			pamhd::mhd::Magnetic_Field(),
			pamhd::mhd::HD1_State(),
			pamhd::mhd::HD2_State(),
			//pamhd::mhd::Cell_Type_Field(),
			pamhd::particle::Nr_Particles_External()
		);


		Cell::set_transfer_all(
			true,
			pamhd::particle::Particles_External()
		);
		grid.start_remote_neighbor_copy_updates();

		pamhd::mhd::apply_fluxes_N(
			grid,
			std::make_pair(Mas1, Mas2),
			std::make_pair(Mom1, Mom2),
			std::make_pair(Nrj1, Nrj2),
			Mag,
			std::make_pair(Mas1_f, Mas2_f),
			std::make_pair(Mom1_f, Mom2_f),
			std::make_pair(Nrj1_f, Nrj2_f),
			Mag_f,
			Cell_t,
			bdy_classifier_fluid1.normal_cell
		);

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
		Cell::set_transfer_all(
			false,
			pamhd::particle::Particles_External()
		);

		pamhd::particle::remove_external_particles<
			pamhd::particle::Nr_Particles_External,
			pamhd::particle::Particles_External
		>(outer_cells, grid);


		simulation_time += time_step;


		/*
		Solution(s) done for this step.
		*/

		// remove divergence of magnetic field
		if (remove_div_B_n > 0 and simulation_time >= next_rem_div_B) {
			next_rem_div_B += remove_div_B_n;

			if (rank == 0) {
				cout << "Removing divergence of B at time "
					<< simulation_time << "...  ";
				cout.flush();
			}

			// save old value of B in case div removal fails
			for (const auto& cell: cell_ids) {
				auto* const cell_data = grid[cell];
				if (cell_data == nullptr) {
					std::cerr <<  __FILE__ << "(" << __LINE__ << "): "
						"No data for cell " << cell
						<< std::endl;
					abort();
				}

				Mag_tmp(*cell_data) = Mag(*cell_data);
			}

			Cell::set_transfer_all(
				true,
				pamhd::mhd::Magnetic_Field(),
				pamhd::mhd::Magnetic_Field_Divergence()
			);
			const auto div_before
				= pamhd::divergence::remove(
					cell_ids,
					{},
					{},
					grid,
					Mag,
					Mag_div,
					[](Cell& cell_data)
						-> typename pamhd::mhd::Scalar_Potential_Gradient::data_type&
					{
						return cell_data[pamhd::mhd::Scalar_Potential_Gradient()];
					},
					options_mhd.poisson_iterations_max,
					options_mhd.poisson_iterations_min,
					options_mhd.poisson_norm_stop,
					2,
					options_mhd.poisson_norm_increase_max,
					false
				);
			Cell::set_transfer_all(false, pamhd::mhd::Magnetic_Field_Divergence());

			grid.update_copies_of_remote_neighbors();
			Cell::set_transfer_all(false, pamhd::mhd::Magnetic_Field());
			const double div_after
				= pamhd::divergence::get_divergence(
					cell_ids,
					grid,
					Mag,
					Mag_div
				);

			// restore old B
			if (div_after > div_before) {
				if (rank == 0) {
					cout << "failed (" << div_after
						<< "), restoring previous value (" << div_before << ")."
						<< endl;
				}
				for (const auto& cell: cell_ids) {
					auto* const cell_data = grid[cell];
					if (cell_data == nullptr) {
						std::cerr <<  __FILE__ << "(" << __LINE__ << "): "
							"No data for cell " << cell
							<< std::endl;
						abort();
					}

					Mag(*cell_data) = Mag_tmp(*cell_data);
				}

			} else {
				if (rank == 0) {
					cout << div_before << " -> " << div_after << endl;
				}
				// TODO: keep pressure/temperature constant over div removal
				// needs latest accumulated particle mass
			}

		}



				/*pamhd::particle::fill_mhd_fluid_values(
					{cell_id},
					grid,
					options_mhd.adiabatic_index,
					options_mhd.vacuum_permeability,
					options_particle.boltzmann,
					Number_Of_Particles_Getter,
					Bulk_Mass_Getter,
					Bulk_Momentum_Getter,
					Bulk_Relative_Velocity2_Getter,
					Part_Int,
					Mas2, Mom2, Nrj2, Mag
				);*/

			// add magnetic nrj to fluid boundary cells
			/*if ((*cell_data)[pamhd::mhd::Cell_Type()] == bdy_classifier_fluid1.value_boundary_cell) {
				const auto& bdy_id = (*cell_data)[pamhd::mhd::Value_Boundary_Id()];

				const auto mass_frac
					= Mas1(*cell_data)
					/ (Mas1(*cell_data) + Mas2(*cell_data));
				Nrj1(*cell_data) += mass_frac * 0.5 * Mag(*cell_data).squaredNorm() / options_mhd.vacuum_permeability;
			}*/

		// update particle lists
		/*pamhd::particle::resize_receiving_containers<
			pamhd::particle::Nr_Particles_Internal,
			pamhd::particle::Particles_Internal
		>(remote_cell_ids, grid);
		pamhd::particle::resize_receiving_containers<
			pamhd::particle::Nr_Particles_External,
			pamhd::particle::Particles_External
		>(remote_cell_ids, grid);
		Cell::set_transfer_all(
			true,
			pamhd::particle::Particles_Internal(),
			pamhd::particle::Particles_External()
		);
		grid.update_copies_of_remote_neighbors();
		Cell::set_transfer_all(
			false,
			pamhd::particle::Particles_Internal(),
			pamhd::particle::Particles_External()
		);*/

		// TODO copy boundaries

		/*
		Save result(s) to file(s)
		*/
		if (
			(save_particle_n >= 0 and (simulation_time == 0 or simulation_time >= end_time))
			or (save_particle_n > 0 and simulation_time >= next_particle_save)
		) {
			if (next_particle_save <= simulation_time) {
				next_particle_save += save_particle_n;
			}

			if (rank == 0) {
				cout << "Saving particles at time " << simulation_time << "...";
			}

			if (
				not pamhd::particle::save<
					pamhd::particle::Electric_Field,
					pamhd::mhd::Magnetic_Field,
					pamhd::mhd::Electric_Current_Density,
					pamhd::particle::Nr_Particles_Internal,
					pamhd::particle::Particles_Internal
				>(
					boost::filesystem::canonical(
						boost::filesystem::path(output_directory)
					).append("particle_").generic_string(),
					grid,
					simulation_time,
					options_mhd.adiabatic_index,
					options_mhd.vacuum_permeability,
					options_particle.boltzmann
				)
			) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << "): "
					"Couldn't save particle result."
					<< std::endl;
				MPI_Finalize();
				return EXIT_FAILURE;
			}

			if (rank == 0) {
				cout << "done." << endl;
			}
		}

		if (
			(save_mhd_n >= 0 and (simulation_time == 0 or simulation_time >= end_time))
			or (save_mhd_n > 0 and simulation_time >= next_mhd_save)
		) {
			if (next_mhd_save <= simulation_time) {
				next_mhd_save += save_mhd_n;
			}

			if (rank == 0) {
				cout << "Saving fluid at time " << simulation_time << "... ";
			}

			if (
				not pamhd::mhd::save(
					boost::filesystem::canonical(
						boost::filesystem::path(output_directory)
					).append("2mhd_").generic_string(),
					grid,
					0,
					simulation_time,
					options_mhd.adiabatic_index,
					options_mhd.proton_mass,
					options_mhd.vacuum_permeability,
					pamhd::mhd::HD1_State(),
					pamhd::mhd::HD2_State(),
					pamhd::mhd::Magnetic_Field(),
					pamhd::mhd::Electric_Current_Density(),
					//pamhd::mhd::Cell_Type(),
					pamhd::mhd::MPI_Rank(),
					pamhd::mhd::Resistivity()
				)
			) {
				std::cerr <<  __FILE__ << "(" << __LINE__ << "): "
					"Couldn't save mhd result."
					<< std::endl;
				abort();
			}

			if (rank == 0) {
				cout << "done." << endl;
			}
		}
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
