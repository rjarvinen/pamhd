#! /usr/bin/env python3
'''
Program for creating configuration files for test particle program from output of MHD test program.

Copyright 2017 Ilja Honkonen
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

* Neither the name of copyright holders nor the names of their contributors
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
'''

from argparse import ArgumentParser
from json import dump, encoder, load
import os

try:
	import numpy
except:
	exit("Couldn't import numpy")


'''
Returns simulation time in given file.
'''
def get_simulation_time(file_name):
	if not os.path.isfile(file_name):
		raise Exception('Given file name (' + file_name + ') is not a file.')

	infile = open(file_name, 'rb')

	# read simulation header, given by source/mhd/save.hpp
	file_version = numpy.fromfile(infile, dtype = 'uint64', count = 1)
	if file_version != 2:
		exit('Unsupported file version: ' + str(file_version))
	sim_params = numpy.fromfile(infile, dtype = '4double', count = 1)
	simulation_time, adiabatic_index, proton_mass, vacuum_permeability = sim_params[0]

	# check file endiannes
	endianness = numpy.fromfile(infile, dtype = 'uint64', count = 1)[0]
	if endianness != numpy.uint64(0x1234567890abcdef):
		raise Exception(
			'Wrong endiannes in given file, expected ' + str(hex(0x1234567890abcdef)) \
			+ ' but got ' + str(hex(endianness))
		)

	return simulation_time


'''
Returns data of PAMHD MHD test program output.

Does nothing and returns None if mhd_data is None.

Returns the following in a numpy array:
simulation time,
adiabatic index,
proton_mass,
vacuum permeability.

mhd_data will have cell ids as keys and each value is a list of
tuples of following items:
coordinate of cell's center,
cell's length,
mass density,
momentum density,
total energy density,
magnetic field,
electric current density,
cell type,
mpi rank,
electric resistivity,
background magnetic field at cell face in positive x direction,
background magnetic field at cell face in positive y direction,
background magnetic field at cell face in positive z direction

Every call appends only one tuple to each key, i.e. if called
on an empty dictionary every list will have only one tuple.
'''
def load_mhd(file_name, mhd_data, x_min, x_max, y_min, y_max, z_min, z_max):
	if mhd_data == None:
		return None

	if not os.path.isfile(file_name):
		raise Exception('Given file name (' + file_name + ') is not a file.')

	infile = open(file_name, 'rb')

	# read simulation header, given by source/mhd/save.hpp
	file_version = numpy.fromfile(infile, dtype = 'uint64', count = 1)
	if file_version != 2:
		exit('Unsupported file version: ' + str(file_version))
	sim_params = numpy.fromfile(infile, dtype = '4double', count = 1)

	# from this point onward format is given by dccrg's save_grid_data()
	# check file endiannes
	endianness = numpy.fromfile(infile, dtype = 'uint64', count = 1)[0]
	if endianness != numpy.uint64(0x1234567890abcdef):
		raise Exception(
			'Wrong endiannes in given file, expected ' + str(hex(0x1234567890abcdef)) \
			+ ' but got ' + str(hex(endianness))
		)

	# number of refinement level 0 cells in each dimension
	ref_lvl_0_cells = numpy.fromfile(infile, dtype = '3uint64', count = 1)[0]
	#print(ref_lvl_0_cells)

	# maximum refinement level of grid cells
	max_ref_lvl = numpy.fromfile(infile, dtype = 'intc', count = 1)[0]
	if max_ref_lvl > numpy.uint32(0):
		raise Exception('Refinement level > 0 not supported')

	# length of every cells' neighborhood in cells of identical size
	neighborhood_length = numpy.fromfile(infile, dtype = 'uintc', count = 1)[0]
	#print(neighborhood_length)

	# whether grid is periodic each dimension (0 == no, 1 == yes)
	periodicity = numpy.fromfile(infile, dtype = '3uint8', count = 1)[0]
	#print(periodicity)

	geometry_id = numpy.fromfile(infile, dtype = 'intc', count = 1)[0]
	if geometry_id != numpy.int32(1):
		raise Exception('Unsupported geometry')

	# starting coordinate of grid
	grid_start = numpy.fromfile(infile, dtype = '3double', count = 1)[0]
	#print(grid_start)

	# length of cells of refinement level 0
	lvl_0_cell_length = numpy.fromfile(infile, dtype = '3double', count = 1)[0]
	#print(lvl_0_cell_length)

	# total number of cells in grid
	total_cells = numpy.fromfile(infile, dtype = 'uint64', count = 1)[0]
	#print(total_cells)

	# id of each cell and offset in bytes to data of each cell
	cell_ids_data_offsets = numpy.fromfile(infile, dtype = '2uint64', count = total_cells)
	#print(cell_ids_data_offsets)

	# until this point format decided by dccrg
	# from this point onward format decided by save() call of tests/mhd/test.cpp

	# read mhd data
	for item in cell_ids_data_offsets:
		cell_id = item[0]

		# calculate cell geometry, defined by get_center() function in 
		# dccrg_cartesian_geometry.hpp file of dccrg
		cell_id = int(cell_id - 1)
		cell_index = (
			int(cell_id % ref_lvl_0_cells[0]),
			int(cell_id / ref_lvl_0_cells[0] % ref_lvl_0_cells[1]),
			int(cell_id / (ref_lvl_0_cells[0] * ref_lvl_0_cells[1]))
		)

		cell_center = (
			grid_start[0] + lvl_0_cell_length[0] * (0.5 + cell_index[0]),
			grid_start[1] + lvl_0_cell_length[1] * (0.5 + cell_index[1]),
			grid_start[2] + lvl_0_cell_length[2] * (0.5 + cell_index[2])
		)
		if \
			cell_center[0] < x_min or cell_center[0] > x_max \
			or cell_center[1] < y_min or cell_center[1] > y_max \
			or cell_center[2] < z_min or cell_center[2] > z_max \
		:
			continue

		cell_id = int(cell_id + 1)

		infile.seek(item[1], 0)
		temp = numpy.fromfile(
			infile,
			dtype = 'double, 3double, double, 3double, 3double, intc, intc, double, 3double, 3double, 3double',
			count = 1
		)[0]

		data = (
			cell_center,
			lvl_0_cell_length,
			temp[0],
			temp[1],
			temp[2],
			temp[3],
			temp[4],
			temp[5],
			temp[6],
			temp[7],
			temp[8],
			temp[9],
			temp[10]
		)
		if cell_id in mhd_data:
			mhd_data[cell_id].append(data)
		else:
			mhd_data[cell_id] = [data]

	return sim_params, ref_lvl_0_cells, grid_start, lvl_0_cell_length


'''
Returns lists of coordinates and values from mhd_data filled by load_mhd.

Returns lists of x, y, z, values where values are arranged in so that
values' x coordinate increases first, then y and lastly z. This format
is suitable for pamhd::boundaries::Multivariable_Boundaries class.
'''
def get_coords_and_values(mhd_data, ref_lvl_0_cells, grid_start, lvl_0_cell_length):
	cells = sorted([cell for cell in mhd_data])
	if len(cells) == 0:
		exit('Unexpected number of cells in mhd_data: ' + str(len(cells)))

	# get data coordinates in every dimension
	xs, ys, zs, values = [], [], [], []
	xis, yis, zis = set(), set(), set()
	for cell in cells:
		tmp = int(cell - 1)
		index = (
			int(tmp % ref_lvl_0_cells[0]),
			int(tmp / ref_lvl_0_cells[0] % ref_lvl_0_cells[1]),
			int(tmp / (ref_lvl_0_cells[0] * ref_lvl_0_cells[1]))
		)
		xis.add(index[0])
		yis.add(index[1])
		zis.add(index[2])
	xis = sorted([x for x in xis])
	yis = sorted([y for y in yis])
	zis = sorted([z for z in zis])

	xs = [grid_start[0] + lvl_0_cell_length[0] * (0.5 + xi) for xi in xis]
	ys = [grid_start[1] + lvl_0_cell_length[1] * (0.5 + yi) for yi in yis]
	zs = [grid_start[2] + lvl_0_cell_length[2] * (0.5 + zi) for zi in zis]

	for time_index in range(len(mhd_data[cells[0]])):
		for cell in cells:
			values.append(mhd_data[cell][time_index])

	return xs, ys, zs, values


if __name__ == '__main__':

	parser = ArgumentParser(
		description
			= 'Converts MHD output to boundary conditions of test particle program.'
	)
	parser.add_argument(
		'--pretty',
		default = False,
		action = 'store_true',
		help = 'Write output in more easily readable format'
	)
	# TODO
	#parser.add_argument(
	#	'--precision',
	#	default = 16,
	#	help = 'Write floats in scientific notation using this many decimal digits'
	#)
	parser.add_argument(
		'--solver',
		default = 'rkf78',
		help = 'Particle solver'
	)
	parser.add_argument(
		'--boltzmann',
		type = float,
		default = 1.38064852e-23,
		help = 'Relates average particle energy to temperature'
	)
	parser.add_argument(
		'--in-json',
		metavar = 'I',
		type = str,
		required = True,
		help = 'Use file I as baseline configuration and replace geometry, initial and boundary conditions'
	)
	parser.add_argument(
		'--out-json',
		metavar = 'O',
		type = str,
		required = True,
		help = 'Write result to file O'
	)
	parser.add_argument(
		'--fields-min-x',
		type = float,
		default = -1e308,
		help = 'Minimum extent in x dimension of time-dependent value boundary box for electric and magnetic fields (prepend negative numbers with a space, e.g. " -1" in bash)'
	)
	parser.add_argument(
		'--fields-max-x',
		type = float,
		default = 1e308,
		help = 'Maximum extent of E & B value boundary box'
	)
	parser.add_argument(
		'--fields-min-y',
		type = float,
		default = -1e308,
		help = 'Minimum Y extent of boundary box'
	)
	parser.add_argument(
		'--fields-max-y',
		type = float,
		default = 1e308,
		help = 'Maximum y extent of box'
	)
	parser.add_argument(
		'--fields-min-z',
		type = float,
		default = -1e308,
		help = 'Minimum z extent of box'
	)
	parser.add_argument(
		'--fields-max-z',
		type = float,
		default = 1e308,
		help = 'Maximum z extent of box'
	)
	parser.add_argument(
		'--fluid-min-x',
		type = float,
		action = 'append',
		default = [],
		help = 'Minimum extent in x dimension of time-dependent value boundary box(es) for particles created from fluid data'
	)
	parser.add_argument(
		'--fluid-max-x',
		type = float,
		action = 'append',
		default = [],
		help = 'Maximum extent in x dimension of value boundary box(es) for particles'
	)
	parser.add_argument(
		'--fluid-min-y',
		type = float,
		action = 'append',
		default = [],
		help = 'Minimum y extent of box(es)'
	)
	parser.add_argument(
		'--fluid-max-y',
		type = float,
		action = 'append',
		default = [],
		help = 'Maximum y extent of box(es)'
	)
	parser.add_argument(
		'--fluid-min-z',
		type = float,
		action = 'append',
		default = [],
		help = 'Minimum z extent of box(es)'
	)
	parser.add_argument(
		'--fluid-max-z',
		type = float,
		action = 'append',
		default = [],
		help = 'Maximum z extent of box(es)'
	)
	parser.add_argument(
		'--nr-particles',
		type = float,
		action = 'append',
		default = [],
		help = 'Number of particles to create in every cell within fluid value boundary box(es)'
	)
	parser.add_argument(
		'--c2m',
		type = float,
		action = 'append',
		default = [],
		help = 'Charge to mass ratio of created particles within fluid value boundary box(es)'
	)
	parser.add_argument(
		'--spm',
		type = float,
		action = 'append',
		default = [],
		help = 'Species mass of created particles within fluid value boundary box(es)'
	)
	parser.add_argument(
		'dc_file',
		type = str,
		nargs = '*',
		help = 'One or more files from MHD test program to use when creating time-dependent value boundaries for fields and particles, can be given in any order'
	)

	args = parser.parse_args()


	with open(args.in_json, 'r') as infile:
		json_data = load(infile)
	if type(json_data) != type(dict()):
		exit('JSON data in input file must be an object but is ' + str(type(json_data)))


	if 'save-mhd-n' in json_data:
		json_data['save-particle-n'] = json_data.pop('save-mhd-n')
	json_data['particle-temp-nrj-ratio'] = args.boltzmann
	json_data['solver-particle'] = args.solver

	json_data.pop('geometries', None)
	json_data.pop('electric-field', None)
	json_data.pop('magnetic-field', None)
	json_data.pop('nr-particles', None)
	json_data.pop('velocity', None)
	json_data.pop('number-density', None)
	json_data.pop('species-mass', None)
	json_data.pop('charge-mass-ratio', None)
	json_data.pop('temperature', None)
	json_data.pop('pressure', None)


	if \
		len(args.fluid_min_x) != len(args.fluid_max_x) \
		or len(args.fluid_min_x) != len(args.fluid_min_y) \
		or len(args.fluid_min_x) != len(args.fluid_max_y) \
		or len(args.fluid_min_x) != len(args.fluid_min_z) \
		or len(args.fluid_min_x) != len(args.fluid_max_z) \
	:
		exit('Number of minimum and maximum extents for fluid boundaries must be equal.')

	if \
		len(args.fluid_min_x) != len(args.nr_particles) \
		or len(args.fluid_min_x) != len(args.c2m) \
		or len(args.fluid_min_x) != len(args.spm) \
	:
		exit('Number of particle properties (nr particles, charge to mass ratio, species mass) not equal to number of fluid boundaries')

	# load electric and magnetic field data
	times_infiles = sorted([(get_simulation_time(f), f) for f in args.dc_file])

	mhd_data = dict()
	sim_params, ref_lvl_0_cells, grid_start, lvl_0_cell_length = None, None, None, None
	for item in times_infiles:
		sim_params, ref_lvl_0_cells, grid_start, lvl_0_cell_length \
			= load_mhd(item[1], mhd_data, args.fields_min_x, args.fields_max_x, args.fields_min_y, args.fields_max_y, args.fields_min_z, args.fields_max_z)

	xs, ys, zs, values = get_coords_and_values(mhd_data, ref_lvl_0_cells, grid_start, lvl_0_cell_length)

	# create field boundaries and their geometry
	box = dict()
	box['start'] = [args.fields_min_x, args.fields_min_y, args.fields_min_z]
	box['end'] = [args.fields_max_x, args.fields_max_y, args.fields_max_z]
	field_geom = dict()
	field_geom['box'] = box
	json_data['geometries'] = [field_geom]

	# magnetic field
	mag_bdy_item = dict()
	mag_bdy_item['geometry-id'] = 0
	mag_bdy_item['time-stamps'] = [item[0] for item in times_infiles]
	mag_bdy_item['values'] = dict()
	mag_bdy_item['values']['x'] = xs
	mag_bdy_item['values']['y'] = ys
	mag_bdy_item['values']['z'] = zs
	mag_bdy_item['values']['data'] = [list(value[5]) for value in values]
	mag = dict()
	mag['default'] = [0, 0, 0]
	mag['value-boundaries'] = [mag_bdy_item]
	json_data['magnetic-field'] = mag

	# electric field
	ele_bdy_item = dict()
	ele_bdy_item['geometry-id'] = 0
	ele_bdy_item['time-stamps'] = [item[0] for item in times_infiles]
	ele_bdy_item['values'] = dict()
	ele_bdy_item['values']['x'] = xs
	ele_bdy_item['values']['y'] = ys
	ele_bdy_item['values']['z'] = zs
	ele_bdy_item['values']['data'] = []
	for value in values:
		J = value[6]
		V = value[3] / value[2]
		B = value[5]
		E = numpy.cross(J - V, B)
		ele_bdy_item['values']['data'].append(list(E))
	ele = dict()
	ele['default'] = [0, 0, 0]
	ele['value-boundaries'] = [ele_bdy_item]
	json_data['electric-field'] = ele

	# particle boundaries
	for i in range(len(args.fluid_min_x)):

		mhd_data = dict()
		for item in times_infiles:
			sim_params, ref_lvl_0_cells, grid_start, lvl_0_cell_length \
				= load_mhd(item[1], mhd_data, args.fluid_min_x[i], args.fluid_max_x[i], args.fluid_min_y[i], args.fluid_max_y[i], args.fluid_min_z[i], args.fluid_max_z[i])

		xs, ys, zs, values = get_coords_and_values(mhd_data, ref_lvl_0_cells, grid_start, lvl_0_cell_length)

		simulation_time, adiabatic_index, proton_mass, vacuum_permeability \
			= sim_params[0]

		box = dict()
		box['start'] = [args.fluid_min_x[i], args.fluid_min_y[i], args.fluid_min_z[i]]
		box['end'] = [args.fluid_max_x[i], args.fluid_max_y[i], args.fluid_max_z[i]]
		fluid_geom = dict()
		fluid_geom['box'] = box
		json_data['geometries'].append(fluid_geom)

		# number density
		nr_den_bdy_item = dict()
		nr_den_bdy_item['geometry-id'] = i + 1
		nr_den_bdy_item['time-stamps'] = [item[0] for item in times_infiles]
		nr_den_bdy_item['values'] = dict()
		nr_den_bdy_item['values']['x'] = xs
		nr_den_bdy_item['values']['y'] = ys
		nr_den_bdy_item['values']['z'] = zs
		nr_den_bdy_item['values']['data'] = [value[2] / proton_mass for value in values]
		if not 'number-density' in json_data:
			nr_den = dict()
			nr_den['default'] = 0
			nr_den['value-boundaries'] = [nr_den_bdy_item]
			json_data['number-density'] = nr_den
		else:
			json_data['number-density']['value-boundaries'].append(nr_den_bdy_item)

		# temperature
		temp_bdy_item = dict()
		temp_bdy_item['geometry-id'] = i + 1
		temp_bdy_item['time-stamps'] = [item[0] for item in times_infiles]
		temp_bdy_item['values'] = dict()
		temp_bdy_item['values']['x'] = xs
		temp_bdy_item['values']['y'] = ys
		temp_bdy_item['values']['z'] = zs
		temp_bdy_item['values']['data'] = []
		for value in values:
			total_energy = value[4]
			kinetic_energy = 0.5 / value[2] * numpy.dot(value[3], value[3])
			magnetic_energy = 0.5 / vacuum_permeability * numpy.dot(value[5], value[5])
			pressure = (total_energy - kinetic_energy - magnetic_energy) * (adiabatic_index - 1)
			temperature = pressure / (value[2] / proton_mass) / args.boltzmann
			temp_bdy_item['values']['data'].append(temperature)
		if not 'temperature' in json_data:
			temp = dict()
			temp['default'] = 0
			temp['value-boundaries'] = [temp_bdy_item]
			json_data['temperature'] = temp
		else:
			json_data['temperature']['value-boundaries'].append(temp_bdy_item)

		# velocity
		vel_bdy_item = dict()
		vel_bdy_item['geometry-id'] = i + 1
		vel_bdy_item['time-stamps'] = [item[0] for item in times_infiles]
		vel_bdy_item['values'] = dict()
		vel_bdy_item['values']['x'] = xs
		vel_bdy_item['values']['y'] = ys
		vel_bdy_item['values']['z'] = zs
		vel_bdy_item['values']['data'] = [list(value[3] / value[2]) for value in values]
		if not 'velocity' in json_data:
			vel = dict()
			vel['default'] = [0, 0, 0]
			vel['value-boundaries'] = [vel_bdy_item]
			json_data['velocity'] = vel
		else:
			json_data['velocity']['value-boundaries'].append(vel_bdy_item)

		# number of particles
		nr_par_bdy_item = dict()
		nr_par_bdy_item['geometry-id'] = i + 1
		nr_par_bdy_item['time-stamps'] = [item[0] for item in times_infiles]
		nr_par_bdy_item['values'] = dict()
		nr_par_bdy_item['values']['x'] = xs
		nr_par_bdy_item['values']['y'] = ys
		nr_par_bdy_item['values']['z'] = zs
		nr_par_bdy_item['values']['data'] = [args.nr_particles[i] for j in range(len(values))]
		if not 'nr-particles' in json_data:
			nr_par = dict()
			nr_par['default'] = 0
			nr_par['value-boundaries'] = [nr_par_bdy_item]
			json_data['nr-particles'] = nr_par
		else:
			json_data['nr-particles']['value-boundaries'].append(nr_par_bdy_item)

		# charge to mass ratio
		c2m_bdy_item = dict()
		c2m_bdy_item['geometry-id'] = i + 1
		c2m_bdy_item['time-stamps'] = [item[0] for item in times_infiles]
		c2m_bdy_item['values'] = dict()
		c2m_bdy_item['values']['x'] = xs
		c2m_bdy_item['values']['y'] = ys
		c2m_bdy_item['values']['z'] = zs
		c2m_bdy_item['values']['data'] = [args.c2m[i] for j in range(len(values))]
		if not 'charge-mass-ratio' in json_data:
			c2m = dict()
			c2m['default'] = 0
			c2m['value-boundaries'] = [c2m_bdy_item]
			json_data['charge-mass-ratio'] = c2m
		else:
			json_data['charge-mass-ratio']['value-boundaries'].append(c2m_bdy_item)

		# particle species mass
		spm_bdy_item = dict()
		spm_bdy_item['geometry-id'] = i + 1
		spm_bdy_item['time-stamps'] = [item[0] for item in times_infiles]
		spm_bdy_item['values'] = dict()
		spm_bdy_item['values']['x'] = xs
		spm_bdy_item['values']['y'] = ys
		spm_bdy_item['values']['z'] = zs
		spm_bdy_item['values']['data'] = [args.spm[i] for j in range(len(values))]
		if not 'species-mass' in json_data:
			spm = dict()
			spm['default'] = 0
			spm['value-boundaries'] = [spm_bdy_item]
			json_data['species-mass'] = spm
		else:
			json_data['species-mass']['value-boundaries'].append(spm_bdy_item)

	with open(args.out_json, 'w') as outfile:
		if args.pretty:
			dump(json_data, outfile, sort_keys = True, indent = 4)
		else:
			dump(json_data, outfile)
