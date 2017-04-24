#! /bin/sh -e

# if given, override RUN with first command line argument
RUN=${1:-}

echo
echo Compiling programs
echo

make


echo
echo Running tests...
echo

# two-fluid magnetic field advection
echo magnetic_field_advection 1 1d +x hll_athena
$RUN ./two_test.exe config_files/magnetohydrodynamic/magnetic_field_advection/1d/+x/two_pop1.json
$RUN ./2mhd2gnuplot.exe results/magnetohydrodynamic/magnetic_field_advection/1d/+x/2hll_athena1/2mhd*dc 2> /dev/null

echo magnetic_field_advection 1 1d -z hll_athena
$RUN ./two_test.exe config_files/magnetohydrodynamic/magnetic_field_advection/1d/-z/two_pop1.json
$RUN ./2mhd2gnuplot.exe results/magnetohydrodynamic/magnetic_field_advection/1d/-z/2hll_athena1/2mhd*dc 2> /dev/null

echo magnetic_field_advection 2 1d +x hll_athena
$RUN ./two_test.exe config_files/magnetohydrodynamic/magnetic_field_advection/1d/+x/two_pop2.json
$RUN ./2mhd2gnuplot.exe results/magnetohydrodynamic/magnetic_field_advection/1d/+x/2hll_athena2/2mhd*dc 2> /dev/null


# shock tube
echo shock_tube mhd 0 1d +x hll_athena
$RUN ./two_test.exe config_files/magnetohydrodynamic/shock_tube/1d/+x/two_pop0.json
$RUN ./2mhd2gnuplot.exe results/magnetohydrodynamic/shock_tube/1d/+x/2hll_athena0/2mhd*dc 2> /dev/null

echo shock_tube mhd 1 1d +x hll_athena
$RUN ./two_test.exe config_files/magnetohydrodynamic/shock_tube/1d/+x/two_pop1.json
$RUN ./2mhd2gnuplot.exe results/magnetohydrodynamic/shock_tube/1d/+x/2hll_athena1/2mhd*dc 2> /dev/null

echo shock_tube mhd 2 1d +x hll_athena
$RUN ./two_test.exe config_files/magnetohydrodynamic/shock_tube/1d/+x/two_pop2.json
$RUN ./2mhd2gnuplot.exe results/magnetohydrodynamic/shock_tube/1d/+x/2hll_athena2/2mhd*dc 2> /dev/null

echo shock_tube mhd 3 1d +x hll_athena
$RUN ./two_test.exe config_files/magnetohydrodynamic/shock_tube/1d/+x/two_pop3.json
$RUN ./2mhd2gnuplot.exe results/magnetohydrodynamic/shock_tube/1d/+x/2hll_athena3/2mhd*dc 2> /dev/null

echo shock_tube mhd 4 1d +x hll_athena
$RUN ./two_test.exe config_files/magnetohydrodynamic/shock_tube/1d/+x/two_pop4.json
$RUN ./2mhd2gnuplot.exe results/magnetohydrodynamic/shock_tube/1d/+x/2hll_athena4/2mhd*dc 2> /dev/null


# Orszag-Tang
echo orszag-tang mhd 2d z1 hll_athena
$RUN ./two_test.exe config_files/magnetohydrodynamic/orszag-tang/2d/z1/two_pop0.json
$RUN ./2mhd2gnuplot.exe results/magnetohydrodynamic/orszag-tang/2d/z1/2hll_athena0/2mhd*dc 2> /dev/null
