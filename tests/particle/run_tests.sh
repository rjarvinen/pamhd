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

# test particle gyration in uniform magnetic field
echo rotation 0d +x B0 euler
$RUN ./massless.exe config_files/massless/rotation/+x/B0_euler.json
$RUN ./particle2gnuplot.exe --horizontal-variable ry --vertical-variable rz results/massless/rotation/+x/B0_euler/*dc

echo rotation 0d +x B1 euler
$RUN ./massless.exe config_files/massless/rotation/+x/B1_euler.json
$RUN ./particle2gnuplot.exe --horizontal-variable ry --vertical-variable rz results/massless/rotation/+x/B1_euler/*dc

echo rotation 0d +x B0+B1 euler
$RUN ./massless.exe config_files/massless/rotation/+x/B01_euler.json
$RUN ./particle2gnuplot.exe --horizontal-variable ry --vertical-variable rz results/massless/rotation/+x/B01_euler/*dc

echo rotation 0d -x B0+B1 euler
$RUN ./massless.exe config_files/massless/rotation/-x/B01_euler.json
$RUN ./particle2gnuplot.exe --horizontal-variable ry --vertical-variable rz results/massless/rotation/-x/B01_euler/*dc

echo rotation 0d +y B0+B1 euler
$RUN ./massless.exe config_files/massless/rotation/+y/B01_euler.json
$RUN ./particle2gnuplot.exe --horizontal-variable rx --vertical-variable rz results/massless/rotation/+y/B01_euler/*dc

echo rotation 0d -y B0+B1 euler
$RUN ./massless.exe config_files/massless/rotation/-y/B01_euler.json
$RUN ./particle2gnuplot.exe --horizontal-variable rx --vertical-variable rz results/massless/rotation/-y/B01_euler/*dc

echo rotation 0d +z B0+B1 euler
$RUN ./massless.exe config_files/massless/rotation/+z/B01_euler.json
$RUN ./particle2gnuplot.exe results/massless/rotation/+z/B01_euler/*dc

echo rotation 0d -z B0+B1 euler
$RUN ./massless.exe config_files/massless/rotation/-z/B01_euler.json
$RUN ./particle2gnuplot.exe results/massless/rotation/-z/B01_euler/*dc


# particles crossing a shock
echo shock crossing 1d +x euler
$RUN ./massless.exe config_files/massless/shock_crossing/1d/+x/euler.json
$RUN ./particle2gnuplot.exe --vertical-variable vx results/massless/shock_crossing/1d/+x/euler/*dc

echo shock crossing 1d +x midpoint
$RUN ./massless.exe config_files/massless/shock_crossing/1d/+x/midpoint.json
$RUN ./particle2gnuplot.exe --vertical-variable vx results/massless/shock_crossing/1d/+x/midpoint/*dc

echo shock crossing 1d +x rk4
$RUN ./massless.exe config_files/massless/shock_crossing/1d/+x/rk4.json
$RUN ./particle2gnuplot.exe --vertical-variable vx results/massless/shock_crossing/1d/+x/rk4/*dc

echo shock crossing 1d +x rkck54
$RUN ./massless.exe config_files/massless/shock_crossing/1d/+x/rkck54.json
$RUN ./particle2gnuplot.exe --vertical-variable vx results/massless/shock_crossing/1d/+x/rkck54/*dc

echo shock crossing 1d +x rkf78
$RUN ./massless.exe config_files/massless/shock_crossing/1d/+x/rkf78.json
$RUN ./particle2gnuplot.exe --vertical-variable vx results/massless/shock_crossing/1d/+x/rkf78/*dc
