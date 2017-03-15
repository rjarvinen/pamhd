# to compile and test pick one makefile from makefiles directory
# or write a new one for your environment
ENVIRONMENT_MAKEFILE = makefiles/homedir

#
# The lines below are not intended to be modified by users
#
CXXFLAGS = -std=c++11 -W -Wall -Wextra -pedantic -O3 -DBOOST_SYSTEM_NO_DEPRECATED
CPPFLAGS = -I source
include $(ENVIRONMENT_MAKEFILE)

# filled by project makefiles
EXECUTABLES =
TESTS =
RESULTS =
CLEAN =

default: all

include \
  tests/prettyprint/project_makefile \
  tests/grid_options/project_makefile \
  tests/muparserx/project_makefile \
  tests/mhd/project_makefile \
  tests/boundaries/project_makefile \
  tests/divergence/project_makefile


all: $(EXECUTABLES)

t: test
test: all $(TESTS)

# removes simulation results
r: results
results: $(RESULTS)

c: clean
clean: results $(CLEAN)


# Rules to run tests common to all projects
%.tst: %.exe
	@printf RUN\ $<...\ \  && $(RUN) ./$< && printf "PASS\n" && touch $@

%.mtst: %.exe
	@printf MPIRUN\ $<...\ \  && $(MPIRUN) ./$< && printf "PASS\n" && touch $@
