PGJOBJS = Figure01.pgf Figure02.pgf Figure03.pgf Figure04.pgf Figure051.pgf Figure052.pgf Figure05.pgf Figure061.pgf Figure062.pgf Figure071.pgf Figure072.pgf Figure081.pgf Figure082.pgf Figure09.pgf Figure10.pgf Figure11.pgf Figure12.pgf Figure13.pgf Figure14.pgf Figure15.pgf Figure16.pgf Figure17.pgf Figure18.pgf Figure19.pgf Figure20.pgf Figure21.pgf
TBLOBJS = Table11.tbl Table12.tbl Table13.tbl Table21.tbl Table22.tbl Table23.tbl
RUNOBJS = Problem0.run Problem1Part1.run Problem1Part2.run Problem2Part1.run Problem2Part2.run Problem2Part3.run Problem3Part1.run Problem3Part2.run Problem3Part3.run

.PHONY: all
all: Report.pdf

Report.pdf: Report.tex $(PGFOBJS) $(TBLOBJS) | Report.aux
	lualatex Report.tex

Report.aux: Report.tex
	lualatex Report.tex

.SECONDEXPANSION:
$(PGFOBJS) $(TBLOBJS) : $(wildcard $(RUNOBJS)) $(if $(wildcard Result.dir), Plot.py)
	python Plot.py

setup.cfg:
	if ![[ -v CONDA_PREFIX ]];\
	then\
		echo "Anaconda not found";\
		exit 1;\
	fi
	echo "[build_ext]" > setup.cfg
	echo "include_dirs=${CONDA_PREFIX}/include" >> setup.cfg
	echo "library_dirs=${CONDA_PREFIX}/lib" >> setup.cfg

exts.so: setup.py setup.cfg exts/theta.c exts/wrappers.c exts/exts.h
	python setup.py build_ext --inplace
	FILENAMES=`ls | grep 'exts.*\.so' | grep -v 'exts.so'`;\
	for FILE in $${FILENAMES};\
	do\
		mv $${FILE} exts.so;\
	done

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py exts.so
	python $<
	touch $@

.PHONY: report
report: Report.pdf

.PHONY: run
run: $(RUNOBJS)
