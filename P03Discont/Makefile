PGFOBJS = Figure01.pgf Figure02.pgf Figure03.pgf Figure04.pgf Figure05.pgf Figure06.pgf Figure07.pgf Figure08.pgf Figure09.pgf Figure10.pgf Figure11.pgf Figure12.pgf Figure13.pgf Figure14.pgf Figure15.pgf Figure16.pgf Figure17.pgf Figure18.pgf Figure19.pgf Figure20.pgf Figure21.pgf Figure22.pgf Figure23.pgf Figure24.pgf
TBLOBJS = Table1.tbl Table2.tbl Table3.tbl
RUNOBJS = Problem0.run Problem2Part1.run Problem2Part2.run Problem2Part3.run Problem2Part4.run Problem2Part5.run Problem2Part6.run

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
