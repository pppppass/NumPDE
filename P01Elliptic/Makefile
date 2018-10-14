PGFOBJS = Figure01.pgf Figure02.pgf Figure03.pgf Figure04.pgf Figure05.pgf\
	Figure06.pgf Figure07.pgf Figure08.pgf Figure09.pgf Figure10.pgf\
	Figure11.pgf Figure12.pgf Figure13.pgf Figure14.pgf Figure15.pgf\
	Figure16.pgf Figure17.pgf Figure18.pgf 
TBLOBJS = Table1.tbl Table2.tbl Table3.tbl
TXTOBJS = Text01.txt Text02.txt Text03.txt Text04.txt Text05.txt Text06.txt\
	Text07.txt Text08.txt Text09.txt Text10.txt Text11.txt Text12.txt
RUNOBJS = ProblemPart1.run ProblemPart2.run ProblemPart3.run

.PHONY: all
all: Report.pdf

Report.pdf: Report.tex $(PGFOBJS) $(TBLOBJS) $(TXTOBJS) | Report.aux
	lualatex Report.tex

Report.aux: Report.tex
	lualatex Report.tex

.SECONDEXPANSION:
$(PGFOBJS) $(TBLOBJS) $(TXTOBJS): $(wildcard $(RUNOBJS)) $(if $(wildcard Result.dir), Plot.py)
	python Plot.py
	sed -i 's/<LABEL1~*>/Error $$ \\normi{e_h} $$/' Figure03.pgf
	sed -i 's/<LABEL2~*>/Estimated error $$ 4 \\normi{ U_h - U_{ h \/ 2 } } \/ 3 $$/' Figure04.pgf
	sed -i 's/<LABEL3~*>/Error $$ \\normi{ U_h^1 - u_h } $$/' Figure05.pgf
	sed -i 's/<LABEL4~*>/Error $$ \\normi{e_h} $$/' Figure09.pgf
	sed -i 's/<LABEL5~*>/Estimated error $$ 4 \\normi{ U_h - U_{ h \/ 2 } } \/ 3 $$/' Figure10.pgf
	sed -i 's/<LABEL6~*>/Error $$ \\normi{ U_h^1 - u_h } $$/' Figure12.pgf
	sed -i 's/<LABEL7~*>/Error $$ \\normi{e_h} $$/' Figure15.pgf
	sed -i 's/<LABEL8~*>/Estimated error $$ 4 \\normi{ U_h - U_{ h \/ 2 } } \/ 3 $$/' Figure16.pgf
	sed -i 's/<LABEL9~*>/Error $$ \\normi{ U_h^1 - u_h } $$/' Figure17.pgf

setup.cfg:
	if ![[ -v CONDA_PREFIX ]];\
	then\
		echo "Anaconda not found";\
		exit 1;\
	fi
	echo "[build_ext]" > setup.cfg
	echo "include_dirs=${CONDA_PREFIX}/include" >> setup.cfg
	echo "library_dirs=${CONDA_PREFIX}/lib" >> setup.cfg

exts.so: setup.py setup.cfg exts/algos.c exts/wrappers.c exts/exts.h
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
