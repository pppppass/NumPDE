PGFOBJS = Figure01.pgf Figure02.pgf Figure03.pgf Figure04.pgf Figure05.pgf Figure06.pgf Figure07.pgf Figure08.pgf Figure09.pgf Figure10.pgf Figure11.pgf Figure12.pgf
RUNOBJS = Problem1Part1.run Problem1Part2.run Problem1Part3.run Problem1Part4.run Problem2Part1.run Problem2Part2.run Problem2Part3.run Problem2Part4.run

.PHONY: all
all: Report.pdf

Report.pdf: Report.tex $(PGFOBJS) | Report.aux
	lualatex Report.tex

Report.aux: Report.tex
	lualatex Report.tex

$(PGFOBJS): $(wildcard $(RUNOBJS))
	touch $@

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py models.py solvers.py
	python $<
	touch $@

.PHONY: report
report: Report.pdf

.PHONY: run
run: $(RUNOBJS)
