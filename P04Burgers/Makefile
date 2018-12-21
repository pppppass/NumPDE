PGFOBJS = Figure1.pgf Figure2.pgf Figure3.pgf Figure4.pgf Figure5.pgf Figure6.pgf Figure7.pgf Figure8.pgf Figure9.pgf
RUNOBJS = Problem1.run Problem2.run Problem3.run

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
