.PHONY: all
all: Answer.pdf exts.so setup.cfg

Answer.pdf: Answer.tex | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex
	lualatex Answer.tex


.PHONY: recursive
recursive: template
	for DIR in $(DIRS);\
	do\
		$(MAKE) -C $${DIR};\
	done

setup.cfg:
	if ![[ -v CONDA_PREFIX ]];\
	then\
		echo "Anaconda not found";\
		exit 1;\
	fi
	echo "[build_ext]" > setup.cfg
	echo "include_dirs=${CONDA_PREFIX}/include" >> setup.cfg
	echo "library_dirs=${CONDA_PREFIX}/lib" >> setup.cfg

exts.so: setup.py exts/algos.c exts/wrappers.c exts/exts.h
	python setup.py build_ext --inplace
	FILENAMES=`ls | grep 'exts.*\.so' | grep -v 'exts.so'`;\
	for FILE in $${FILENAMES};\
	do\
		mv $${FILE} exts.so;\
	done

.PHONY: report
report: Answer.pdf
