.PHONY: all
all: exts.so

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
