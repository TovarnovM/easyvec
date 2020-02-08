.PHONY: build dist redist install install-from-source clean uninstall

build:
	CYTHONIZE=1 ./setup.py build

dist:
	CYTHONIZE=1 ./setup.py sdist bdist_wheel

redist: clean dist

install:
	CYTHONIZE=1 pip install --user .

install-from-source: dist
	pip install --user dist/easyvec-0.0.1.tar.gz

clean:
	$(RM) -r build dist
	git clean -fdX

uninstall:
	pip uninstall easyvec
