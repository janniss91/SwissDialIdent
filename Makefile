VIRTUALENV_DIR=${PWD}/env
PIP=${VIRTUALENV_DIR}/bin/pip

all: virtualenv
	. ${VIRTUALENV_DIR}/bin/activate; \
	$(PIP) install -r requirements.txt

virtualenv:
	if [ ! -e ${PIP} ]; then \
	virtualenv -p python3 $(VIRTUALENV_DIR); \
	fi
	$(PIP) install --upgrade pip

clean:
	-rm -fv .DS_Store .coverage
	find . -name '*.pyc' -exec rm -fv {} \;
	find . -name '*.pyo' -exec rm -fv {} \;
	find . -depth -name '__pycache__' -exec rm -rfv {} \;

dist-clean: clean
	-rm -rfv ${VIRTUALENV_DIR} && \
	find . -depth -name '*.egg-info' -exec rm -rfv {} \;