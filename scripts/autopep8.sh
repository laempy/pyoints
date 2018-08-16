# requires python-autopep8
SCRIPT_PATH=$(dirname $(realpath -s $0))
cd $SCRIPT_PATH
autopep8 -r -i -a -a --experimental -v -v ../pyoints
autopep8 -r -i -a -a --experimental -v -v ../tests
