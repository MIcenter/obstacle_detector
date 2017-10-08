setup: setup.py
	python3 setup.py

init: requirements.txt
	pip3 install -r requirements.txt

test: video_test.py
	echo 'TODO TESTS'
	python3 video_test.py
