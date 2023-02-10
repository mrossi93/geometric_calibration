:: Define a virtual environment using Python 3.8.x 
python38 -u -m virtualenv venv

call "%cd%\venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements_dev.txt
pip install -e .
pip list

PAUSE