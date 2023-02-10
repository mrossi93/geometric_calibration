CALL "%cd%\venv\Scripts\activate.bat"
pyinstaller entrypoint_exe.py --name geoCal ^
--add-data "%cd%\geometric_calibration\app_data\ref_brandis.txt;.\app_data" ^
--add-data "%cd%\geoCal_config.ini;." ^
--add-data "%cd%\geoCal.bat;."
move %cd%\dist\geoCal\matplotlib\mpl-data %cd%\dist\geoCal\matplotlib
PAUSE