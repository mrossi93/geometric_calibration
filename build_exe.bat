CALL "%cd%\venv\Scripts\activate.bat"
pyinstaller entrypoint_exe.py --noconfirm --name geoCal ^
--add-data "%cd%\geometric_calibration\app_data\ref_brandis.txt;.\app_data"
xcopy "%cd%\geoCal.bat" "%cd%\dist\"
xcopy "%cd%\geoCal_config.ini" "%cd%\dist\"
PAUSE