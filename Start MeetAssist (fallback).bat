@echo off
rem Fallback launcher (use this if MeetAssist.exe is blocked by antivirus).
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%"
set "PYTHONNOUSERSITE=1"
start "" "%ROOT%runtime\python\pythonw.exe" -m meetassist.app
