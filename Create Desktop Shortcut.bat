@echo off
rem Puts a MeetAssist shortcut on your Desktop.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\MeetAssist.lnk'); $s.TargetPath='%~dp0MeetAssist.exe'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0MeetAssist.exe,0'; $s.Save()"
echo Done - look for MeetAssist on your Desktop.
pause
