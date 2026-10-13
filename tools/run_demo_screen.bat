@echo off
rem Screen-video demo of the event + RGB fusion detector (events are simulated from the video).
rem Usage: run_demo_screen.bat [--model af^|add^|iaf_add^|rgb^|owod] [--source screen^|video.mp4]
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\demo_screen.py %*
pause
