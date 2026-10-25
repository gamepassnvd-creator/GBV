; Inno Setup script — optional single-installer packaging of the portable bundle.
; Build the portable folder first (packaging\build_windows.ps1), then:
;   1. install Inno Setup 6 (https://jrsoftware.org/isinfo.php)
;   2. compile this file with ISCC.exe or the Inno Studio GUI
;
; Output: dist\installer\GBVMonitor-setup.exe (disk-spanning: for bundles > 2 GB
; the output is split as GBVMonitor-setup-1.bin - simply keep ALL parts together).
; NOTICE: the user extracting via the installer gets the same folder-extraction
; experience as unzipping the portable zip — the app needs no registry entries.

[Setup]
AppName=GBV Monitor
AppVersion=1.0.0
AppCopyright=For civil-society / NGO use
DefaultDirName={sd}\GBVMonitor
DefaultGroupName=GBV Monitor
Compression=lzma2/max
DiskSpanning=yes
DiskSliceSize=2000000000
OutputDir=dist\installer
OutputBaseFilename=GBVMonitor-setup
Uninstallable=yes
UninstallFilesDir={app}\uninst
PrivilegesRequired=lowest

[Files]
Source: "dist\GBVMonitor\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\GBV Monitor"; Filename: "{app}\GBVMonitor.exe"
Name: "{userdesktop}\GBV Monitor"; Filename: "{app}\GBVMonitor.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Dirs]
Name: "{app}\output"; Permissions: users-modify
Name: "{app}\logs";   Permissions: users-modify

[UninstallDelete]
; keep the user's case dataset safe: only uninstall files, not output\
Type: files; Name: "{app}\uninst\*"
