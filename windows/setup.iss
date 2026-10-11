[Setup]
AppId={{7C98F49C-3AC0-448D-90A8-F9037C2FC2BB}
AppName=DeeL Pyess
AppVersion=0.1.4
DefaultDirName={localappdata}\Programs\DeeL Pyess
DefaultGroupName=DeeL Pyess
PrivilegesRequired=lowest
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\artifacts\release
OutputBaseFilename=DeeL-Pyess-Setup-v0.1.4-Windows-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\DeeL Pyess.exe
CloseApplications=yes

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\build\windows\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DeeL Pyess"; Filename: "{app}\DeeL Pyess.exe"
Name: "{autodesktop}\DeeL Pyess"; Filename: "{app}\DeeL Pyess.exe"

[Run]
Filename: "{app}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "DeeL Pyess の専用画面を準備しています..."; Flags: waituntilterminated
Filename: "{app}\DeeL Pyess.exe"; Description: "DeeL Pyess を起動する"; Flags: nowait postinstall skipifsilent
