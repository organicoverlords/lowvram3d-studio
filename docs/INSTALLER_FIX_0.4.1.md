# Installer fix 0.4.1

The 0.4.0 installer passed an inline PowerShell array positionally to `UvPip`.
Windows PowerShell 5.1 bound that array as one string, causing uv to receive:

```text
install -r C:\...\requirements-control.txt
```

as a single subcommand. Version 0.4.1 uses a mandatory named `-Arguments`
array and constructs one flat uv argument vector:

```text
pip install --python C:\...\python.exe -r C:\...\requirements-control.txt
```

The install remains resumable; run `CONTINUE-INSTALL.cmd`.
