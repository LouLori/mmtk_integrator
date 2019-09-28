# mmtk_integrator
develop new PIMC/PIMD integrator for MMTK user

## requirement and pre-install
This is a MMTK (Molecular Modelling Toolkit) integrator extension, so you have to install MMTK and its related package before using.

**MMTK installation script***

```
git clone https://github.com/roygroup/mmtk_install.git
```

then run the installation script:

```
./mmtk_install.sh
```

then alias the developer python of MMTK and set up the library path in your shell, edit the *.bash_profile* at home directory:

```
alias pydev='"$HOME/.mmtk/bin/python" $*'
export MMTK_USE_CYTHON=1
export LD_LIBRARY_PATH="$HOME/.mmtk/lib"
```

## mmtk integrator install and compilation

Edit *setup.py* file with your own username in all line with home directory path:

```
/home/username/.mmtk/...
```

then compiling with the following command:

```
pydev setup.py build_ext --inplace
```

