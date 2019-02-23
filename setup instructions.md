# Setup instructions

Clone this repository (available at [github.com/fons-/SCG-analyse](https://github.com/fons-/SCG-analyse)) to get started!

## Software

Voor het gebruiken van Github raad ik [Github Desktop](https://desktop.github.com/) aan, je hoeft eigenlijk nooit een terminal te gebruiken. Ook de website zelf is erg handig (maar niet om commits mee te maken). 

Om Python code te schrijven raad ik aan om Atom (met plugin 'Kite'), Visual Studio Code (met plugin 'Kite') of Visual Studio 2017 te gebruiken. 

Een cool systeem is Jupyter Notebook, waarbij je tekst, code en grafiekjes allemaal in 1 document krijgt. Hier onder staat uitgelegd hoe je dat aan de praat krijgt.

## Source code

Most relevant code resides in `src/` and in `notebooks/`.

`src/` contains general class definitions and methods, which will be used in Notebooks for analysis and visualisation. Notebooks can be viewed by running Jupyter Notebook, as explained below, or by viewing the files in your browser, at [github.com/fons-/grid-analysis/notebooks](https://github.com/fons-/grid-analysis/notebooks).

## Installing Python packages

(Er bestaan erg veel packages voor python, en elk project heeft een andere collectie packages nodig. Je zou alle packages van alles waar je aan werkt *globaal* kunnen installeren, en ervan uitgaan dat degene met wie je de code deelt die packages ook allemaal heeft. Een nettere manier is om een *virtual environment* te maken voor een project, dat is een soort schone python-installatie waarin je alleen de packages installeert die je nodig hebt. Je slaat dan een lijstje op met benodigde packages, `requirements.txt`, zodat anderen jouw virtual environment exact kunnen namaken (automatisch).)

Make sure Python 3 (tested on 3.7) is installed. When using Windows, Python 3 should be [added to your PATH](https://docs.python.org/3/using/windows.html#using-on-windows).

Open a terminal in the root of the repository. Let's create a virtual environment and install the required packages:

*Unix:*

```bash
python3 -m venv ./venv
source venv/bin/activate
# 'python' will now be mapped to python3
python -m pip install -r requirements.txt
```

*Windows:* (this can also be done using the Visual Studio GUI)

```dos
python -m venv .\venv
venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```



We can exit the virtual environment using the `deactivate` command (Unix & Windows).

## Jupyter Notebook

To use our virtual environment inside Jupyter Notebook, we need to install it as a *Kernel*. First activate the virtual environment, then run:

```
ipython kernel install --user --name=venv
```

### Running Jupyter Notebook

In the root directory, run:

```
jupyter notebook
```

Follow the instructions printed in the console to open Jupyter.

When you open a Python Notebook, you should now be able to select the virtual environment using Kernel > Change Kernel > venv.

## Visual Studio

Open `grid-analysis.sln` in Visual Studio 2017 to get started. Make sure that the virtual environment is listed under "Python Environments" in the Solution Explorer. If possible, Activate the environment and Open its Interactive Environment. When viewing a `.py` file, you can press Ctrl+Enter to run a code block or selection. 

