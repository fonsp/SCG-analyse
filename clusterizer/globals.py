# -*- coding: utf-8 -*-

from pathlib import Path


def subdirectories_of(path):
    return [item.resolve() for item in path.iterdir() if item.is_dir()]


current_path = Path().resolve()
git_path = current_path

# While we are not at the root of the git directory:
while '.git' not in map(lambda p: p.name, subdirectories_of(git_path)):
    # Move one directory up, and check again
    git_path = git_path.parent.resolve()
    if len(git_path.parts) <= 1:
        raise Warning("This script is not running in the git repository. Configure data path manually.")


data_path = git_path / "data" / "origineel"
print("Git root path found at: "+str(git_path))
print("Using data path:        " + str(data_path))

if not Path(data_path).is_dir():
    raise Warning("Data path does not exist")

#datadir = Path('C:/Data/datascg/')
datadir = data_path

datadumpsdir = datadir / 'datadumps'
datadumpdir = datadumpsdir / 'Datadump-server'
cache = datadumpsdir / 'cache'
circuitcache = cache / 'circuits'

# Lijst van .csv-bestanden
data_file_paths = list(data_path.glob("*.csv"))

if len(data_file_paths) == 0:
    raise Warning("No .csv files found in " + str(data_path) + "; Note that data files are not stored in this repository, and must be downloaded manually.")


# Geef het circuitnummer (4 cijfers) dat in de naam van een Path staat
circuitnum_from_file_path = lambda file_path: int(file_path.name.split('-')[0])

partial_discharges = {circuitnum_from_file_path(fp) for fp in data_file_paths if 'pd' in fp.name}

# Sla de nummers op van alle circuits waarvoor PD data bestaat.
available_circuits = list(partial_discharges)
