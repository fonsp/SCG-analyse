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
