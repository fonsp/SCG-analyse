#!/bin/bash

if (python -m pip freeze 2> /dev/null | grep -i PIL > /dev/null); then
  python generatePDfromimage.py > 99999999-pd.csv
  echo "Klaar!"
else
  echo "Je hebt de Python package 'Pillow' nodig."
fi
