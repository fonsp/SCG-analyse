# Om een plaatje om te zetten in een PD-bestand

### Stap 1

Zet een foto in deze map met de naam `vragen.png`. Zwart -> puntjes, wit -> niks.

### Stap 2

In bash:

```bash
chmod +x makePD.sh
./makePD.sh
```

In Windows:

```
python alliander.py > 99999999-pd.csv
```

### Stap 3
Zet het zojuist gegenereerde bestand `99999999-pd.csv` in de map `../data/origineel`. Maak in die map ook een kopie van `3010-cableconfig.csv`, en noem deze `99999999-cableconfig.csv`.

### Stap 4
Je kunt nu Circuit '99999999' importeren en clusteriseren zoals elk ander circuit. Let erop dat de standaardparameters van de cluster-algoritmen misschien niet geschikt zijn voor dit nep-circuit. 

---

# Om DBSCAN op een eigen plaatje los te laten

### Stap 1

Zet een foto in deze map met de naam `alliander.png`. Zwart -> puntjes, wit -> niks.

### Stap 2

In bash:

```bash
chmod +x makeJS.sh
./makeJS.sh
```

In Windows:

```
python generateJSfromimage.py > alliander.js
```

### Stap 3

Ga naar [https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) en open de developer console. Plak hierin de code uit het zojuist gegenereerde bestand `alliander.js`.

### Stap 4

Klik op de vorm _Smiley_ (het wordt niet een smiley maar jouw eigen plaatje), en kies geschikte waardes voor epsilon en MinPts.

### Stap 5

Gebruik een schermopnameprogramma om de `.gif` op te nemen.
