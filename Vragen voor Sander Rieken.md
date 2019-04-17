### Vragen over eindproduct:
1. Wat vind jij van parameters? (Dezelfde params voor elk circuit, of parameters fitten voor elk circuit?)
1. Runtime?
1. Wat is specifiek de input/output (datatypes) van wat we schrijven?
1. Moet het (uiteindelijk) realtime toepasbaar zijn?
1. Wat is een cluster? Moet het gevoeliger zijn dan de warnings gegeven door DNV GL?
1. Moeten clusters een compliceerde vorm (cirkel, ovaal) hebben of is een rechthoek (in afstand/tijd) goed genoeg?


### Vragen over data:
1. Wat gebeurt [hier](https://github.com/fons-/SCG-analyse/blob/403c256e7820539236bb504e87e4be7ca7adee08/notebooks/Clustering%20eerste%20poging%20-%20Fons.ipynb) of [hier](https://github.com/fons-/SCG-analyse/blob/fonzy/notebooks/Clustering%20eerste%20poging%20-%20Fons.ipynb)? Het valt op dat binnen dezelfde dataset op verschillende periodes het PD-gedrag (frequentie en pC-sterkte) sterk verschilt, of soms volledig mist. Misschien is a) de belasting totaal anders, omdat een circuit ergens anders in het netwerk wordt in-/uitgeschakeld; b) de SCG-apparatuur opnieuw gecalibreerd; c) de SCG-apparatuur verplaatst naar een ander circuit?
1. Waarom komen de clusters **hier TODO LINK** niet overeen met de moffen uit de cable config?
1. Wanneer wordt een warning gegeven?
1. Kunnen we meer circuits krijgen?
1. Klopt het dat de PD locaties niet continu zijn, maar al gediscretiseerd? (Elke ~25cm?)


### Wat is jouw mening over:
1. Het wel of niet meerekenen van PD lading? (TODO: Voorbeeld van het veschil)
1. Hoe verifieren we of iets echt een cluster is? Warnings gebruiken of _juist niet_?
