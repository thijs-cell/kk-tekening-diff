# Muiden D2.1 — WT-PLG-D2

## Project
Tekeningstijl met renvooi "legenda bouwkundig"

## Relevante wandtypes (DETECTEREN)

Deze wandtypes moet de tool detecteren en classificeren:

- **Kalkzandsteenwand CS12** — blokken/elementen gelijmd, groene diagonale arcering
- **Kalkzandsteenwand CS20** — blokken/elementen gelijmd, groene diagonale arcering
- **Kalkzandsteenwand CS36** — blokken/elementen gelijmd, groene kruisarcering
- **Gibo binnenwand** — normaal, d=100mm, gestreept patroon

## Niet-relevante elementen (NEGEREN)

Deze staan wel in het renvooi maar zijn geen wand-wijzigingen waar Dicky om geeft:

- Metselwerk type 1 (bruin, WF 210x100x51, wildverband)
- Metselwerk type 2 (witgrijs, ca. 215x102x52, wildverband)
- Isolatie generiek d=171mm
- Geluidsreducerende binnendeur-kozijn combinatie
- Overstroomrooster in binnendeur

## Belangrijke notitie voor Vision

CS12, CS20 en CS36 hebben dezelfde groene kleur — onderscheid is via arceringspatroon:
- CS12 en CS20: schuine streepjes
- CS36: kruisarcering (visueel distinct)

Dit is precies waar de vector-pipeline op vastloopt. Vision moet via arceringspatroon onderscheiden.
