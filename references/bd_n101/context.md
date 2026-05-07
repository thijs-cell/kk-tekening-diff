# BD-N101

## Project
Tekeningstijl met renvooi "BOUWKUNDIG"

## Relevante wandtypes (DETECTEREN)

Deze wandtypes moet de tool detecteren en classificeren:

### Metselwerk / steenachtig
- **100mm baksteen metselwerk**
- **65mm baksteen gekanteld dikformaat metselwerk**
- **HSB element 272mm**

### Kalkzandsteen
- **100mm kalkzandsteen**
- **150mm kalkzandsteen**
- **250mm kalkzandsteen**

### Isolatie
- **Isolatie div. diktes**
- **Hoogwaardige isolatie div. diktes**

### Gipsblokken
- **Gipsblokken 100mm**
- **Gipsblokken 70mm** (let op: zwaar 100mm staat ook in renvooi maar wordt hier samengevat onder gipsblokken)

### Metalstud
- **Metalstudwand 100mm**
- **Metalstudwand 150mm**
- **Metalstud voorzetwand 174mm**
- **Metalstud woningscheiden 255mm**

### Roosterwand
- **Roosterwand**

### Beton
- **Betonwand 175mm**
- **Betonwand 250mm**
- **Betonwand 350mm**
- **Prefab betonwand 175mm**
- **Prefab betonwand 250mm**

## Niet-relevante elementen (NEGEREN)

- Hemelwaterafvoer (HWA), Entree-symbool, Gevelrooster
- Brandveiligheid markeringen (rookmelder, brandblusser, etc.)
- Vloertegels, Mussenkast, Gierzwaluwkast
- Raam/glas symbolen (C, Vg, S)
- Binnendeur-symbolen
- Algemene tekstmarkeringen

## Belangrijke notitie voor Vision

Dit is de meest complexe tekening qua wandtypes — 19 verschillende relevante types. Onderscheid voornamelijk via:
- Lijndikte (massieve wanden vs lichte wanden)
- Arceringspatroon (gips vs metalstud vs beton)
- Kleur (beton donkerder dan gips)
