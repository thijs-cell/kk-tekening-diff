# 5102 — Eerste verdieping

## Project
Tekeningstijl met renvooi "RENVOOI MATERIALEN" — zwart-wit tekening met arceringspatronen

## Relevante wandtypes (DETECTEREN)

Deze wandtypes moet de tool detecteren en classificeren:

### Gevel
- **Gevelmetselwerk in banden, staand uitgevoerd**
- **Prefab betonnen gevelpaneel**

### Kalkzandsteen (let op: meerdere diktes per type)
- **Kalkzandsteen 67mm, 100mm, 120mm, 150mm** (21mm en 300mm varianten)
- **Kalkzandsteen CS36 214mm en 300mm**

### Metalstud
- **Metal stud wand 125mm**
- **Metal stud wand 205mm**
- **Metal stud voorzetwand 100mm**
- **Metal stud voorzetwand 75mm**

### Lichte binnenwanden (let op: 3 varianten met verschillende gips-types)
- **Lichte binnenwand 70mm en 100mm — gips normaal (GNL)**
- **Lichte binnenwand 70mm en 100mm — gehydrofobeerd, gips zwaar (GHL)**
- **Lichte binnenwand 70mm — gips zwaar (GZL)**

## Niet-relevante elementen (NEGEREN)

- Wandtegels (badkamer, toilet, keuken)
- HSB-element details (dampopen folie, dampremmende folie, gipsplaat)
- Renvooi algemeen: ingang, meterkast, geluidwerende deur, hemelwaterafvoer, noodoverstort
- Renvooi brandveiligheid: alle WBDBO markeringen, brandmelder, vluchtroute, mobiele brandblusser
- Algemene opmerkingen en symbolen

## Belangrijke notitie voor Vision

**Dit is een ZWART-WIT tekening zonder kleurcodering.** De vector-pipeline kan hier weinig mee — kleurmatching werkt niet. Vision moet onderscheid maken via:

- **Arceringspatroon** (primair onderscheid)
- **Lijndikte en spatiëring** van de arcering
- **Type vulling** (gestippeld vs gestreept vs solide)

De vector-pipeline gaf hier in eerdere tests "type onbekend" voor 80%+ van de detecties. Dit is precies de tekeningstijl waar Vision de grootste meerwaarde kan hebben — mits we het renvooi als referentie meegeven.

Onderscheidende kenmerken om op te letten:
- Gevelmetselwerk: kruisarcering met diagonale strepen
- Kalkzandsteen: enkele diagonale strepen
- Metalstud: stippellijn-patroon
- Lichte binnenwanden: gestreept met verschillende dichtheden voor GNL/GHL/GZL
