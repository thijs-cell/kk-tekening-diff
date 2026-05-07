# 56 de Helling

## Project
Twee renvooien aangeleverd: oud (renvooi_oud.png) en nieuw (renvooi_nieuw.png)

## Verschil tussen renvooien

**Nieuw item in renvooi_nieuw.png dat niet in renvooi_oud.png staat:**
- **Sandwichpaneel** — kruisarcering, gemarkeerd met rode pijl in nieuw renvooi

Dit betekent dat er waarschijnlijk sandwichpaneel-wanden zijn TOEGEVOEGD in de nieuwe tekening.

## Wandtypes (gecombineerde lijst van oud + nieuw)

### Steenachtig
- Kalkzandsteen 120mm
- Kalkzandsteen 100mm

### Gips
- Gibo 100mm
- Gibo zwaar 70mm
- Gibo 70mm

### HSB / Sandwich
- HSB-wand
- Sandwichpaneel (alleen in nieuw renvooi)

### Voorzetwanden / isolatie
- Voorzetwand: isolatie + biobased plaat + gips
- Isolatie + stuc
- PIR + OSB
- Hardschuimisolatie

### Beton
- Prefabbeton
- Beton

### Gevelafwerking
- Rhombus gevelafwerking
- Mato gevelafwerking

### Overige
- Achterwand toilet

## Niet-relevant voor wand-detectie (negeren)

- Peilmaat (1200+)
- Brandwerendheid markeringen (30 minuten, 60 minuten)
- Zelfsluitende deur
- Geluidsisolatiewaarde indicatoren (Rw,p > 36 dB, Rw > 40 dB)
- Entree pijl
- Vluchtwegaanduiding

## Belangrijke notitie voor Vision

Tekening heeft kleurcodering — wandtypes zijn visueel goed onderscheidbaar via arceringspatroon.

Op de pagina staan ook **rode markeringen** van de architect (revisiewolkjes/aantekeningen). Deze moeten genegeerd worden tijdens wand-detectie — het zijn geen wand-wijzigingen.
