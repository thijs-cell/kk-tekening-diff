# Ground truth — cluster 0 deel 4

## Tekening
56 de Helling, pagina 1, cluster 0 (zone met 3383 vector-wijzigingen),
deel 4 = rechter kwart van het cluster.

## Bestanden in deze map

- `cluster_0_deel4_oud.png` — render van oude tekening, dit gebied
- `cluster_0_deel4_nieuw_screenshot.png` — screenshot van nieuwe tekening
  met rode pijlen (D-markeringen) van architect, plus mijn eigen
  visuele aantekeningen waar wijzigingen zitten

## Door Thijs visueel geconstateerde wijzigingen

### Wijziging 1 — Gibo zwaar muur dikte-wijziging
- Type: **gewijzigd** (dikte/materiaal-variant)
- Oud: Gibo zwaar 70mm
- Nieuw: Gibo zwaar 100mm
- Locatie: linker zone van deel 4, rond ZIGGO/KPN ruimte (naast 0.04 Meterruimte)
- Visueel: muur is dikker geworden in de nieuwe tekening

### Wijziging 2 — Hardschuimisolatie toegevoegd
- Type: **toegevoegd**
- Wandtype: hardschuimisolatie
- Locatie: nieuw aanwezig in deel 4 (niet aanwezig in oude tekening)
- Visueel: dit wandtype staat in het renvooi maar werd in oude tekening
  niet gebruikt op deze locatie

## Doel van deze ground truth

Test of de sub-cluster pipeline (Vision-detectie op kleinere crops)
deze 2 wijzigingen kan vinden. Verwacht resultaat:
- Recall: 2/2 wijzigingen gevonden
- Precision: zo min mogelijk false positives op deze zone

Als Vision er minder dan 2 vindt: Vision detecteert te grof.
Als Vision er meer dan 2 unieke wijzigingen vindt: false positives,
mogelijk door rode D-markeringen of andere ruis.

## Kanttekening bij rode markeringen

In deze zone staan rode D-pijlen die door de architect zijn geplaatst
om wijzigingen aan te wijzen. Vision moet deze NEGEREN als
wand-wijziging — het zijn meta-aantekeningen, geen wanden.

De screenshot toont juist deze rode pijlen om te bewijzen dat de
architect zelf ook erkent dat hier wijzigingen zijn.
