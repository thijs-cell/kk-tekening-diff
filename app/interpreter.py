"""
Interpreter voor K&K diff resultaten.

Vertaalt ruwe diff-data naar een gestructureerd, leesbaar rapport
gericht op wat Richard nodig heeft:
  1. Ruimtenaam wijzigingen (andere eisen mogelijk)
  2. Maatvoering binnenwanden
  3. Wanddikte wijzigingen (70/100mm)
  4. Wand indeling: wanden weg, veranderd, bijgekomen
"""

import re
from collections import defaultdict

from .layout_detect import PageLayout


# ---------------------------------------------------------------------------
# Constanten
# ---------------------------------------------------------------------------

WANDDIKTES = {50, 70, 100, 120, 150, 200}

_RE_RUIMTE = re.compile(
    r"(keuken|badkamer|toilet|hal(?:letje)?|gang|slaapkamer|woonkamer|berging|"
    r"technische ruimte|techn\.|meterkast|meterruimte|bijkeuken|entree|overloop|"
    r"balkon|terras|tuin|garage|wasruimte|cv[- ]?ruimte|hydrofoor|"
    r"at(?:elier)?[\./]\w*|buitenberging|stookruimte|trappenhuis|werkruimte|lift)",
    re.IGNORECASE,
)

_RE_MAAT_MM = re.compile(r"^\d{2,5}$")
_RE_OPP = re.compile(r"\d+[.,]\d+\s*m[²2]")
_RE_MERK = re.compile(r"merk\s+\w+", re.IGNORECASE)
_RE_WANDTYPE = re.compile(
    r"(gibo|hsb|sandwichpaneel|prefab|kalkzandsteen|beton|isolatie|"
    r"voorzetwand|gips|stuc|PIR|OSB|gevel|rhombus|hardschuim|"
    r"achterwand|biobased)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_wanddikte(tekst: str) -> bool:
    stripped = tekst.strip()
    if " " in stripped:
        return False
    try:
        return int(stripped) in WANDDIKTES
    except ValueError:
        return False


def _is_wanddikte_paar(oud: str, nieuw: str) -> bool:
    return _is_wanddikte(oud) and _is_wanddikte(nieuw)


def _is_maatvoering(tekst: str) -> bool:
    return bool(_RE_MAAT_MM.match(tekst.strip()))


def _is_ruimtelabel(tekst: str) -> bool:
    return bool(_RE_RUIMTE.search(tekst))


def _is_oppervlakte(tekst: str) -> bool:
    return bool(_RE_OPP.search(tekst))


def _is_wandtype(tekst: str) -> bool:
    return bool(_RE_WANDTYPE.search(tekst))


def _afstand(p1: list | tuple, p2: list | tuple) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _in_legenda(pos: list | tuple, pagina_breedte: float,
                layout: PageLayout | None = None) -> bool:
    """Check of positie in de legenda-zone zit.

    Gebruikt layout auto-detectie als beschikbaar, anders fallback
    op ratio-gebaseerde check (0.88 * breedte).
    """
    if layout is not None:
        return layout.is_in_legenda(pos)
    if pagina_breedte <= 0:
        return False
    return pos[0] / pagina_breedte > 0.88


# ---------------------------------------------------------------------------
# Context ophalen: wat staat er in de buurt van een wijziging?
# ---------------------------------------------------------------------------

def _zoek_context(
    pos: list | tuple,
    alle_tekst: list[dict],
    max_afstand: float = 120.0,
    eigen_tekst: str = "",
) -> dict:
    """
    Zoek omliggende tekst bij een positie.
    Returns dict met: ruimte, merk, wandtype, extra_context
    """
    buren = []
    for t in alle_tekst:
        d = _afstand(pos, t["pos"])
        if d < max_afstand and t["tekst"] != eigen_tekst:
            buren.append((d, t["tekst"]))
    buren.sort()

    ruimte = None
    merk = None
    wandtype = None
    extra = []

    for d, tekst in buren:
        if ruimte is None and _is_ruimtelabel(tekst):
            ruimte = tekst
        elif merk is None and _RE_MERK.search(tekst):
            merk = tekst.strip()
        elif wandtype is None and _is_wandtype(tekst):
            wandtype = tekst.strip()
        elif len(extra) < 2 and len(tekst) > 2 and not _is_maatvoering(tekst):
            extra.append(tekst.strip())

    return {
        "ruimte": ruimte,
        "merk": merk,
        "wandtype": wandtype,
        "extra": extra,
    }


def _beschrijf_locatie(ctx: dict) -> str:
    """Maak een leesbare locatiebeschrijving uit context."""
    delen = []
    if ctx.get("ruimte"):
        delen.append(ctx["ruimte"])
    if ctx.get("merk"):
        delen.append(ctx["merk"])
    return " - ".join(delen) if delen else ""


# ---------------------------------------------------------------------------
# Hoofdfunctie
# ---------------------------------------------------------------------------

def interpreteer_diff(diff_result: dict, alle_tekst: list[dict] | None = None,
                      pagina_breedte: float = 0,
                      layout: PageLayout | None = None) -> dict:
    """
    Vertaal ruwe diff-data naar een gestructureerd rapport.

    Parameters:
        diff_result: output van run_diff()
        alle_tekst: lijst van alle tekst-items op de NIEUWE pagina (voor context)
        pagina_breedte: breedte van de pagina in punten (fallback voor legenda)
        layout: auto-gedetecteerde layout (als beschikbaar)
    """
    # Layout ophalen uit diff_result als niet meegegeven
    if layout is None:
        layout = diff_result.get("_layout_obj")

    # Als geen tekst meegegeven, gebruik wat we hebben uit diff
    if alle_tekst is None:
        alle_tekst = _verzamel_alle_tekst_uit_diff(diff_result)

    ruimtenaam = []
    wanddikte = []
    maatvoering = []
    oppervlakte = []
    legenda = []
    bouwkundig = []
    overig = []

    for item in diff_result.get("tekst_gewijzigd", []):
        oud = item.get("oud_tekst", "")
        nieuw = item.get("nieuw_tekst", "")
        pos = item.get("nieuw_pos", item.get("oud_pos", [0, 0]))
        cat = item.get("categorie", "overig")

        # Legenda-items apart houden
        if _in_legenda(pos, pagina_breedte, layout):
            if _is_wandtype(oud) or _is_wandtype(nieuw):
                legenda.append({
                    "oud": oud, "nieuw": nieuw, "pos": pos,
                    "type": "legenda_wandopbouw",
                })
            else:
                legenda.append({
                    "oud": oud, "nieuw": nieuw, "pos": pos,
                    "type": "legenda_overig",
                })
            continue

        ctx = _zoek_context(pos, alle_tekst, eigen_tekst=nieuw)
        locatie = _beschrijf_locatie(ctx)

        # 1. Ruimtenaam wijziging
        if _is_ruimtelabel(oud) and _is_ruimtelabel(nieuw) and oud.lower() != nieuw.lower():
            ruimtenaam.append({
                "oud_naam": oud,
                "nieuw_naam": nieuw,
                "pos": pos,
                "waarschuwing": "Naamswijziging kan andere eisen met zich meebrengen!",
            })
            continue

        # Skip mismatches (ruimtelabel vs iets anders = slechte match)
        if (_is_ruimtelabel(oud)) != (_is_ruimtelabel(nieuw)):
            overig.append({
                "oud": oud, "nieuw": nieuw, "pos": pos,
                "locatie": locatie, "type": "label_mismatch",
            })
            continue

        # 2. Wanddikte wijziging
        if _is_wanddikte_paar(oud, nieuw):
            wanddikte.append({
                "oud_dikte": oud,
                "nieuw_dikte": nieuw,
                "pos": pos,
                "locatie": locatie,
                "beschrijving": f"Wanddikte {oud}mm -> {nieuw}mm"
                                + (f" ({locatie})" if locatie else ""),
            })
            continue

        # 3. Oppervlakte wijziging
        if _is_oppervlakte(oud) or _is_oppervlakte(nieuw):
            oppervlakte.append({
                "oud": oud, "nieuw": nieuw, "pos": pos,
                "locatie": locatie,
                "beschrijving": f"Oppervlakte {oud} -> {nieuw}"
                                + (f" ({locatie})" if locatie else ""),
            })
            continue

        # 4. Maatvoering wijziging
        if _is_maatvoering(oud) and _is_maatvoering(nieuw):
            verschil = ""
            try:
                delta = int(nieuw) - int(oud)
                if delta != 0:
                    verschil = f" ({'+' if delta > 0 else ''}{delta}mm)"
            except ValueError:
                pass
            maatvoering.append({
                "oud_maat": oud, "nieuw_maat": nieuw, "pos": pos,
                "locatie": locatie,
                "beschrijving": f"{oud} -> {nieuw}mm{verschil}"
                                + (f" - {locatie}" if locatie else ""),
            })
            continue

        # 5. Bouwkundige wijziging (wandtype tekst)
        if _is_wandtype(oud) or _is_wandtype(nieuw):
            bouwkundig.append({
                "oud": oud, "nieuw": nieuw, "pos": pos,
                "locatie": locatie,
                "beschrijving": f"'{oud}' -> '{nieuw}'"
                                + (f" ({locatie})" if locatie else ""),
            })
            continue

        # 6. Overig
        overig.append({
            "oud": oud, "nieuw": nieuw, "pos": pos,
            "locatie": locatie, "type": cat,
        })

    # --- Ruimtenaam uit toegevoegd/verdwenen koppelen ---
    ruimtenaam += _koppel_ruimtenaam_uit_toeg_verdw(diff_result)

    # --- Scope/demarcatie vertalen ---
    scope = _interpreteer_scope(diff_result, alle_tekst, layout)

    # --- Wand indeling ---
    wand_indeling = _analyseer_wand_indeling(diff_result)

    # --- Nieuwe wanden via tekst ---
    nieuwe_wanden_tekst = _detecteer_wanden_via_nieuwe_tekst(
        diff_result, alle_tekst, pagina_breedte, layout,
    )

    # --- Deduplicatie ---
    ruimtenaam = _dedup_ruimtenaam(ruimtenaam)

    # --- Samenvatting ---
    samenvatting = _genereer_samenvatting(
        ruimtenaam, wanddikte, maatvoering, oppervlakte,
        bouwkundig, scope, wand_indeling, legenda, overig,
        nieuwe_wanden_tekst,
    )

    return {
        "ruimtenaam_wijzigingen": ruimtenaam,
        "wanddikte_wijzigingen": wanddikte,
        "maatvoering_wijzigingen": maatvoering,
        "oppervlakte_wijzigingen": oppervlakte,
        "bouwkundige_wijzigingen": bouwkundig,
        "scope_wijzigingen": scope,
        "wand_indeling": wand_indeling,
        "nieuwe_wanden_tekst": nieuwe_wanden_tekst,
        "legenda_wijzigingen": legenda,
        "overige_wijzigingen": overig,
        "samenvatting_tekst": samenvatting,
    }


# ---------------------------------------------------------------------------
# Hulpfuncties voor interpreteer_diff
# ---------------------------------------------------------------------------

def _verzamel_alle_tekst_uit_diff(diff_result: dict) -> list[dict]:
    """Verzamel alle bekende tekst-items als fallback voor context."""
    items = []
    for item in diff_result.get("tekst_gewijzigd", []):
        items.append({"tekst": item.get("nieuw_tekst", ""),
                       "pos": item.get("nieuw_pos", [0, 0])})
    for item in diff_result.get("tekst_toegevoegd", []):
        items.append({"tekst": item.get("tekst", ""),
                       "pos": item.get("pos", [0, 0])})
    for item in diff_result.get("tekst_verdwenen", []):
        items.append({"tekst": item.get("tekst", ""),
                       "pos": item.get("pos", [0, 0])})
    return items


def _koppel_ruimtenaam_uit_toeg_verdw(diff_result: dict) -> list[dict]:
    """Koppel verdwenen+toegevoegde ruimtelabels in dezelfde buurt als naamswijziging."""
    verdwenen = [
        {"tekst": i["tekst"], "pos": i["pos"]}
        for i in diff_result.get("tekst_verdwenen", [])
        if _is_ruimtelabel(i.get("tekst", ""))
    ]
    toegevoegd = [
        {"tekst": i["tekst"], "pos": i["pos"]}
        for i in diff_result.get("tekst_toegevoegd", [])
        if _is_ruimtelabel(i.get("tekst", ""))
    ]

    result = []
    matched_t = set()
    for vl in verdwenen:
        for ti, tl in enumerate(toegevoegd):
            if ti in matched_t:
                continue
            if _afstand(vl["pos"], tl["pos"]) < 100:
                if vl["tekst"].lower() != tl["tekst"].lower():
                    result.append({
                        "oud_naam": vl["tekst"],
                        "nieuw_naam": tl["tekst"],
                        "pos": tl["pos"],
                        "waarschuwing": "Naamswijziging kan andere eisen met zich meebrengen!",
                    })
                    matched_t.add(ti)
                    break
    return result


def _kleur_verschil_klein(oud: str, nieuw: str) -> bool:
    """Check of kleurverschil minimaal is (ruis)."""
    import re
    if oud == nieuw:
        return True
    m_oud = re.match(r"grijs\((\d+)%\)", oud)
    m_nieuw = re.match(r"grijs\((\d+)%\)", nieuw)
    if m_oud and m_nieuw:
        return abs(int(m_oud.group(1)) - int(m_nieuw.group(1))) < 15
    if "zwart" in oud and "zwart" in nieuw:
        return True
    return False


def _interpreteer_scope(diff_result: dict, alle_tekst: list[dict],
                        layout: PageLayout | None = None) -> list[dict]:
    """Vertaal vulkleur-wijzigingen naar leesbare scope-beschrijvingen.
    Gebruikt wandtype-namen uit de legenda waar beschikbaar."""
    result = []
    pw = diff_result.get("meta", {}).get("pagina_breedte", 0)

    for item in diff_result.get("vul_kleur_gewijzigd", []):
        pos = item.get("pos", [0, 0])

        # Skip legenda zone
        if _in_legenda(pos, pw, layout):
            continue

        # Skip kleine vlakjes (ruis)
        opp = item.get("oppervlakte", 0)
        if opp < 100:
            continue

        oud_naam = item.get("oud_naam", "")
        nieuw_naam = item.get("nieuw_naam", "")

        # Skip minimale kleurverschillen
        if _kleur_verschil_klein(oud_naam, nieuw_naam):
            continue

        # Gebruik wandtype-namen uit legenda als beschikbaar
        oud_wandtype = item.get("oud_wandtype")
        nieuw_wandtype = item.get("nieuw_wandtype")

        ctx = _zoek_context(pos, alle_tekst)
        locatie = _beschrijf_locatie(ctx)

        beschrijving = _beschrijf_wandtype_wijziging(
            oud_wandtype, nieuw_wandtype, oud_naam, nieuw_naam, locatie,
        )

        result.append({
            "pos": pos,
            "bbox": item.get("bbox", []),
            "locatie": locatie,
            "oud_wandtype": oud_wandtype,
            "nieuw_wandtype": nieuw_wandtype,
            "oud_kleur": oud_naam,
            "nieuw_kleur": nieuw_naam,
            "beschrijving": beschrijving,
        })
    return result


def _beschrijf_wandtype_wijziging(
    oud_wt: str | None, nieuw_wt: str | None,
    oud_kleur: str, nieuw_kleur: str, locatie: str,
) -> str:
    """Maak een leesbare beschrijving van een wandtype-wijziging."""
    oud_label = oud_wt or oud_kleur
    nieuw_label = nieuw_wt or nieuw_kleur

    oud_is_leeg = oud_wt is None and ("wit" in oud_kleur.lower())
    nieuw_is_leeg = nieuw_wt is None and ("wit" in nieuw_kleur.lower())

    if oud_is_leeg and nieuw_wt:
        basis = f"{nieuw_wt} toegevoegd"
    elif nieuw_is_leeg and oud_wt:
        basis = f"{oud_wt} verwijderd"
    elif oud_wt and nieuw_wt:
        basis = f"{oud_wt} \u2192 {nieuw_wt}"
    else:
        basis = f"{oud_label} \u2192 {nieuw_label}"

    return basis + (f" ({locatie})" if locatie else "")


def _vertaal_kleurwijziging(oud: str, nieuw: str, locatie: str) -> str:
    """Vertaal een kleurwijziging naar wat het waarschijnlijk betekent."""
    # wit -> gekleurd = wand komt erbij in scope
    # gekleurd -> wit = wand verdwijnt uit scope
    # kleur A -> kleur B = ander wandtype
    oud_is_wit = "wit" in oud.lower()
    nieuw_is_wit = "wit" in nieuw.lower()

    if oud_is_wit and not nieuw_is_wit:
        basis = "Wand toegevoegd aan scope (was ongemarkeerd)"
    elif not oud_is_wit and nieuw_is_wit:
        basis = "Wand uit scope gehaald (was gemarkeerd)"
    elif not oud_is_wit and not nieuw_is_wit:
        basis = f"Wandtype/scope gewijzigd ({oud} -> {nieuw})"
    else:
        basis = f"Markering gewijzigd ({oud} -> {nieuw})"

    return basis + (f" - {locatie}" if locatie else "")


def _analyseer_wand_indeling(diff_result: dict) -> dict:
    lijnen_toegevoegd = diff_result.get("lijnen_toegevoegd", 0)
    lijnen_verdwenen = diff_result.get("lijnen_verdwenen", 0)
    if not isinstance(lijnen_toegevoegd, int):
        lijnen_toegevoegd = 0
    if not isinstance(lijnen_verdwenen, int):
        lijnen_verdwenen = 0

    beoordeling = _beoordeel_wand_indeling(lijnen_toegevoegd, lijnen_verdwenen)

    # Nieuwe/verdwenen wanden uit wall_detect
    nieuwe_wanden = diff_result.get("nieuwe_wanden", [])
    verdwenen_wanden = diff_result.get("verdwenen_wanden", [])

    return {
        "lijnen_toegevoegd": lijnen_toegevoegd,
        "lijnen_verdwenen": lijnen_verdwenen,
        "nieuwe_wanden_count": len(nieuwe_wanden),
        "verdwenen_wanden_count": len(verdwenen_wanden),
        "beoordeling": beoordeling,
    }


def _beoordeel_wand_indeling(toegevoegd: int, verdwenen: int) -> str:
    if toegevoegd == 0 and verdwenen == 0:
        return ""

    delen = []
    if toegevoegd > 0:
        delen.append(f"{toegevoegd} nieuwe lijnelementen")
    if verdwenen > 0:
        delen.append(f"{verdwenen} verwijderde lijnelementen")

    if toegevoegd > 200 or verdwenen > 200:
        delen.append("LET OP: Significante wijziging in indeling!")

    return " | ".join(delen)


def _detecteer_wanden_via_nieuwe_tekst(
    diff_result: dict,
    alle_tekst: list[dict],
    pagina_breedte: float = 0,
    layout: PageLayout | None = None,
) -> list[dict]:
    """Detecteer nieuwe wanden via nieuw toegevoegde wanddikte-tekst.

    Als een tekst "70", "100", etc. nieuw verschijnt (tekst_toegevoegd),
    en het is geen wijziging van een bestaande maat, dan is dit
    waarschijnlijk een nieuwe wand met die dikte.

    Dedupliceer dichtbijzijnde teksten om dubbele meldingen te voorkomen.
    """
    result = []
    wanddikte_set = {50, 70, 100, 120, 150, 200}
    gezien_posities: list[tuple] = []

    for item in diff_result.get("tekst_toegevoegd", []):
        tekst = item.get("tekst", "").strip()
        pos = item.get("pos", [0, 0])

        # Skip legenda-zone
        if _in_legenda(pos, pagina_breedte, layout):
            continue

        # Moet een wanddikte-waarde zijn
        try:
            waarde = int(tekst)
        except ValueError:
            continue
        if waarde not in wanddikte_set:
            continue

        # Dedup: skip als er al een melding is binnen 60pt
        te_dichtbij = False
        for gp in gezien_posities:
            if _afstand(pos, gp) < 60:
                te_dichtbij = True
                break
        if te_dichtbij:
            continue

        gezien_posities.append(tuple(pos))

        ctx = _zoek_context(pos, alle_tekst, eigen_tekst=tekst)
        locatie = _beschrijf_locatie(ctx)

        result.append({
            "dikte_mm": waarde,
            "pos": pos,
            "locatie": locatie,
            "beschrijving": f"Nieuwe wand {waarde}mm"
                            + (f" (bij {locatie})" if locatie else ""),
        })

    return result


def _dedup_ruimtenaam(items: list) -> list:
    gezien = set()
    uniek = []
    for r in items:
        key = (r["oud_naam"].lower(), r["nieuw_naam"].lower())
        if key not in gezien:
            gezien.add(key)
            uniek.append(r)
    return uniek


# ---------------------------------------------------------------------------
# Samenvatting genereren
# ---------------------------------------------------------------------------

def _genereer_samenvatting(
    ruimtenaam, wanddikte, maatvoering, oppervlakte,
    bouwkundig, scope, wand_indeling, legenda, overig,
    nieuwe_wanden_tekst=None,
) -> list[str]:
    regels = []

    # Ruimtenaam (kritiek)
    if ruimtenaam:
        regels.append("=== RUIMTENAAM WIJZIGINGEN ===")
        for r in ruimtenaam:
            regels.append(f"  - '{r['oud_naam']}' -> '{r['nieuw_naam']}'")
            regels.append(f"    >> LET OP: Andere eisen mogelijk bij naamswijziging!")
        regels.append("")

    # Wanddikte (kritiek)
    if wanddikte:
        regels.append("=== WANDDIKTE WIJZIGINGEN ===")
        for w in wanddikte:
            regels.append(f"  - {w['beschrijving']}")
        regels.append("")

    # Maatvoering
    if maatvoering:
        regels.append(f"=== MAATVOERING WIJZIGINGEN ({len(maatvoering)}) ===")
        for m in maatvoering:
            regels.append(f"  - {m['beschrijving']}")
        regels.append("")

    # Oppervlakte
    if oppervlakte:
        regels.append(f"=== OPPERVLAKTE WIJZIGINGEN ({len(oppervlakte)}) ===")
        for o in oppervlakte:
            regels.append(f"  - {o['beschrijving']}")
        regels.append("")

    # Bouwkundig
    if bouwkundig:
        regels.append(f"=== BOUWKUNDIGE WIJZIGINGEN ({len(bouwkundig)}) ===")
        for b in bouwkundig:
            regels.append(f"  - {b['beschrijving']}")
        regels.append("")

    # Scope
    if scope:
        regels.append(f"=== SCOPE/DEMARCATIE WIJZIGINGEN ({len(scope)}) ===")
        for s in scope:
            regels.append(f"  - {s['beschrijving']}")
        regels.append("")

    # Nieuwe wanden via tekst
    if nieuwe_wanden_tekst:
        regels.append(f"=== NIEUWE WANDEN ({len(nieuwe_wanden_tekst)}) ===")
        for w in nieuwe_wanden_tekst:
            regels.append(f"  - {w['beschrijving']}")
        regels.append("")

    # Wand indeling
    beoordeling = wand_indeling.get("beoordeling", "")
    if beoordeling:
        regels.append("=== WAND INDELING ===")
        regels.append(f"  - {beoordeling}")
        regels.append("")

    # Legenda (compact, want dit is meestal ruis)
    if legenda:
        wandopbouw = [l for l in legenda if l["type"] == "legenda_wandopbouw"]
        if wandopbouw:
            regels.append(f"=== LEGENDA WANDOPBOUW ({len(wandopbouw)} wijzigingen) ===")
            regels.append(f"  Wandopbouw-tabel is gewijzigd. Controleer de legenda.")
            regels.append("")

    # Overig (heel compact)
    if overig:
        regels.append(f"=== OVERIGE WIJZIGINGEN ({len(overig)}) ===")
        for o in overig[:5]:
            loc = f" ({o['locatie']})" if o.get("locatie") else ""
            regels.append(f"  - '{o['oud']}' -> '{o['nieuw']}'{loc}")
        if len(overig) > 5:
            regels.append(f"  ... en {len(overig) - 5} meer")
        regels.append("")

    if not regels:
        regels.append("Geen significante wijzigingen gevonden.")

    return regels
