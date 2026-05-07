"""
Testcases voor de inline Hungarian (Jonker-Volgenant) implementatie in wand_diff.py.
Doel: verifieer correctheid met handmatig-gevalideerde matrices voordat integratie start.
Run: python test_hungarian.py
"""

import math


# ---------------------------------------------------------------------------
# De te testen implementatie (zelfde code als straks in wand_diff.py)
# ---------------------------------------------------------------------------

def _hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """
    Minimale-cost matching via het Jonker-Volgenant-achtige pad-augmentatie
    algoritme (pure Python, geen scipy).

    Parameters
    ----------
    cost : N×M matrix (list of lists), N <= M aanbevolen (anders transponeer).

    Returns
    -------
    Lijst van (rij, kolom) paren — optimale toewijzing.
    Onmatched rijen (als N > M) worden weggelaten.
    """
    INF = float("inf")
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])

    # Werk altijd met N <= M (transponeer indien nodig)
    transposed = n > m
    if transposed:
        cost = [[cost[r][c] for r in range(n)] for c in range(m)]
        n, m = m, n

    # u[i] = potentiaal rijen, v[j] = potentiaal kolommen
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)   # p[j] = welke rij is gekoppeld aan kolom j (1-geïndexeerd, 0=vrij)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minval = [INF] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minval[j]:
                        minval[j] = cur
                        way[j] = j0
                    if minval[j] < delta:
                        delta = minval[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minval[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    pairs = []
    for j in range(1, m + 1):
        if p[j] != 0:
            ri, ci = p[j] - 1, j - 1
            if transposed:
                ri, ci = ci, ri
            pairs.append((ri, ci))
    return pairs


def _total_cost(cost, pairs):
    return sum(cost[r][c] for r, c in pairs)


# ---------------------------------------------------------------------------
# Testcases
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name, pairs, cost, expected_cost, expected_pairs=None):
    actual = _total_cost(cost, pairs)
    ok = math.isclose(actual, expected_cost, abs_tol=1e-9)
    if expected_pairs is not None:
        ok = ok and set(pairs) == set(expected_pairs)
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {name}: cost={actual:.4f} (verwacht {expected_cost:.4f})"
          + (f", pairs={pairs}" if not ok else ""))
    results.append(ok)


# --- TC1: 1×1 triviale matrix ---
cost1 = [[5.0]]
pairs1 = _hungarian(cost1)
check("TC1 1×1 triviaal", pairs1, cost1, 5.0, [(0, 0)])

# --- TC2: 2×2 identiteitsachtige matrix ---
cost2 = [
    [4.0, 1.0],
    [2.0, 3.0],
]
# Optimaal: (0,1)=1 + (1,0)=2 = 3
pairs2 = _hungarian(cost2)
check("TC2 2×2 optimaal is diagonaal-swap", pairs2, cost2, 3.0, [(0, 1), (1, 0)])

# --- TC3: 3×3 klassiek voorbeeldmatrix ---
cost3 = [
    [4.0, 2.0, 8.0],
    [2.0, 3.0, 7.0],
    [3.0, 5.0, 6.0],
]
# Optimaal: (0,1)=2 + (1,0)=2 + (2,2)=6 = 10
pairs3 = _hungarian(cost3)
check("TC3 3×3 klassiek", pairs3, cost3, 10.0, [(0, 1), (1, 0), (2, 2)])

# --- TC4: 3×3 met duidelijk diagonaal optimum ---
cost4 = [
    [1.0, 9.0, 9.0],
    [9.0, 2.0, 9.0],
    [9.0, 9.0, 3.0],
]
# Optimaal: diagonaal = 1+2+3 = 6
pairs4 = _hungarian(cost4)
check("TC4 3×3 diagonaal-dominantie", pairs4, cost4, 6.0, [(0, 0), (1, 1), (2, 2)])

# --- TC5: 3×3 gelijke kosten (meerdere optima) ---
cost5 = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
]
# Elke perfecte matching heeft cost 3
pairs5 = _hungarian(cost5)
check("TC5 3×3 gelijke kosten (cost=3)", pairs5, cost5, 3.0)

# --- TC6: Rechthoekig 2×3 (minder rijen dan kolommen) ---
cost6 = [
    [3.0, 1.0, 2.0],
    [2.0, 3.0, 1.0],
]
# Optimaal: (0,1)=1 + (1,2)=1 = 2
pairs6 = _hungarian(cost6)
check("TC6 2×3 rechthoekig", pairs6, cost6, 2.0, [(0, 1), (1, 2)])

# --- TC7: Rechthoekig 3×2 (meer rijen dan kolommen — transponeer intern) ---
cost7 = [
    [3.0, 2.0],
    [1.0, 3.0],
    [2.0, 1.0],
]
# Optimaal 2 van 3 rijen: (1,0)=1 + (2,1)=1 = 2
pairs7 = _hungarian(cost7)
actual7 = _total_cost(cost7, pairs7)
ok7 = math.isclose(actual7, 2.0, abs_tol=1e-9) and len(pairs7) == 2
tag7 = PASS if ok7 else FAIL
print(f"  [{tag7}] TC7 3×2 meer rijen dan kolommen: cost={actual7:.4f} (verwacht 2.0), pairs={pairs7}")
results.append(ok7)

# --- TC8: 4×4 wand-achtig scenario (centroid-afstanden) ---
# Vier oude segmenten, vier nieuwe segmenten. Afstanden in pt.
# Verwacht: elk segment matcht met zijn tegenhanger (diagonaal), cost = 5+8+3+6 = 22
cost8 = [
    [5.0, 50.0, 80.0, 90.0],
    [60.0, 8.0, 70.0, 85.0],
    [75.0, 65.0, 3.0, 55.0],
    [88.0, 72.0, 45.0, 6.0],
]
pairs8 = _hungarian(cost8)
check("TC8 4×4 wand-scenario (diagonaal optimaal)", pairs8, cost8, 22.0,
      [(0, 0), (1, 1), (2, 2), (3, 3)])

# --- TC9: Alle oneindig behalve één pad (degenerate) ---
INF = 1e12
cost9 = [
    [INF, INF, 7.0],
    [INF, 3.0, INF],
    [2.0, INF, INF],
]
# Enige geldige matching: (0,2)=7 + (1,1)=3 + (2,0)=2 = 12
pairs9 = _hungarian(cost9)
check("TC9 3×3 geforceerde matching", pairs9, cost9, 12.0, [(0, 2), (1, 1), (2, 0)])

# --- TC10: 5×5 willekeurige matrix, cost handmatig berekend ---
cost10 = [
    [12.0,  4.0,  7.0, 18.0,  3.0],
    [ 6.0, 11.0,  2.0,  9.0, 14.0],
    [15.0,  8.0, 13.0,  5.0, 10.0],
    [ 1.0, 16.0,  9.0,  7.0, 12.0],
    [11.0,  3.0, 15.0,  4.0,  8.0],
]
# Handmatig gevalideerd optimaal: (0,4)=3 + (1,2)=2 + (2,3)=5 + (3,0)=1 + (4,1)=3 = 14
pairs10 = _hungarian(cost10)
check("TC10 5×5 willekeurig (cost=14)", pairs10, cost10, 14.0,
      [(0, 4), (1, 2), (2, 3), (3, 0), (4, 1)])

# ---------------------------------------------------------------------------
# Samenvatting
# ---------------------------------------------------------------------------
print()
passed = sum(results)
total = len(results)
if passed == total:
    print(f"\033[92mAlle {total} tests geslaagd.\033[0m")
else:
    print(f"\033[91m{passed}/{total} tests geslaagd — STOP, repareer voordat je verder gaat.\033[0m")
    raise SystemExit(1)
