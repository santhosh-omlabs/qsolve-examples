"""Small built-in geometry library (Angstrom). Any molecule can instead be passed as XYZ text."""

LIBRARY: dict[str, str] = {
    "H2": "H 0 0 0\nH 0 0 0.74",
    "LiH": "Li 0 0 0\nH 0 0 1.595",
    "HF": "H 0 0 0\nF 0 0 0.917",
    "N2": "N 0 0 0\nN 0 0 1.098",
    "CO": "C 0 0 0\nO 0 0 1.128",
    "BeH2": "Be 0 0 0\nH 0 0 1.33\nH 0 0 -1.33",
    "H2O": "O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692",
    "NH3": "N 0 0 0.1165\nH 0 0.9377 -0.2718\nH 0.8121 -0.4689 -0.2718\nH -0.8121 -0.4689 -0.2718",
    "CH4": "C 0 0 0\nH 0.6288 0.6288 0.6288\nH -0.6288 -0.6288 0.6288\n"
    "H -0.6288 0.6288 -0.6288\nH 0.6288 -0.6288 -0.6288",
    "Li+": "Li 0 0 0",
    "Na+": "Na 0 0 0",
}
CHARGES: dict[str, int] = {"Li+": 1, "Na+": 1}
