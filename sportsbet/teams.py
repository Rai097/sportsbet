"""Map full team names (as used by The Odds API and ESPN) to nflverse abbreviations."""

FULL_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

ABBR_TO_FULL = {v: k for k, v in FULL_TO_ABBR.items()}

# Older abbreviations that appear in historical nflverse data.
ABBR_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}


def to_abbr(name: str) -> str:
    """Return the nflverse abbreviation for a full name or an aliased abbreviation."""
    if name in FULL_TO_ABBR:
        return FULL_TO_ABBR[name]
    if name in ABBR_ALIASES:
        return ABBR_ALIASES[name]
    if name in ABBR_TO_FULL:
        return name
    raise KeyError(f"Unknown team: {name!r}")
