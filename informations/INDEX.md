# Romanian Cities Information Package

## Overview
- Total cities: 315
- Successfully downloaded: 315
- Failed: 0

## Directory Structure
```
informations/
├── __init__.py           # Package initialization
├── cities/               # HTML files for each city
│   ├── bucuresti.html
│   ├── cluj-napoca.html
│   └── ...
├── metadata.json         # Download statistics
└── INDEX.md             # This file
```

## Usage
```python
from pathlib import Path

# Load a city's HTML
cities_dir = Path('informations/cities')
bucharest_html = (cities_dir / 'bucuresti.html').read_text(encoding='utf-8')

# Parse with BeautifulSoup
from bs4 import BeautifulSoup
soup = BeautifulSoup(bucharest_html, 'html.parser')
```

## Files Downloaded
- București: `cities/bucurești.html`
- Cluj-Napoca: `cities/cluj-napoca.html`
- Iași: `cities/iași.html`
- Constanța: `cities/constanța.html`
- Timișoara: `cities/timișoara.html`
- Brașov: `cities/brașov.html`
- Craiova: `cities/craiova.html`
- Galați: `cities/galați.html`
- Oradea: `cities/oradea.html`
- Ploiești: `cities/ploiești.html`
... and 305 more cities
