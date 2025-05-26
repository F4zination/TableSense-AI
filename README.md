# Tablesense-AI

![Tablesense-AI Logo](https://raw.githubusercontent.com/tablesense-ai/tablesense-ai/main/logo.png)


[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


## Übersicht

Tablesense-AI ist eine Python-Bibliothek zur intelligenten Verarbeitung und Analyse von tabellarischen Daten mithilfe von Large Language Models (LLMs). Die Bibliothek ermöglicht es, strukturierte Daten zu interpretieren, zu transformieren und natürlichsprachliche Antworten auf Basis von Tabellendaten zu generieren.

## Funktionen

- **Intelligente Datenanalyse**: Verarbeitung und Analyse von tabellarischen Daten
- **Natürlichsprachliche Schnittstelle**: Interaktion mit Tabellendaten über natürliche Sprache
- **Flexible Agentenarchitektur**: Erweiterbare Basis für verschiedene Anwendungsfälle
- **Templating-System**: Umwandlung von strukturierten Daten in natürlichsprachliche Beschreibungen


## Konfiguration

Erstellen Sie eine `.env`-Datei im Hauptverzeichnis Ihres Projekts:

```dotenv
API_KEY="Ihr-API-Schlüssel"
API_BASE="Ihre-API-Basis-URL"
```

## Verwendung

### Einfacher Agent

```python
from tablesense_ai.agent import SimpleAgent
import pandas as pd

# Agent initialisieren
agent = SimpleAgent(
    llm_model="model-name",
    temperature=0.7,
    max_retries=3,
    max_tokens=1000,
    base_url="http://example.com/api",
    api_key="your-api-key",
    system_prompt="Analysiere die folgenden Tabellendaten:"
)

# Daten laden
dataframe = pd.read_csv("ihre_daten.csv")

# Anfrage stellen
ergebnis = agent.eval(
    question="Wie viele Einträge haben einen Wert über 100?",
    dataset=dataframe
)

print(ergebnis)
```

## Projektstruktur

```
tablesense-ai/
├── tablesense_ai/
│   ├── agent/
│   │   ├── base.py          # Basisklasse für Agenten
│   │   ├── serialization/   # Datenserialisierer
│   │   │   └── jinja_templates/  # Templates zur Textgenerierung
├── tests/                   # Testfälle
├── .env                     # Umgebungsvariablen
└── README.md                # Diese Datei
```

## Anforderungen

- Python 3.8+
- pandas
- Jinja2
- python-dotenv

## Lizenz

MIT