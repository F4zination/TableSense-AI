
## Poetry Anleitung 

### Poetry installieren
- [Poetry](https://python-poetry.org/docs/#installation) 



### Virtuelle Umgebung aktivieren
Hinweis: Ab Poetry 2.0 ist `poetry shell` nicht mehr standardmäßig verfügbar. Dies muss manuell installiert werden: https://github.com/python-poetry/poetry-plugin-shell

Alternativ und mittlerweile empfohlen:
```bash
poetry env activate
```
Anschließend diese Ausgabe ausführen.

Oder kombiniert:
```bash
$(poetry env activate)
```
<br>
Mit exit verlässt man die virtuelle Umgebung.

<br>

### Abhängigkeiten installieren
   ```bash
   poetry install
   ```

### Weitere zukünftige Abhängikeiten hinzufügen
   ```bash
   poetry add desired_library
   ```
### Entferne Abhängigkeiten
   ```bash
   poetry remove <paketname>
   ```

<br>


### Anwendung ausführen
```bash
poetry run python your_script.py
```
Durch die Nutzung von Streamlit für die UI:
```bash
poetry run streamlit run smolagent.py
```
