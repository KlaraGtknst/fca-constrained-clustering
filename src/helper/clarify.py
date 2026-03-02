import argparse
import sys

def read_cxt(filepath):
    """Liest eine Burmeister (.cxt) Datei und extrahiert die Bestandteile."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    
    if not lines or lines[0].strip() != 'B':
        raise ValueError("Ungültiges Format: Datei muss mit 'B' beginnen (Burmeister Format).")
    
    # Zeile 1: 'B' (lines[0])
    # Zeile 2: Name des Kontexts (lines[1]) - in der Regel leer
    name = lines[1] if len(lines) > 1 else ""
    
    # Zeile 3: Anzahl der Objekte (lines[2])
    num_objs = int(lines[2].strip())
    
    # Zeile 4: Anzahl der Attribute (lines[3])
    num_attrs = int(lines[3].strip())
    
    # Zeile 5: Zwingende Leerzeile (lines[4])
    
    # Ab Zeile 6 (Index 5) beginnen die Objekte
    idx = 5
    objs = lines[idx : idx + num_objs]
    idx += num_objs
    
    # Danach die Attribute
    attrs = lines[idx : idx + num_attrs]
    idx += num_attrs
    
    # Danach die Matrix
    matrix = lines[idx : idx + num_objs]
    
    if len(matrix) != num_objs:
        print(f"Warnung: {num_objs} Zeilen erwartet, aber {len(matrix)} gefunden.", file=sys.stderr)
        
    return name, objs, attrs, matrix

def clarify_objects(matrix):
    """
    Gruppiert identische Zeilen (Object Clarification). 
    Gibt die neuen Objekt-Labels (Anzahl) und die eindeutigen Matrix-Zeilen zurück.
    """
    unique_rows =[]
    row_counts = {}
    
    for row in matrix:
        if row not in row_counts:
            unique_rows.append(row)
            row_counts[row] = 0
        row_counts[row] += 1
        
    # Das neue Objekt-Label ist die Anzahl der ursprünglichen Objekte, die diese Zeile hatten
    new_objs = [str(row_counts[row]) for row in unique_rows]
    return new_objs, unique_rows

def write_cxt(filepath, name, objs, attrs, matrix):
    """Schreibt die bereinigten Daten strikt zurück ins Burmeister (.cxt) Format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        # Zeile 1
        f.write("B\n")
        # Zeile 2 (Name oder leer)
        f.write(f"{name}\n")
        # Zeile 3 & 4 (Anzahl)
        f.write(f"{len(objs)}\n")
        f.write(f"{len(attrs)}\n")
        # Zeile 5 (Die zwingende leere Zeile)
        f.write("\n")
        
        # Listen schreiben
        for obj in objs:
            f.write(f"{obj}\n")
            
        for attr in attrs:
            f.write(f"{attr}\n")
            
        for row in matrix:
            f.write(f"{row}\n")

def main():
    parser = argparse.ArgumentParser(description="Object-Clarification eines formalen Kontexts im Burmeister-Format.")
    parser.add_argument("input_file", help="Pfad zur .cxt Eingabedatei")
    parser.add_argument("output_file", help="Pfad zur .cxt Ausgabedatei")
    
    args = parser.parse_args()
    
    try:
        # 1. Ursprünglichen Kontext parsen
        name, objs, attrs, matrix = read_cxt(args.input_file)
        
        # 2. Objekte "clarifyen"
        new_objs, new_matrix = clarify_objects(matrix)
        
        # 3. Output schreiben
        write_cxt(args.output_file, name, new_objs, attrs, new_matrix)
        
        print("✅ Clarification erfolgreich ausgeführt!")
        print(f"   Ursprüngliche Anzahl Objekte: {len(objs)}")
        print(f"   Neue Anzahl Objekte:          {len(new_objs)}")
        print(f"   Attribute (unverändert):      {len(attrs)}")
        
    except Exception as e:
        print(f"❌ Fehler: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
