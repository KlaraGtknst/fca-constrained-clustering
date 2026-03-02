import sys
import os
from pathlib import Path

# --- 1. Kontext-Klasse (mit cxt-Parser und -Exporter) ---

class FormalContext:
    def __init__(self, objects, attributes, incidence_matrix):
        self.objects = objects       # Liste der Dokumentnamen
        self.attributes = attributes # Liste der Merkmalsnamen
        self.matrix = incidence_matrix # Dict: doc -> set(attribute_names)

    @classmethod
    def from_cxt(cls, filepath):
        """Liest einen Kontext aus dem Burmeister (.cxt) Format ein."""
        with open(filepath, 'r', encoding='utf-8') as f:
            # Leere Zeilen ignorieren, da Burmeister-Format manchmal variiert
            lines = [line.strip() for line in f if line.strip()]
            
        if not lines or lines[0] != "B":
            raise ValueError(f"Datei {filepath} ist kein gültiges Burmeister-Format.")
        
        num_obj = int(lines[1])
        num_attr = int(lines[2])
        
        objects = lines[3 : 3 + num_obj]
        attributes = lines[3 + num_obj : 3 + num_obj + num_attr]
        matrix_lines = lines[3 + num_obj + num_attr : 3 + num_obj + num_attr + num_obj]
        
        incidence = {obj: set() for obj in objects}
        for i, row in enumerate(matrix_lines):
            obj = objects[i]
            for j, char in enumerate(row):
                if char.upper() == 'X':
                    incidence[obj].add(attributes[j])
                    
        return cls(objects, attributes, incidence)
        
    def save_cxt(self, filepath):
        """Speichert den Kontext im Burmeister (.cxt) Format."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("B\n\n")
            f.write(f"{len(self.objects)}\n")
            f.write(f"{len(self.attributes)}\n\n")
            
            for obj in self.objects:
                f.write(f"{obj}\n")
                
            for attr in self.attributes:
                f.write(f"{attr}\n")
                
            for obj in self.objects:
                row_str = "".join(["X" if attr in self.matrix[obj] else "." for attr in self.attributes])
                f.write(f"{row_str}\n")
                
        print(f"Schnitt-Kontext gespeichert: {filepath}")


# --- 2. Einzelner Hüllen-Operator (X'') ---

class ContextClosure:
    """Führt die Standard-Ableitungen X -> X' -> X'' für einen Einzelkontext durch."""
    def __init__(self, context):
        self.objects = context.objects
        self.attributes = context.attributes
        self.matrix = context.matrix
        
        # Optimierung: Merkmale zu ihren Dokumenten mappen (beschleunigt X'')
        self.attr_extents = {attr: set() for attr in self.attributes}
        for obj, attrs in self.matrix.items():
            for attr in attrs:
                self.attr_extents[attr].add(obj)
                
    def closure(self, doc_set):
        # 1. Schritt: X' (Alle gemeinsamen Merkmale der Dokumente)
        if not doc_set:
            intent = set(self.attributes)
        else:
            doc_list = list(doc_set)
            intent = set(self.matrix[doc_list[0]])
            for d in doc_list[1:]:
                intent.intersection_update(self.matrix[d])
                if not intent:
                    break
                    
        # 2. Schritt: X'' (Alle Dokumente, die diese gemeinsamen Merkmale haben)
        if not intent:
            return set(self.objects)
            
        intent_list = list(intent)
        extent = set(self.attr_extents[intent_list[0]])
        for a in intent_list[1:]:
            extent.intersection_update(self.attr_extents[a])
            if not extent:
                break
                
        return extent


# --- 3. GEMEINSAMER Hüllen-Operator (Alternierende Fixpunkt-Suche) ---

class CombinedClosureOracle:
    def __init__(self, ctx1, ctx2):
        # Prüfen ob beide Kontexte über denselben Objekten operieren
        if set(ctx1.objects) != set(ctx2.objects):
            raise ValueError("Die beiden Kontexte haben unterschiedliche Dokumentenmengen!")
            
        # Deterministische Sortierung (unabhängig davon, wie sie im File standen)
        self.sorted_docs = sorted(list(ctx1.objects))
        self.doc_map = {doc: i for i, doc in enumerate(self.sorted_docs)}
        
        self.c1 = ContextClosure(ctx1)
        self.c2 = ContextClosure(ctx2)
        
    def get_forced_closure(self, current_docs):
        """
        Alterniert die beiden Hüllenoperatoren, bis sich das Set nicht mehr ändert.
        Das Ergebnis ist garantiert abgeschlossen in BEIDEN Kontexten.
        """
        current = set(current_docs)
        while True:
            next_set = self.c1.closure(current)
            next_set = self.c2.closure(next_set)
            
            # Fixpunkt erreicht? (Da G endlich ist, bricht das zügig ab)
            if next_set == current:
                return current
            current = next_set


# --- 4. Next-Closure Algorithmus (mit Irreduzibilitäts-Check) ---

def generate_intersection_context(oracle):
    docs = oracle.sorted_docs
    n = len(docs)
    all_docs_set = set(docs)
    
    current_set = oracle.get_forced_closure(set())
    reduced_clusters =[]
    
    print("Starte Next-Closure für gemeinsamen Begriffsverband...")
    
    while True:
        # --- 1. On-the-fly Irreduzibilitäts-Check ---
        if len(current_set) < n:
            intersection_supersets = all_docs_set
            is_irreducible = True
            
            for d in all_docs_set:
                if d not in current_set:
                    candidate_closure = oracle.get_forced_closure(current_set | {d})
                    intersection_supersets = intersection_supersets.intersection(candidate_closure)
                    
                    if intersection_supersets == current_set:
                        is_irreducible = False
                        break
            
            if is_irreducible:
                reduced_clusters.append(current_set)

        # --- 2. Standard Next-Closure Übergang ---
        next_set = None
        current_indices = [oracle.doc_map[d] for d in current_set]
        
        for i in range(n - 1, -1, -1):
            if i in current_indices:
                current_indices.remove(i)
            else:
                candidate_indices = current_indices + [i]
                candidate_docs = {docs[idx] for idx in candidate_indices}
                closure_docs = oracle.get_forced_closure(candidate_docs)
                closure_indices = sorted([oracle.doc_map[d] for d in closure_docs])
                
                is_canonical = True
                for j in range(i):
                    in_closure = j in closure_indices
                    in_candidate = j in candidate_indices
                    if in_closure != in_candidate:
                        is_canonical = False; break
                        
                if is_canonical:
                    next_set = closure_docs
                    break
                    
        if next_set is None: 
            break
            
        current_set = next_set

    print(f"Fertig! Gemeinsame irreduzible Begriffe gefunden: {len(reduced_clusters)}")
    
    # --- 3. Neuen Kontext aus den irreduziblen Begriffen bauen ---
    incidence = {d: set() for d in docs}
    attr_names =[]
    
    for i, cluster in enumerate(reduced_clusters):
        attr_name = f"Common_M{i}"
        attr_names.append(attr_name)
        for doc in cluster:
            incidence[doc].add(attr_name)
            
    return FormalContext(docs, attr_names, incidence)


# --- MAIN ---

if __name__ == "__main__":
    # Pfade anpassen!
    cxt1_path = "context1.cxt"
    cxt2_path = "context2.cxt"
    output_path = "intersection_context.cxt"
    
    if not (Path(cxt1_path).is_file() and Path(cxt2_path).is_file()):
        print("Bitte stelle sicher, dass 'context1.cxt' und 'context2.cxt' existieren.")
        sys.exit(1)

    print("Lade Kontexte...")
    ctx1 = FormalContext.from_cxt(cxt1_path)
    ctx2 = FormalContext.from_cxt(cxt2_path)
    
    print("Baue gemeinsamen Hüllenoperator auf...")
    oracle = CombinedClosureOracle(ctx1, ctx2)
    
    # Schnitt-Kontext berechnen
    new_ctx = generate_intersection_context(oracle)
    
    # Das Ergebnis speichern
    new_ctx.save_cxt(output_path)