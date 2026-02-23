import sys
import os
from pathlib import Path


# --- 1. MLB Oracle & Logic ---
class MLBOracle:
    def __init__(self, filepath):
        self.constraints = set()
        self.documents = self.load_constraints(filepath)
        self.saturate_constraints()
        self.sorted_docs = sorted(list(self.documents))
        self.doc_map = {doc: i for i, doc in enumerate(self.sorted_docs)}

    def load_constraints(self, filepath):
        documents = set()
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 3:
                        pair = sorted([parts[0], parts[1]])
                        self.constraints.add((pair[0], pair[1], parts[2]))
                        documents.update(parts)
        except FileNotFoundError:
            sys.exit(1)
        return documents

    def saturate_constraints(self):
        changed = True
        while changed:
            changed = False
            current_list = list(self.constraints)
            n = len(current_list)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    x1, y1, z1 = current_list[i]
                    pair1 = {x1, y1}
                    x2, y2, z2 = current_list[j]
                    pair2 = {x2, y2}
                    
                    # Theorem 2
                    if z1 == z2:
                        intersection = pair1.intersection(pair2)
                        if len(intersection) == 1:
                            new_sibs = list(pair1.union(pair2) - intersection)
                            new_c = (sorted(new_sibs)[0], sorted(new_sibs)[1], z1)
                            if new_c not in self.constraints:
                                self.constraints.add(new_c); changed = True
                    # Theorem 3
                    if z1 in pair2:
                        pivot = pair1.intersection(pair2)
                        if len(pivot) == 1:
                            new_c = (x1, y1, z2)
                            if new_c not in self.constraints:
                                self.constraints.add(new_c); changed = True

    def get_forced_closure(self, current_docs):
        closure = set(current_docs)
        changed = True
        while changed:
            changed = False
            for dx, dy, dz in self.constraints:
                if dx in closure and dz in closure and dy not in closure:
                    closure.add(dy); changed = True
                if dy in closure and dz in closure and dx not in closure:
                    closure.add(dx); changed = True
        return closure

# --- 2. Kontext-Klasse mit Burmeister-Export ---

class FormalContext:
    def __init__(self, objects, attributes, incidence_matrix):
        self.objects = objects       # Liste der Dokumentnamen
        self.attributes = attributes # Liste der Merkmalsnamen (C1, C2...)
        self.matrix = incidence_matrix # Dict: doc -> set(attribute_names)
        
    def save_cxt(self, filepath):
        """
        Speichert den Kontext im Burmeister (.cxt) Format.
        """
        with open(filepath, 'w') as f:
            # 1. Header "B"
            f.write("B\n\n")
            
            # 2. Anzahl Objekte und Attribute
            f.write(f"{len(self.objects)}\n")
            f.write(f"{len(self.attributes)}\n\n")
            
            # 3. Objektnamen (Zeilenweise)
            for obj in self.objects:
                f.write(f"{obj}\n")
                
            # 4. Attributnamen (Zeilenweise)
            for attr in self.attributes:
                f.write(f"{attr}\n")
                
            # 5. Inzidenzmatrix (X für True, . für False)
            for obj in self.objects:
                row_str = ""
                for attr in self.attributes:
                    if attr in self.matrix[obj]:
                        row_str += "X"
                    else:
                        row_str += "."
                f.write(f"{row_str}\n")
                
        print(f"Datei gespeichert: {filepath}")

# --- 3. Generierungs-Logik (Reduziert) ---

def filter_irreducibles(all_clusters, all_docs_set):
    clusters = [set(c) for c in all_clusters]
    irreducibles = []
    
    # Top-Element (Cluster mit allen Docs) ist oft redundant für die Struktur
    full_set = set(all_docs_set)
    
    for c in clusters:
        # Wenn der Cluster alle Dokumente enthält, überspringen wir ihn oft (optional)
        if c == full_set:
            continue
            
        supersets = [s for s in clusters if c < s]
        
        if not supersets:
            # Wenn es keine Obermengen gibt (und es nicht das FullSet war), behalten wir es
            irreducibles.append(c)
            continue
            
        intersection_of_supers = set(all_docs_set)
        for s in supersets:
            intersection_of_supers = intersection_of_supers.intersection(s)
            
        if c != intersection_of_supers:
            irreducibles.append(c)
            
    return irreducibles

def generate_context_obj(oracle):
    docs = oracle.sorted_docs
    n = len(docs)
    current_set = oracle.get_forced_closure(set())
    valid_clusters = []
    
    # Next-Closure Loop
    while True:
        valid_clusters.append(current_set)
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
                    next_set = closure_docs; break
        if next_set is None: break
        current_set = next_set

    # Reduzieren auf irreduzible Merkmale
    reduced_clusters = filter_irreducibles(valid_clusters, set(docs))
    
    incidence = {d: set() for d in docs}
    attr_names = []
    
    for i, cluster in enumerate(reduced_clusters):
        # Name: M{i} oder eine Beschreibung
        attr_name = f"M{i}"
        attr_names.append(attr_name)
        for doc in cluster:
            incidence[doc].add(attr_name)
            
    return FormalContext(docs, attr_names, incidence)

# --- MAIN ---

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    constraint_file = Path("resources/banksearch/ground_truth/mlb_banksearch_docids.txt")
    save_file = "mlb.cxt"
    if not Path(constraint_file).is_file():
        constraint_file = PROJECT_ROOT / constraint_file
        save_file = PROJECT_ROOT / "resources/banksearch/ground_truth" / save_file

    oracle = MLBOracle(constraint_file)
    
    # 1. Kontext generieren
    ctx = generate_context_obj(oracle)
    
    # 2. Als .cxt speichern
    ctx.save_cxt(save_file)