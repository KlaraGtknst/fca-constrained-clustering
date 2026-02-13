import itertools

# Werte im dreiwertigen Kontext K = (G, M, {x, -, ?}, I)
VAL_TRUE = 1     # Inzidenz
VAL_FALSE = 0    # Negative Inzidenz
VAL_UNKNOWN = -1 # Unbestimmt

class ObjectExplorationContext:
    def __init__(self, objects):
        # Wir operieren direkt auf G (Objekte als "Attribute" der Implikationen)
        self.objects = sorted(list(objects))
        self.obj_map = {o: i for i, o in enumerate(self.objects)}
        
        # M (Cluster/Themen) wächst dynamisch durch Gegenbeispiele
        self.attributes = {} 
        self.attr_counter = 0
        
        # Kanonische Basis der Objekt-Implikationen
        self.implications = []
        
        # MLB-Constraints als Experten-Wissen
        self.mlb_constraints = []

    def add_mlb_constraint(self, x, y, z):
        """Registriert Constraint (x,y,z): dist(x,y) < dist(x,z)."""
        if {x, y, z}.issubset(set(self.objects)):
            self.mlb_constraints.append((x, y, z))

    def add_attribute_column(self, values, name=None):
        """Fügt M ein neues Attribut (Cluster) als Gegenbeispiel hinzu."""
        if name is None:
            name = f"c_{self.attr_counter}"
            self.attr_counter += 1
        
        row = [VAL_UNKNOWN] * len(self.objects)
        for obj, val in values.items():
            if obj in self.obj_map:
                row[self.obj_map[obj]] = val
        
        self.attributes[name] = row
        # Konsistenz wiederherstellen: Fragezeichen gemäß neuer Basis auflösen
        self.reduce_incomplete_entries()

    def reduce_incomplete_entries(self):
        """
        Wendet die Implikationsbasis auf den partiellen Kontext an.
        Setzt '?' auf '1', wenn Prämisse in einer Spalte vollständig '1' ist.
        """
        changed = True
        while changed:
            changed = False
            for values in self.attributes.values():
                for (premise, conclusion) in self.implications:
                    p_idxs = [self.obj_map[o] for o in premise]
                    
                    if all(values[i] == VAL_TRUE for i in p_idxs):
                        for conc_obj in conclusion:
                            c_idx = self.obj_map[conc_obj]
                            if values[c_idx] == VAL_UNKNOWN:
                                values[c_idx] = VAL_TRUE
                                changed = True

    def compute_syntactic_closure(self, A):
        """Berechnet A^L (Hülle bzgl. der aktuellen Basis)."""
        closure = set(A)
        changed = True
        while changed:
            changed = False
            for (premise, conclusion) in self.implications:
                if set(premise).issubset(closure):
                    if not set(conclusion).issubset(closure):
                        closure.update(conclusion)
                        changed = True
        return sorted(list(closure))

    def get_semantic_closure_in_partial_context(self, A):
        """
        Berechnet A'' im partiellen Kontext.
        g in A'' gdw. kein m in M existiert mit A subset m' und g not in m' (explizit 0).
        Fragezeichen werden konservativ behandelt (widerlegen die Hülle nicht).
        """
        candidates = set(self.objects)
        
        for values in self.attributes.values():
            # Prüfe A <= m'
            premise_holds = True
            for a in A:
                if values[self.obj_map[a]] != VAL_TRUE:
                    premise_holds = False
                    break
            
            if premise_holds:
                # Entferne Kandidaten, die explizit nicht im Attribut enthalten sind
                to_remove = set()
                for cand in candidates:
                    if values[self.obj_map[cand]] == VAL_FALSE:
                        to_remove.add(cand)
                candidates -= to_remove
                
        return sorted(list(candidates))

    def consult_expert(self, premise, conclusion):
        """
        Prüft Hypothese A -> B gegen MLB-Constraints.
        Rückgabe: (Accepted, Info)
        """
        premise_set = set(premise)
        conclusion_set = set(conclusion)
        
        # 1. Separation Requirement
        # Ein MLB(x,y,z) verbietet Implikationen, die z erzwingen, wenn x,y gegeben sind.
        # Check: Wenn A <= {x,y} und z in B \ A, dann Widerspruch.
        for (x, y, z) in self.mlb_constraints:
            if premise_set.issubset({x, y}):
                if z in conclusion_set and z not in premise_set:
                    return False, ("SEPARATION", x, y, z)

        # 2. Connectivity Requirement (Konvexität)
        # MLB(x,y,z) fordert: Wenn x, z im Cluster, dann auch y.
        # Wir erweitern die Konklusion bis zum Fixpunkt.
        extended_conclusion = set(conclusion)
        changed = True
        while changed:
            changed = False
            scope = premise_set | extended_conclusion
            for (x, y, z) in self.mlb_constraints:
                if x in scope and z in scope:
                    if y not in extended_conclusion and y not in premise_set:
                        extended_conclusion.add(y)
                        changed = True
        
        return True, sorted(list(extended_conclusion))

    def explore(self):
        """
        Next-Closure auf P(G).
        Generiert kanonische Basis für die durch MLB definierte Theorie.
        """
        A = [] # Start mit lektisch kleinstem Element
        
        while True:
            # 1. Syntaktische vs. Semantische Hülle
            A_L = self.compute_syntactic_closure(A)
            A_double_prime = self.get_semantic_closure_in_partial_context(A_L)
            
            # Wenn A^L != A'', ist A^L pseudo-abgeschlossen im Kontext -> Hypothese
            if set(A_L) != set(A_double_prime):
                premise = A_L
                conclusion_diff = [g for g in A_double_prime if g not in premise]
                
                print(f"Hypothese: {{{', '.join(premise)}}} -> {{{', '.join(conclusion_diff)}}}")
                
                accepted, feedback = self.consult_expert(premise, A_double_prime)
                
                if accepted:
                    final_conclusion = feedback
                    if set(final_conclusion) != set(A_double_prime):
                        print(f"  -> Verfeinert durch Connectivity zu: {final_conclusion}")
                    else:
                        print("  -> Akzeptiert.")
                    
                    self.implications.append((premise, final_conclusion))
                    self.reduce_incomplete_entries()
                    # Basis geändert -> Neustart mit demselben A
                    continue
                
                else:
                    # Ablehnung durch Separation -> Gegenbeispiel generieren
                    _, x, y, z = feedback
                    print(f"  -> Abgelehnt (Separation): ({x}, {y}) schließt {z} aus.")
                    
                    # Neuer Cluster, der x,y enthält (1) aber z nicht (0).
                    # Prämisse muss explizit auf 1 gesetzt werden, um als Gegenbeispiel zu wirken.
                    new_attr_vals = {x: VAL_TRUE, y: VAL_TRUE, z: VAL_FALSE}
                    for p in premise:
                        new_attr_vals[p] = VAL_TRUE
                        
                    self.add_attribute_column(new_attr_vals, name=f"Sep_{x}{y}|{z}")
                    # Kontext geändert -> A'' ändert sich -> Neustart mit demselben A
                    continue

            # 2. Next-Closure Schritt
            next_A = self.get_next_closure(A_L)
            if next_A is None:
                break
            A = next_A

    def get_next_closure(self, current_set):
        """Standard Next-Closure Schritt."""
        for i in range(len(self.objects) - 1, -1, -1):
            obj = self.objects[i]
            if obj in current_set:
                current_set = [o for o in current_set if o != obj]
            else:
                candidate = current_set + [obj]
                candidate_closure = self.compute_syntactic_closure(candidate)
                
                is_canonical = True
                for o_c in candidate_closure:
                    if self.obj_map[o_c] < i and o_c not in candidate:
                        is_canonical = False
                        break
                
                if is_canonical:
                    return candidate_closure
        return None

    def print_results(self):
        print("\n=== Resultierende Basis ===")
        for p, c in self.implications:
            diff = [x for x in c if x not in p]
            if diff:
                print(f"{{{', '.join(p)}}} -> {{{', '.join(diff)}}}")
        
        print("\n=== Finaler Partieller Kontext (G x M) ===")
        print(f"{'Objekt':<10} | " + " | ".join(self.attributes.keys()))
        print("-" * (10 + len(self.attributes)*15))
        for i, obj in enumerate(self.objects):
            row_str = []
            for vals in self.attributes.values():
                v = vals[i]
                char = "x" if v == 1 else ("-" if v == 0 else "?")
                row_str.append(char)
            print(f"{obj:<10} | {' | '.join(row_str)}")


if __name__ == "__main__":
    docs = ["d1", "d2", "d3", "d4"]
    ctx = ObjectExplorationContext(docs)

    # Hierarchie-Annahme: ((d1, d2), d3), d4
    # Constraints definieren die Topologie.
    ctx.add_mlb_constraint("d1", "d2", "d3")
    ctx.add_mlb_constraint("d2", "d3", "d4")
    
    ctx.explore()
    ctx.print_results()