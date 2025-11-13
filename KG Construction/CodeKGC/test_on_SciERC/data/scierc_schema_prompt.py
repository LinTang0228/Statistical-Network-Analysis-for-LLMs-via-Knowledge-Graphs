# Updated schema_prompt.py for SciERC dataset
"""Extract triples using only the provided relations: Used-for, Feature-of, Hyponym-of, Part-of, Conjunction, Compare, Evaluate-for. Do not use or invent other relations. Use the entity type: Entity for all entities."""
from typing import List
class Rel:
    # Allow the name to be optional
    def __init__(self, name: str = ""):
        self.name = name
class Used_For(Rel):
    def __init__(self):
        super().__init__("Used-for")
class Feature_Of(Rel):
    def __init__(self):
        super().__init__("Feature-of")
class Hyponym_Of(Rel):
    def __init__(self):
        super().__init__("Hyponym-of")
class Part_Of(Rel):
    def __init__(self):
        super().__init__("Part-of")
class Conjunction(Rel):
    def __init__(self):
        super().__init__("Conjunction")
class Compare(Rel):
    def __init__(self):
        super().__init__("Compare")
class Evaluate_For(Rel):
    def __init__(self):
        super().__init__("Evaluate-for")
class Entity:
    # Allow the name to be optional
    def __init__(self, name: str = ""):
        self.name = name
class Triple:
    def __init__(self, head: Entity, relation: Rel, tail: Entity):
        self.head = head
        self.relation = relation
        self.tail = tail
class Extract:
    def __init__(self, triples: List[Triple] = []):
        self.triples = triples
# --- ALIASES AT THE END ---
# Create lowercase aliases for the model to use
entity = Entity
# Create aliases to handle inconsistent capitalization and common variants
Used_for = Used_For
Feature_of = Feature_Of
Hyponym_of = Hyponym_Of
Part_of = Part_Of
Evaluate_for = Evaluate_For