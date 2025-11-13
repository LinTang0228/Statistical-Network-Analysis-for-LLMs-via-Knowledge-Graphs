# schema_prompt.py

"""Extract triples using only the provided relations: adverse effect. Do not use or invent other relations. Use the appropriate entity types: AdverseEffect, Drug."""

from typing import List
class Rel:
    # Allow the name to be optional
    def __init__(self, name: str = ""):
        self.name = name
class Adverse_Effect(Rel):
    def __init__(self):
        super().__init__("adverse effect")
class Entity:
    # Allow the name to be optional
    def __init__(self, name: str = ""):
        self.name = name
class AdverseEffect(Entity):
    def __init__(self, name: str = ""):
        super().__init__(name)
class Drug(Entity):
    def __init__(self, name: str = ""):
        super().__init__(name)
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
adverse_effect = AdverseEffect
drug = Drug
# Create aliases to handle inconsistent capitalization and common variants
Adverse_effect = Adverse_Effect
Adverse_Effect_Rel = Adverse_Effect