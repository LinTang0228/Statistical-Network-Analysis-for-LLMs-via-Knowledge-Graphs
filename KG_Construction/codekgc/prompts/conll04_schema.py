"""Extract triples using only the provided relations: Work For, Located In, Organization based in, Live In, Kill. Do not use or invent other relations. Use the appropriate entity types: Person, Location, Organization, Other."""

from typing import List
class Rel:
    # Allow the name to be optional
    def __init__(self, name: str = ""):
        self.name = name
class Work_For(Rel):
    def __init__(self):
        super().__init__("Work For")
class Located_In(Rel):
    def __init__(self):
        super().__init__("Located In")
class OrgBased_In(Rel):
    def __init__(self):
        super().__init__("Organization based in")
class Live_In(Rel):
    def __init__(self):
        super().__init__("Live In")
class Kill(Rel):
    def __init__(self):
        super().__init__("Kill")
class Entity:
    # Allow the name to be optional
    def __init__(self, name: str = ""):
        self.name = name
class Person(Entity):
    def __init__(self, name: str = ""):
        super().__init__(name)
class Location(Entity):
    def __init__(self, name: str = ""):
        super().__init__(name)
class Organization(Entity):
    def __init__(self, name: str = ""):
        super().__init__(name)
class Other(Entity):
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
person = Person
location = Location
organization = Organization
other = Other
# Create aliases to handle inconsistent capitalization and common variants
Located_in = Located_In
OrgBased_in = OrgBased_In
Live_in = Live_In
Work_for = Work_For
Killed = Kill
Born_in = Live_In
Died_in = Live_In