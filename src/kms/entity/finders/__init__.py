"""Entity finders: one cursor-walk over the node stream, emitting each found block as member ids.

``block`` is the general, type-agnostic finder; ``problem``/``definition``/``theorem`` are the
validated math path it will replace once measured at parity (``docs/GENERALIZATION.md``, step 5).
All four are copies of the same growing-window walk — only the "what am I looking for" clause of the
Signature differs, which is what makes the collapse into one finder safe."""
