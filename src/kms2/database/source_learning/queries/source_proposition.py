"""Cypher statements for loading proposition-grounded learning requests."""

READ_SOURCE_ATOMIC_FLASHCARD_REQUESTS = """
MATCH (source:Source {uuid: $source_uuid})
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(owner:SourceBlock)
      -[:MEMBER_OF]->(proposition:SourceProposition)
      -[:IN_SOURCE_HUB]->(hub:SourcePropositionHub)
MATCH (source)-[:FIRST_BLOCK]->(first_block:SourceBlock)
MATCH path = (first_block)-[:NEXT_BLOCK*0..]->(owner)
WITH owner, hub, proposition, length(path) AS owner_position
ORDER BY owner_position, proposition.uuid
WITH hub, collect({
    owner: owner,
    proposition: proposition,
    owner_position: owner_position
}) AS members
WITH hub, members, members[0].proposition.text AS proposition_statement
UNWIND members AS member
WITH hub, member.owner AS owner, member.proposition AS proposition,
     member.owner_position AS owner_position, proposition_statement
RETURN proposition.uuid AS source_proposition_uuid,
       proposition.text AS proposition_text,
       owner.uuid AS source_block_uuid,
       owner.block_type AS block_type,
       owner.content AS content,
       hub.uuid AS hub_uuid,
       proposition_statement
ORDER BY owner_position, source_proposition_uuid
"""
