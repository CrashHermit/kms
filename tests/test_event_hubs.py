from kms.construction import local_event_hubs, triplet_extractor


def test_triplet_prompt_separates_events_from_results():
    prompt = triplet_extractor._TripletSignature.__doc__
    assert 'full clause, explanatory phrase' in prompt
    assert 'shortest complete noun phrase' in prompt
    assert 'Never infer events' in prompt


def test_event_adjudication_prompt_rejects_related_events():
    prompt = local_event_hubs.EventHubAdjudicationSignature.__doc__
    assert 'Return FALSE when one is a consequence' in prompt
    assert 'battle and its victory are distinct events' in prompt
    assert 'Shared verbs or' in prompt


def test_event_synthesis_prompt_preserves_occurrence_boundaries():
    prompt = local_event_hubs.EventHubSynthesisSignature.__doc__
    assert 'not its consequence' in prompt
    assert 'Never invent time, participants' in prompt
