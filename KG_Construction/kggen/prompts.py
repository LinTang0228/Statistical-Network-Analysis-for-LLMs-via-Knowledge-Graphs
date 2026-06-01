"""Zero-shot extraction prompts for KGGen on the three benchmark datasets."""

PROMPT_ADE = (
    'Extract adverse drug events (ADEs) from the sentence into triples of the form '
    '("Specific Adverse Effect", "adverse_effect", "Drug/Chemical").\n'
    'Rules:\n'
    '1. The relation must always be the literal string "adverse_effect" (lowercase).\n'
    '2. Only extract causal ADEs (drug causes effect); ignore unrelated, negated, or '
    'uncertain mentions.\n'
    '3. The order is always (Effect, "adverse_effect", Drug). Flip if extracted reversely.\n'
    '4. Return a list of unique triples, or [] if none are present.\n'
    '5. Use precise phrases; trim unnecessary modifiers; avoid generic entities such as '
    '"patient" or ages.'
)

PROMPT_CONLL04 = (
    'Extract relations from the sentence using ONLY the relation names: '
    'kill, work_for, organization_based_in, live_in, located_in.\n'
    'Rules:\n'
    '1. Do NOT introduce any relation outside this list; any other name is invalid.\n'
    '2. Subject and object must be complete noun phrases taken from the sentence.\n'
    '3. Output a list of unique triples (Subject, relation, Object), or [] if none.\n'
    '4. Infer relations where strongly implied (e.g., "born in" implies live_in; '
    '"director of" implies work_for).'
)

PROMPT_SCIERC = (
    'Extract relations from the sentence using ONLY the relation names: '
    'used-for, feature-of, hyponym-of, part-of, compare, evaluate-for, conjunction.\n'
    'Rules:\n'
    '1. Do NOT introduce any relation outside this list; any other name is invalid.\n'
    '2. Subject and object must be complete noun phrases taken from the sentence.\n'
    '3. Output a list of unique triples (Subject, relation, Object), or [] if none.'
)

PROMPTS = {
    "ade": PROMPT_ADE,
    "conll04": PROMPT_CONLL04,
    "scierc": PROMPT_SCIERC,
}
