"""Official ORena FOCUS capability taxonomy used by the visualiser.

The descriptions are concise paraphrases of the challenge taxonomy at
https://procedure.orena-focus-challenge.org/data/.
"""

from __future__ import annotations

TAXONOMY_SOURCE = "https://procedure.orena-focus-challenge.org/data/"

CAPABILITY_GROUPS = [
    {
        "code": "1",
        "name": "Object recognition and identity matching",
        "definition": (
            "Recognise object identity, semantic type, visible attributes, spatial "
            "context, and consistency across time."
        ),
        "children": [
            {
                "code": "1a",
                "name": "Object identification",
                "definition": "Classify the semantic type of a visible object instance.",
            },
            {
                "code": "1b",
                "name": "Object instance identity matching",
                "definition": "Match the same object instance across different times.",
            },
            {
                "code": "1c",
                "name": "Object attributes and state",
                "definition": "Recognise visible object properties or handling states.",
            },
            {
                "code": "1d",
                "name": "Object spatial localization (camera)",
                "definition": "Locate an object relative to the image plane.",
            },
            {
                "code": "1e",
                "name": "Object spatial localization (situs)",
                "definition": "Locate an object relative to anatomical structures.",
            },
        ],
    },
    {
        "code": "2",
        "name": "Temporal grounding",
        "definition": "Locate object occurrences and events in time, including duration.",
        "children": [
            {
                "code": "2a",
                "name": "Temporal localization",
                "definition": "Identify when an object occurrence or event takes place.",
            },
            {
                "code": "2b",
                "name": "Duration estimation",
                "definition": "Estimate how long an object occurrence or event persists.",
            },
        ],
    },
    {
        "code": "3",
        "name": "Aggregation",
        "definition": "Combine evidence across objects, instances, events, or time points.",
        "children": [
            {
                "code": "3a",
                "name": "Object aggregation",
                "definition": "Aggregate information across object instances or categories.",
            },
            {
                "code": "3b",
                "name": "Event aggregation",
                "definition": "Aggregate foreign-object events across time.",
            },
        ],
    },
    {
        "code": "4",
        "name": "Event and procedural understanding",
        "definition": "Recognise actions, events, and their procedural structure over time.",
        "children": [
            {
                "code": "4a",
                "name": "Foreign object interaction recognition",
                "definition": "Identify actions or manipulations involving a foreign object.",
            },
            {
                "code": "4b",
                "name": "Foreign object usage purpose",
                "definition": "Identify the visually demonstrated procedural role of an object.",
            },
            {
                "code": "4c",
                "name": "Temporal ordering",
                "definition": "Determine relative event order or an event's position in a series.",
            },
        ],
    },
    {
        "code": "5",
        "name": "Complex reasoning",
        "definition": (
            "Reason beyond direct observation about functions, causes, relationships, "
            "outcomes, or extended evidence."
        ),
        "children": [
            {
                "code": "5a",
                "name": "Functional reasoning",
                "definition": "Infer an object's or action's function beyond direct observation.",
            },
            {
                "code": "5b",
                "name": "Causal and consequence reasoning",
                "definition": "Infer causes, effects, or actual and potential outcomes.",
            },
            {
                "code": "5c",
                "name": "Multi-step compositional reasoning",
                "definition": "Combine several dependent reasoning steps across evidence types.",
            },
        ],
    },
]
