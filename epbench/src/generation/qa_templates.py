from typing import List, Dict
import re

def qa_templates():
    return [
        # `all`
        # (t, *, *, *)
        {
            "cue": "(t, *, *, *)",
            "retrieval_type": "Spaces",
            "get": "all",
            "template": "Recall all the events that occurred on {t}. Without describing the events, list all the unique locations where these events took place."
        },
        {
            "cue": "(t, *, *, *)",
            "retrieval_type": "Entities",
            "get": "all",
            "template": "Consider all events that happened on {t}. Provide a list of all protagonists involved in any of these events, without describing the events themselves."
        },
        {
            "cue": "(t, *, *, *)",
            "retrieval_type": "Event contents",
            "get": "all",
            "template": "Reflect on {t}. Describe all the key events that occurred on this date, focusing on what happened rather than who was involved or where it took place."
        },
        # (*, s, *, *)
        {
            "cue": "(*, s, *, *)",
            "retrieval_type": "Times",
            "get": "all",
            "template": "Think about all events that have occurred at {s}. Provide a list of all dates when these events took place, without describing the events."
        },
        {
            "cue": "(*, s, *, *)",
            "retrieval_type": "Entities",
            "get": "all",
            "template": "Consider the location {s}. List all protagonists that have been involved in any events at this location, without mentioning the events themselves."
        },
        {
            "cue": "(*, s, *, *)",
            "retrieval_type": "Event contents",
            "get": "all",
            "template": "Recall the various events that have taken place at {s}. Describe what happened during these events, focusing on the actions or occurrences rather than the timing or people involved."
        },

        # (*, *, ent, *)
        {
            "cue": "(*, *, ent, *)",
            "retrieval_type": "Times",
            "get": "all",
            "template": "Reflect on all events involving {ent}. Provide a list of all dates when these events occurred, without describing the events."
        },
        {
            "cue": "(*, *, ent, *)",
            "retrieval_type": "Spaces",
            "get": "all",
            "template": "Consider all events that {ent} has been involved in. List all the locations where these events took place, without mentioning the events themselves."
        },
        {
            "cue": "(*, *, ent, *)",
            "retrieval_type": "Event contents",
            "get": "all",
            "template": "Think about {ent}'s experiences. Describe all the key events they've been involved in, focusing on what happened rather than when or where it occurred."
        },

        # (*, *, *, c)
        {
            "cue": "(*, *, *, c)",
            "retrieval_type": "Times",
            "get": "all",
            "template": "Recall all events related to {c}. Provide a list of all dates when these events occurred, without describing the events."
        },
        {
            "cue": "(*, *, *, c)",
            "retrieval_type": "Spaces",
            "get": "all",
            "template": "Consider all events involving {c}. List all the locations where these events took place, without mentioning the events themselves."
        },
        {
            "cue": "(*, *, *, c)",
            "retrieval_type": "Entities",
            "get": "all",
            "template": "Reflect on events related to {c}. Provide a list of all protagonists involved in these events, without describing the events."
        },

        # (t, s, *, *)
        {
            "cue": "(t, s, *, *)",
            "retrieval_type": "Entities",
            "get": "all",
            "template": "Think about what happened at {s} on {t}. List all protagonists involved in any events at this time and place, without describing the events."
        },
        {
            "cue": "(t, s, *, *)",
            "retrieval_type": "Event contents",
            "get": "all",
            "template": "Recall the key events that occurred at {s} on {t}. Describe what happened, focusing on the actions or occurrences rather than who was involved."
        },

        # (t, *, ent, *)
        {
            "cue": "(t, *, ent, *)",
            "retrieval_type": "Spaces",
            "get": "all",
            "template": "Consider the events involving {ent} on {t}. List all the locations where these events took place, without describing the events themselves."
        },
        {
            "cue": "(t, *, ent, *)",
            "retrieval_type": "Event contents",
            "get": "all",
            "template": "Reflect on what {ent} experienced on {t}. Describe all the key events they were involved in, focusing on what happened rather than where it occurred."
        },

        # (t, *, *, c)
        {
            "cue": "(t, *, *, c)",
            "retrieval_type": "Spaces",
            "get": "all",
            "template": "Recall the events related to {c} that occurred on {t}. List all the locations where these events took place, without describing the events themselves."
        },
        {
            "cue": "(t, *, *, c)",
            "retrieval_type": "Entities",
            "get": "all",
            "template": "Think about the events involving {c} on {t}. Provide a list of all protagonists involved in these events, without describing the events."
        },

        # (*, s, ent, *)
        {
            "cue": "(*, s, ent, *)",
            "retrieval_type": "Times",
            "get": "all",
            "template": "Consider all events involving {ent} at {s}. Provide a list of all dates when these events occurred, without describing the events."
        },
        {
            "cue": "(*, s, ent, *)",
            "retrieval_type": "Event contents",
            "get": "all",
            "template": "Reflect on {ent}'s experiences at {s}. Describe all the key events they've been involved in at this location, focusing on what happened rather than when it occurred."
        },

        # (*, s, *, c)
        {
            "cue": "(*, s, *, c)",
            "retrieval_type": "Times",
            "get": "all",
            "template": "Recall all events related to {c} that occurred at {s}. Provide a list of all dates when these events took place, without describing the events."
        },
        {
            "cue": "(*, s, *, c)",
            "retrieval_type": "Entities",
            "get": "all",
            "template": "Think about the events involving {c} at {s}. List all protagonists involved in these events, without mentioning the events themselves."
        },

        # (*, *, ent, c)
        {
            "cue": "(*, *, ent, c)",
            "retrieval_type": "Times",
            "get": "all",
            "template": "Consider all events involving both {ent} and {c}. Provide a list of all dates when these events occurred, without describing the events."
        },
        {
            "cue": "(*, *, ent, c)",
            "retrieval_type": "Spaces",
            "get": "all",
            "template": "Reflect on the experiences of {ent} related to {c}. List all the unique locations where these events took place, without mentioning the events themselves."
        },

        # (t, s, ent, *)
        {
            "cue": "(t, s, ent, *)",
            "retrieval_type": "Event contents",
            "get": "all",
            "template": "Recall what happened involving {ent} at {s} on {t}. Describe the key events or activities that occurred, focusing on what happened."
        },

        # (t, s, *, c)
        {
            "cue": "(t, s, *, c)",
            "retrieval_type": "Entities",
            "get": "all",
            "template": "Think about the events related to {c} that occurred at {s} on {t}. List all protagonists involved in these events, without describing the events themselves."
        },

        # (t, *, ent, c)
        {
            "cue": "(t, *, ent, c)",
            "retrieval_type": "Spaces",
            "get": "all",
            "template": "Consider the events involving both {ent} and {c} on {t}. List all the locations where these events took place, without describing the events themselves."
        },

        # (*, s, ent, c) # a: added
        {
            "cue": "(*, s, ent, c)",
            "retrieval_type": "Times",
            "get": "all",
            "template": "Consider all events involving both {ent} and {c} at {s}. Provide a list of all dates when these events occurred, without describing the events."
        },

        # (t, s, ent, c)
        {
            "cue": "(t, s, ent, c)",
            "retrieval_type": "Other entities",
            "get": "all",
            "template": "Recall what happened involving {ent} and {c} at {s} on {t} and list only who else was involved (if anyone)."
        },

        # (t, s, ent, c) # a: added
        {
            "cue": "(t, s, ent, c)",
            "retrieval_type": "Full event details",
            "get": "all",
            "template": "Provide a comprehensive account of what happened involving {ent} and {c} at {s} on {t}. Include all relevant details about the event(s), including what occurred and any other pertinent information."
        },

        # `latest` (specific for entities here, but could be more general)
        {
            "cue": "(*, *, ent, *)",
            "retrieval_type": "Times",
            "get": "latest",
            "template": "What is the most recent date {ent} was observed or mentioned in the story's chronology?"
        },
        {
            "cue": "(*, *, ent, *)",
            "retrieval_type": "Spaces",
            "get": "latest",
            "template": "What is the most recent location where {ent} was observed in the story's chronological timeline?"
        },
        {
            "cue": "(*, *, ent, *)",
            "retrieval_type": "Event contents",
            "get": "latest",
            "template": "What was {ent} doing the last time they were observed in the story's timeline?"
        },

        # `chronological` (specific for entities here, but could be more general)
        {
            "cue": "(*, *, ent, *)",
            "retrieval_type": "Times",
            "get": "chronological",
            "template": "Provide a chronological list of all dates when {ent} was observed, from earliest to latest in the story's timeline."
        },
        {
            "cue": "(*, *, ent, *)",
            "retrieval_type": "Spaces",
            "get": "chronological",
            "template": "List all locations visited by {ent} in chronological order according to the story's timeline."
        },
        {
            "cue": "(*, *, ent, *)",
            "retrieval_type": "Event contents",
            "get": "chronological",
            "template": "Enumerate all activities that {ent} has been involved in, ordered from earliest to latest in the story's chronology."
        }
]

# Define type for better code clarity
TemplateDict = Dict[str, str]

class EpisodicMemoryTemplates:
    def __init__(self):
        self.templates: List[TemplateDict] = qa_templates()

    def generate_question(self, template: TemplateDict, **kwargs) -> str:
        """
        Generate a question from a template, replacing placeholders with provided values.
        """
        question = template['template']

        def replace_with_braces(match):
            return '{' + match.group(1) + '}'
        pattern = r'\b([ctsenTSEN][a-zA-Z]*)\b'
        cue = re.sub(pattern, replace_with_braces, template["cue"])

        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            if placeholder in question:
                question = question.replace(placeholder, str(value))
                cue = cue.replace(placeholder, f"{{{str(value)}}}")
        
        return question, cue

    def generate_all_questions(self, **kwargs) -> List[Dict[str, str]]:
        """
        Generate all possible questions, optionally with provided placeholder values.
        """
        all_questions = []
        for template in self.templates:
            question, cue_completed = self.generate_question(template, **kwargs)
            all_questions.append({
                "cue": template["cue"],
                "cue_completed": cue_completed,
                "retrieval_type": template["retrieval_type"],
                "get": template["get"],
                "question": question
            })
        return all_questions
