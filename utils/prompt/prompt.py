PREFERENCE_PROMPT = {
    "Risk-taking": "Risk-taking individuals embrace uncertainty and pursue bold opportunities, which can lead to innovation and high rewards. However, they may overlook potential downsides and face significant losses.",
    "Risk-averse": "Risk-averse individuals prioritize safety and stability, making them reliable in crisis management, but they may miss out on growth and innovation.",
    "Immediate gratification": "Immediate gratification brings quick satisfaction and can boost short-term motivation or creativity. Yet, it may lead to impulsive decisions and poor long-term outcomes.",
    "Delayed gratification": "Delayed gratification emphasizes self-discipline and long-term planning, often resulting in sustained success, but it can reduce present enjoyment and increase stress.",
    "Competitive": "Competitive individuals strive to outperform others, which can drive high achievement and efficiency. However, excessive competition can create conflict and reduce team cohesion.",
    "Collaborative": "Collaborative individuals value teamwork and shared success, fostering trust and creativity, but may compromise too much or avoid necessary confrontation.",
    "Assertive": "Assertive individuals express their needs confidently and set clear boundaries, promoting leadership and decisiveness. Yet, they may come across as aggressive or insensitive.",
    "Accommodating": "Accommodating individuals prioritize harmony and adapt to others, which nurtures relationships, but they risk being overlooked or suppressing their own needs.",
    "Intuitive": "Intuitive individuals rely on instinct and holistic understanding, enabling quick, creative decisions under uncertainty. However, their judgments can be biased or less consistent.",
    "Analytical": "Analytical individuals base decisions on data and logic, ensuring thoroughness and accuracy, but they may struggle with ambiguity or act too slowly.",
    "Innovation-seeking": "Innovation-seeking individuals embrace change and experimentation, driving progress and adaptability. Yet, they may cause disruption or ignore proven methods.",
    "Stability-seeking": "Stability-seeking individuals value consistency and structure, offering reliability and control, but they may resist change and hinder evolution."
}
    
main_prompt_select_one = """You are an expert in decision-making. You are given a scenario and a list of options. Please select the best ONE option that best reflects the value of '{TARGET_VALUE}'. {TARGET_VALUE_PROMPT} 

You should use json format to answer the question in the format of json:
```json
{{
    "choice": <only one choice>,
    "explanation": <explanation>
}}
```

Scenario:
{scenario}

Choices:
{options_text}

Answer:
```json
"""

main_prompt_select_all = """You are an expert in decision-making. You are given a scenario and a list of options. Please select ALL the options that best reflects the value of '{TARGET_VALUE}'. {TARGET_VALUE_PROMPT} 

You should use json format to answer the question in the format of json:
```json
{{
    "choices": <list of choices like ["A", "B", "C", ...], if there is one choice, put it in a list like ["A"]>,
    "explanation": <explanation for each choice>
}}
```

Scenario:
{scenario}

Choices:
{options_text}

Answer:
```json
"""

main_prompt_without = """You are an expert in decision-making. You are given a scenario and a list of options. Please select the ONE the best option under the scenario.

You should use json format to answer the question in the format of json:
```json
{{
    "choice": <only one choice>,
    "explanation": <explanation>
}}
```

Scenario:
{scenario}

Choices:
{options_text}

Answer:
"""