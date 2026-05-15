"""
generation_config.py
Defines the 12 persona archetypes, their preference vectors, and the
prompt templates used to generate synthetic training conversations.

All conversation text generated from these templates must be in Romanian.
The meta-instructions to the generation LLM are in English for quality,
but the OUTPUT (the conversation itself) must be fully Romanian.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 8 preference dimensions (matches the RFC)
# ---------------------------------------------------------------------------
PREFERENCE_DIMENSIONS = [
    "clima",
    "economie_locuri_munca",
    "demografie",
    "transport",
    "educatie",
    "sanatate",
    "cultura_turism",
    "cost_viata",
]

# ---------------------------------------------------------------------------
# 12 persona archetypes
# Each has:
#   description         : Romanian-language description injected into the prompt
#   preference_vector   : values used for city scoring / sampling
# ---------------------------------------------------------------------------
PERSONAS: dict[str, dict] = {
    "tanar_profesionist_it": {
        "description": (
            "Tânăr de 27 de ani, programator full-stack cu 4 ani experiență, "
            "lucrează remote pentru o firmă din Cluj. Vrea să se mute din București, "
            "preferă un oraș mai mic dar activ, cu natură în apropiere și chirii rezonabile."
        ),
        "preference_vector": {
            "clima": "moderat, nu extrem, aproape de munte sau natură",
            "economie_locuri_munca": "IT puternic sau remote-friendly, coworking",
            "demografie": "oras mediu, 80k-300k locuitori",
            "transport": "cale ferată spre București sau aeroport util",
            "educatie": "nu prioritar",
            "sanatate": "spital disponibil",
            "cultura_turism": "viața activă, restaurante, trasee montane",
            "cost_viata": "chirii sub București, accesibil",
        },
    },
    "familie_cu_copii": {
        "description": (
            "Cuplu, 35 și 33 de ani, cu doi copii (8 și 5 ani). "
            "Părinții lucrează în educație și contabilitate. "
            "Prioritizează școli bune, siguranță, spații verzi și un cartier liniștit."
        ),
        "preference_vector": {
            "clima": "temperată, fără extreme, aer curat",
            "economie_locuri_munca": "locuri de muncă stabile în educație și servicii",
            "demografie": "oras mediu sau mare, comunitate stabilă",
            "transport": "transport public bun, drumuri naționale",
            "educatie": "școli și licee de calitate, grădinițe",
            "sanatate": "spital pediatric sau spital general",
            "cultura_turism": "parcuri, zone verzi, activități pentru copii",
            "cost_viata": "prețuri moderate la locuințe",
        },
    },
    "pensionar_activ": {
        "description": (
            "Pensionar de 65 de ani, fost inginer. "
            "Sănătos, activ, iubește natura și liniștea. "
            "Pensie medie, vrea costuri reduse și un climat bland."
        ),
        "preference_vector": {
            "clima": "bland, cald vara, iarnă suportabilă, fără poluare",
            "economie_locuri_munca": "nu relevant",
            "demografie": "oras mic sau mediu, max 100k, comunitate prietenoasă",
            "transport": "transport public sau tren pentru deplasări ocazionale",
            "educatie": "nu relevant",
            "sanatate": "spital, policlinică, ambulanță",
            "cultura_turism": "natură, drumeții ușoare, viață culturală moderată",
            "cost_viata": "foarte accesibil, costuri mici la utilități și alimente",
        },
    },
    "antreprenor_mic": {
        "description": (
            "Antreprenor de 42 de ani, deține un mic business în retail și servicii. "
            "Caută un oraș cu trafic comercial, acces rutier bun și costuri operaționale mai mici decât în București."
        ),
        "preference_vector": {
            "clima": "nu prioritar",
            "economie_locuri_munca": "economie locală activă, comerț, servicii",
            "demografie": "oras mediu sau mare, minim 60k locuitori",
            "transport": "drumuri naționale și europene, autostradă preferabil",
            "educatie": "nu prioritar",
            "sanatate": "spital disponibil",
            "cultura_turism": "nu prioritar",
            "cost_viata": "chirie spațiu comercial accesibilă",
        },
    },
    "student_masterat": {
        "description": (
            "Student la master, 24 de ani, domeniul informatică. "
            "Buget limitat, caută un oraș cu universitate bună, viață studențească, "
            "chirii mici și transport ieftin."
        ),
        "preference_vector": {
            "clima": "nu prioritar",
            "economie_locuri_munca": "part-time IT, internshipuri",
            "demografie": "oras universitar, min 80k",
            "transport": "transport urban bun, tren ieftin",
            "educatie": "universitate de stat cu profil tehnic sau economic",
            "sanatate": "policlinică sau spital studențesc",
            "cultura_turism": "viață studențească, cluburi, evenimente culturale",
            "cost_viata": "chirii foarte mici, transport subvenționat",
        },
    },
    "medic_specialist": {
        "description": (
            "Medic specialist în chirurgie, 38 de ani. "
            "Caută un spital de urgență sau un spital universitar unde să-și continue cariera. "
            "Vrea un oraș cu infrastructură medicală bună și posibilitate de avansare profesională."
        ),
        "preference_vector": {
            "clima": "nu prioritar",
            "economie_locuri_munca": "spital universitar sau de urgență, carieră medicală",
            "demografie": "oras mare sau mediu-mare",
            "transport": "aeroport sau tren rapid pentru conferințe",
            "educatie": "facultate de medicină sau UMF",
            "sanatate": "spital de urgență, spital universitar",
            "cultura_turism": "nu prioritar",
            "cost_viata": "moderat",
        },
    },
    "profesor_universitar": {
        "description": (
            "Profesor universitar de 50 de ani, domeniu: filologie și literatură română. "
            "Caută un oraș cu o universitate activă, viață culturală bogată și "
            "o comunitate academică bine dezvoltată."
        ),
        "preference_vector": {
            "clima": "nu prioritar",
            "economie_locuri_munca": "universitate cu profil umanist",
            "demografie": "oras universitar sau cultural",
            "transport": "tren sau conexiuni bune",
            "educatie": "universitate de stat, biblioteci, institute de cercetare",
            "sanatate": "spital disponibil",
            "cultura_turism": "teatre, muzee, festivaluri culturale, viață academică",
            "cost_viata": "moderat",
        },
    },
    "lucrator_industrie": {
        "description": (
            "Muncitor calificat de 33 de ani, experiență în sudură și producție industrială. "
            "Caută un loc de muncă stabil în industrie sau producție, cu salariu decent și cost de trai accesibil."
        ),
        "preference_vector": {
            "clima": "nu prioritar",
            "economie_locuri_munca": "fabrici, industrie auto, producție, construcții",
            "demografie": "oras mediu cu industrie activă",
            "transport": "drumuri naționale, transport spre zona industrială",
            "educatie": "nu prioritar",
            "sanatate": "spital disponibil",
            "cultura_turism": "nu prioritar",
            "cost_viata": "accesibil, chirii mici",
        },
    },
    "roman_intors_strainatate": {
        "description": (
            "Român de 45 de ani, s-a întors din Germania după 12 ani. "
            "Vrea să se reinstaleze, caută un echilibru între calitatea vieții occidentale "
            "și costurile românești. Lucrează remote pentru o firmă germană."
        ),
        "preference_vector": {
            "clima": "bland, aer curat, preferabil climat continental moderat",
            "economie_locuri_munca": "remote-friendly, acces internet bun",
            "demografie": "oras civilizat, nu aglomerat, comunitate mixtă sau cultă",
            "transport": "aeroport sau tren rapid spre vest",
            "educatie": "nu prioritar",
            "sanatate": "spital modern, policlinică privată",
            "cultura_turism": "patrimoniu cultural, gastronomie, natură",
            "cost_viata": "sub vest-european, dar calitate bună",
        },
    },
    "cuplu_fara_copii": {
        "description": (
            "Cuplu fără copii, 32 și 30 de ani. Ea lucrează în marketing, el în arhitectură. "
            "Caută un oraș cu viață activă, restaurante bune, natură accesibilă și "
            "un cartier interesant de locuit. Nu vor un oraș prea mare."
        ),
        "preference_vector": {
            "clima": "plăcut, veri calde dar nu caniculare, natură accesibilă",
            "economie_locuri_munca": "diverse, servicii creative și tehnice",
            "demografie": "oras mediu, 80k-250k, dinamic",
            "transport": "tren sau aeroport pentru vacanțe, mașina personală",
            "educatie": "nu prioritar",
            "sanatate": "spital disponibil",
            "cultura_turism": "restaurante, cafenele, expoziții, hiking",
            "cost_viata": "accesibil dar nu cel mai ieftin",
        },
    },
    "lucrator_remote": {
        "description": (
            "Freelancer de 29 de ani, lucrează în design grafic și video pentru clienți internaționali. "
            "Are nevoie de internet bun, costuri mici și natură în apropiere. "
            "Nu este legat de niciun angajator local."
        ),
        "preference_vector": {
            "clima": "aproape de munte sau natură, aer curat",
            "economie_locuri_munca": "nu relevant local, important acces internet",
            "demografie": "oras mic sau mediu, liniștit",
            "transport": "nu neapărat frecvent, dar existent",
            "educatie": "nu relevant",
            "sanatate": "spital sau policlinică disponibilă",
            "cultura_turism": "natură, trasee, cafenele, liniște",
            "cost_viata": "chirii foarte mici, costuri minime",
        },
    },
    "refugiat_intern": {
        "description": (
            "Persoană de 38 de ani din mediu rural (sat mic, fără servicii), "
            "se mută la oraș pentru prima dată. "
            "Caută un loc cu servicii de bază, loc de muncă disponibil și un start nou."
        ),
        "preference_vector": {
            "clima": "nu prioritar",
            "economie_locuri_munca": "locuri de muncă accesibile, nu necesită calificări înalte",
            "demografie": "oras mic sau mediu, nu intimidant",
            "transport": "transport urban de bază",
            "educatie": "acces la recalificare sau cursuri",
            "sanatate": "spital sau ambulanță disponibilă",
            "cultura_turism": "nu prioritar",
            "cost_viata": "cel mai accesibil posibil",
        },
    },
}

# ---------------------------------------------------------------------------
# Generation system prompt (English meta-instructions, Romanian output)
# ---------------------------------------------------------------------------
GENERATION_SYSTEM_PROMPT = """\
You are generating high-quality training data for a Romanian city recommendation chatbot.

Your task: produce ONE complete multi-turn conversation in ROMANIAN between:
- USER: a person with the described profile who wants to relocate in Romania
- ASSISTANT: a knowledgeable Romanian city recommendation advisor

CRITICAL RULES:
1. ALL conversation text must be written in ROMANIAN (use correct Romanian diacritics: ă, â, î, ș, ț).
2. The assistant must NOT know or reveal any city facts from its own knowledge.
   City facts are only available from the [CONTEXT RAG] block provided.
3. The conversation has two phases:
   PHASE 1 (turns 1-3): The assistant asks clarifying questions to gather preferences.
     - NO city names or facts during this phase.
     - Cover dimensions: climă, economie, demografie, transport, educație, sănătate, cultură, cost.
   PHASE 2 (final turn): After the [CONTEXT RAG] block is injected, the assistant presents
     ONLY the information from that block. The assistant must:
     - Cite context with phrases like "conform datelor furnizate", "datele indică", "pe baza informațiilor disponibile"
     - Rank cities by fit to the stated preferences
     - Explicitly disclaim what data is NOT available (e.g., exact rent prices, current job listings)
4. The final assistant recommendation turn should be detailed and helpful.
5. Do NOT invent statistics, percentages, or facts not present in the CITY DATA block.

OUTPUT FORMAT — return a JSON object with EXACTLY this structure:
{
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "system", "content": "[CONTEXT RAG]\\n...city data here...\\n[END CONTEXT RAG]"},
    {"role": "assistant", "content": "...final grounded recommendation..."}
  ],
  "metadata": {
    "persona_type": "...",
    "cities_referenced": ["city_key_1", "city_key_2"],
    "turn_count": 7,
    "recommendation_given": true
  }
}

The [CONTEXT RAG] entry must be a system message placed immediately before the final assistant turn.
"""

# ---------------------------------------------------------------------------
# User prompt template (filled per scenario)
# ---------------------------------------------------------------------------
GENERATION_USER_PROMPT_TEMPLATE = """\
PERSONA:
{persona_description}

STATED PREFERENCES (use these to guide what the assistant should ask about and how to rank cities):
{preference_summary}

CITY DATA (these are the ONLY cities available — place this as [CONTEXT RAG] in the conversation):
{city_cards}

Generate the conversation now. Remember: all dialogue text in ROMANIAN, no invented facts.
"""
