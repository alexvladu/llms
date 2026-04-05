# RFC: AI-Powered Conversational City Recommendation System
## Technical Workpaper

### Executive Summary

This workpaper proposes the design and development of an intelligent conversational chatbot system utilizing a fine-tuned Small Language Model (SLM) to provide personalized city relocation recommendations. The system systematically elicits user preferences through natural language dialogues across eight standardized evaluation dimensions, then applies a multi-criteria decision analysis framework to generate ranked city suggestions with explainable reasoning. The architecture emphasizes progressive preference aggregation, enabling the system to transition autonomously from exploratory dialogue to recommendation generation upon achieving requisite preference data density.

### Overview

The proposed system comprises three integrated components: (1) a conversational preference elicitation engine powered by a fine-tuned SLM, (2) a normalized city evaluation framework, and (3) an intelligent dialogue state management system. Users interact through natural language queries, allowing the system to infer both explicit and implicit preference signals across eight distinct evaluation criteria: professional opportunity assessment (job market, salary expectations), climatic and environmental conditions (weather patterns, air quality), socioeconomic factors (cost of living, taxation regime), infrastructure quality (urban congestion, public transit), and cultural/linguistic alignment (language prevalence, community interests).

The recommendation engine applies weighted aggregation across standardized city attribute indices to generate preference-aligned rankings. As dialogue progresses and preference certainty thresholds are satisfied, the system transitions to active recommendation mode, presenting multiple candidate cities.

### Motivation & Problem Statement

Picking a city to move to is tough. You need to juggle jobs, weather, money, pollution, cost of living, taxes, and whether people speak your language. Right now, you're bouncing between job sites, weather apps, real estate platforms, and Wikipedia just to compare a few cities. The information is scattered everywhere, it's hard to compare things fairly, and you often end up frustrated with your choice.

The proposed conversational AI approach addresses this fragmentation through several mechanisms: (1) **unified preference elicitation** via natural language interaction reduces user friction compared to structured questionnaires; (2) **contextual inference capabilities** enable the system to extrapolate implicit preferences and detect trade-off patterns; (3) **progressive decision support** delivers increasingly refined recommendations as preference data accumulates; (4) **deployment efficiency** through SLM architecture enables cost-effective operation at scale while maintaining inference latency suitable for real-time dialogue interaction.

The SLM-based approach provides optimal balance between model expressiveness and computational efficiency, enabling on-premises deployment without substantial GPU infrastructure requirements while maintaining dialogue naturalness and preference understanding equivalent to larger foundation models.

### Objectives & Specifications

**What We Want to Build:**

1. **Smart Conversation** – A chatbot that asks the right questions, remembers what you said, and figures out when it has enough info to recommend cities

2. **Fair City Scoring** – A way to compare cities fairly using the same factors for each one (jobs, weather, cost, air quality, etc.), then rank them based on what matters to you

3. **Know When to Stop Asking** – The system should realize when it's gathered enough information and jump straight to recommendations instead of asking pointless follow-up questions

4. **Cover All the Important Stuff** – Jobs, salaries, weather, air quality, traffic, taxes, language, and things to do

**How We'll Know It Works:**
- 80%+ of users think our top recommendation is actually good for them
- The chatbot understands what you want (80%+ accuracy)
- You get recommendations after about 3 questions, not 20
- We have info on 500+ cities around the world
- It responds in under 2 seconds per message

**Later On (Phase 2):**
- Real-time job listings and housing prices
- Help families decide together (not just one person)
- Mobile app
- Smarter comparison features

---
**Document Classification:** Technical Proposal  
**Status:** Request for Comment (RFC)  
**Version:** 1.0  
**Issue Date:** March 18, 2026  
**Format:** Technical Workpaper (one page)
