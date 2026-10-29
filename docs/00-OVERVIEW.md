# 00 Overview

Purpose: one-page summary of the hackathon and the project.

## Hackathon
- Name: Nebius x NVIDIA Global AI Hackathon 2026 (from README). Track: Physical AI.
- Organisers: Nebius, funded by NVIDIA; judges from NVIDIA, Nebius and Tavily (third-party listing).
- Challenge: build a working AI system on Nebius AI Cloud or Token Factory using at least one NVIDIA open model (e.g. Nemotron). See `01-RULES-AND-CRITERIA.md`.
- Deadline: 2026-10-30 10:00 PDT (17:00 UTC, 18:00 Barcelona).
- Links: https://nebiusglobalaihackathon.devpost.com/ (Devpost). Discord: TODO. Status: Unconfirmed

## Goal
Turn warehouse camera frames into a searchable log of safety events and let supervisors query it in plain language, with every claim citing observation ids.

## Pitch
SafeFloor sends each camera frame to a Nemotron vision-language model with the safety rules of that zone and stores structured observations. A Nemotron agent on Nebius Token Factory answers questions through four tools (list zones, search observations, count violations, file incident report), shows its tool trace and files evidence-backed incident reports.
