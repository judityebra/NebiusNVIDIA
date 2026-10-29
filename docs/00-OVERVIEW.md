# 00 Overview

Purpose: one-page summary of the hackathon and the project.

## Hackathon
- Name: Nebius x NVIDIA Global AI Hackathon 2026 (from README). Track: Physical AI.
- Organisers: TODO. Status: Unconfirmed
- Challenge statement: TODO. Status: Unconfirmed
- Deadline: 2026-10-30 10:00 PT (17:00 UTC, 18:00 Barcelona), per `docs/SUBMISSION.md`.
- Links (event page, Devpost, Discord): TODO. Status: Unconfirmed

## Goal
Turn warehouse camera frames into a searchable log of safety events and let supervisors query it in plain language, with every claim citing observation ids.

## Pitch
SafeFloor sends each camera frame to a Nemotron vision-language model with the safety rules of that zone and stores structured observations. A Nemotron agent on Nebius Token Factory answers questions through four tools (list zones, search observations, count violations, file incident report), shows its tool trace and files evidence-backed incident reports.
