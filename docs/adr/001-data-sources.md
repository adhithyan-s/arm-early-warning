# ADR 001: Data Sources Selection

## Status
Accepted

## Context
AMR surveillance data exists across multiple sources: WHO GLASS, ECDC EARS-Net, CDC NARMS, and country-level databases. We need to decide which to use as primary sources.

## Decision
Use **ECDC EARS-Net** as primary (Europe-focused, clean annual format, publicly downloadable CSV) and **World Bank health indicators** as supplementary features (healthcare spend, hospital beds per capita).

## Reasoning
- ECDC data has consistent schema from 2000–present
- World Bank API is free and well-documented
- WHO GLASS is newer (2017+) — too short a time window for trend modeling
- CDC NARMS is US-only, limiting generalizability

## Consequences
- European geographic focus limits global claims
- Need to join two different data schemas carefully
- World Bank data lags by 1-2 years — need to handle this in feature 
  engineering