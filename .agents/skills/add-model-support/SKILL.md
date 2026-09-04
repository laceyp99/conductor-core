---
name: add-model-support
description: Ensure the LLMs mentioned from an existing Conductor Core provider are represented accurately. Trigger when asked to "add support for" a given model name or "update model metadata" (whether its reasoning controls, cost, or rate limits). Use this as task guidance, defer to global skills for behavioral rules.
---

# Add Model Support

Use this skill when a provider already supported by Conductor Core announces a new model and you need to wire it into the package safely, or even to capture any post-release updates to ensure a model stays represented correctly.

This workflow is for existing providers only. Do not use it to add a brand-new provider module or broad evaluation coverage.

## Inputs

- Model identifier
- Provider name (Optional)
- Official source URLs (Optional)

If the user does not provide a provider or URL, search the web first and prefer official release notes, pricing pages, model docs, and API references. Delegate to subagents for context heavy research tasks using the `explorer` subagent.

## What This Skill Produces

- A researched update to the packaged `model_list.json` for the new model.
- Any minimum compatibility changes required in `conductor_core.providers`.
- A reported note of any minimum downstream changes required for Conductor Main controls or Eval model selection.
- A summary of sources, assumptions, touched files, and validation results.

## Procedure

1. Confirm scope before editing.
   - Only proceed if the provider already exists in this repo.
   - If the request actually requires a new provider, stop and ask for a confirmation after estimating the work required to implement.

2. Research the model from official sources.
   - Prefer vendor docs over third-party summaries.
   - Capture the public model identifier, pricing, context or max token limits, published rate limits, and any request-parameter constraints relevant to this repo.
   - Specifically determine whether the model supports or restricts temperature, extended thinking, effort-style reasoning controls, or always-on adaptive thinking.
   - If official data is incomplete, do not invent values. Record the gap and ask for maintainer direction if the missing field blocks a safe edit.

3. Update the model registry.
   - Edit `src/conductor_core/resources/model_list.json` under the existing provider key.
   - Preserve the current schema and nearby provider conventions.
   - Add `extended_thinking`, `always_on_adaptive_thinking`, `effort_options` when applicable, `max_tokens`, `cost`, and `rate_limits` only from supported evidence.

4. Verify the provider module.
   - Inspect the provider module and check whether the new model works with the current request construction, parameter names, parsing path, and cost calculation.
   - Make the smallest provider-side change needed.
   - Keep changes local to the provider unless a real compatibility constraint forces a nearby adjustment.

6. Validate immediately after the first substantive edit.
   - Prefer a focused syntax, import, or error check for the touched files.
   - If there is no narrow executable check, use the most local validation available and report what remains manual.
   - Do not run the long evaluation scripts for this workflow.

7. Report the outcome.
   - Summarize the official sources used.
   - List the fields added or changed in the packaged `model_list.json`.
   - Report downstream checks as follow-up work instead of assuming sibling repositories exist.
   - State assumptions, missing vendor details, and validation status.

## Project Notes

Use the repo-specific guide at [repo integration notes](./references/repo-integration.md) for the current file map, schema expectations, and app/provider control points.

## Completion Criteria

- The model is present under the correct provider in `src/conductor_core/resources/model_list.json`.
- The matching provider module still uses valid request parameters for that model.
- Available downstream clients expose only controls the model supports.
- The final response includes sources, assumptions, touched files, and validation results.
