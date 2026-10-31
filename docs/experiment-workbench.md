---
title: Experiment workbench
description: Upload labeled tables or design synthetic data, review inputs, and compare prediction methods in a localhost browser.
---

## Experiment workbench

Upload your labeled CSV/Parquet, choose saved data, or describe a synthetic prediction
problem. Review the inputs and run a comparison without switching between scripts.

## Upload your own labeled table

Choose **Upload CSV / Parquet** in the local workbench. Effective limits are shown
before upload. Parsing and preview are local, resource-bounded operations; no
planner, provider or prediction method is called.

1. Upload UTF-8 comma-separated CSV (BOM accepted) or Parquet.
2. Inspect the first 20 rows, suggested types and missing counts.
3. Give the dataset a new name; choose its outcome column and classification or
   regression. Drop identifiers/future-only fields and declare categorical predictors.
4. Choose group/time roles when needed and validate random, grouped or temporal
   evaluation. Structural columns are not predictors. Temporal folds do not repeat.
5. Review the validated target summary, manifest and source fingerprint, then
   explicitly confirm saving. Editing any role invalidates that confirmation.
6. Configure methods, review Start and monitor, and reopen saved reports through
   Experiment history.

The source bytes are preserved. Quibble does not repair malformed rows, rename
duplicate headers, remove missing targets or silently shorten training data.
Sample text is shortened only for display. Cancel discards the owned stage;
committed datasets and original files are unaffected. See
[upload limits and provenance](testing-your-data.md#browser-upload) for details.

The optional language-model connection drafts a specification, not prediction results.
Quibble's local generation code creates the rows and outcomes. Existing prediction
methods then train and are evaluated on held-out outcomes. This does not benchmark
ChatGPT as a predictor.

## Appearance and accessible navigation

Open **System → Settings and theme** to choose independent color mode
(System, Light, Dark), accent (Blue, Teal, Indigo, Violet, Fuchsia, Slate), and
contrast (Standard, High). Blue is the default. The native controls and labeled
preview do not change datasets, exact-run approvals, or saved evidence. Verdict
words and colors remain independent of accent; preview badges are not results.
The shared product header provides the same appearance choices on About, Docs,
Blog, and exported static documentation.

Preferences synchronize across controls and browser tabs. Cleared storage restores
defaults; invalid preferences use validated defaults with a warning. If storage is
blocked or full, changes still work for the current page and a visible warning
explains that they may not be saved. System mode follows OS changes; operating-system
forced colors and reduced motion remain supported.

On desktop, **Compact navigation** toggles the labeled sidebar into an icon rail.
Only that bounded preference is saved; temporarily opening a group is not. Focus
or hover shows a supplementary label. Activate a group with Enter or Space, Tab to
its links, and press Escape to close it and restore focus. The icons are original
local SVG shapes dedicated under CC0-1.0; the notice is in the packaged HTML.
At narrow widths, Menu opens the full labeled drawer. Tab stays inside it; Close
menu or Escape restores focus to Menu and re-enables the background.

## Visual walkthrough

These screens use the current neutral light theme and the bundled fictional
`grocery_sample` dataset. They show a real local comparison with HistGB and LightGBM
on a shared holdout split, not a mockup. A single split does not establish
across-split stability. No planning provider, private
data, or optional model-weight download is needed for this example. Results depend
on the dataset, configuration, and hardware, not real grocery operations.

### Choose and review data

Choose a ready-made dataset, inspect its sample and assumptions, and confirm review.
The saved rows are reused unchanged.

![Current workbench showing the reviewed fictional grocery dataset](assets/screenshots/workbench-choose-data.png)

### Configure the comparison

Choose methods, device policy, and smoke or full evaluation. Full runs require
approval; smoke runs do not establish replacement evidence.

![Methods and settings with HistGB, LightGBM, CPU execution, and a holdout split](assets/screenshots/workbench-configure-run.png)

### Read the evidence

Open the complete saved report, including execution status, requested methods and
saved policy. Scroll through quality, disagreement, checks, failures and limitations;
the visible excerpt is not a replacement approval.

![Saved full-run report inside the sidebar workbench, with execution and policy controls](assets/screenshots/workbench-results.png)

### Navigate, print and download complete evidence

The report's **Complete report sections** links follow every Markdown heading,
including unfamiliar or future sections. They scroll within the full report;
they do not filter evidence into tabs. **Print complete report** includes all report
sections and expands code blocks rather than clipping scrollable content.
**Download Markdown report** preserves the original Markdown, including local
artifact references. Failed and interrupted execution remains explicit, even
when a partial report exists.

### Verify saved evidence without retraining

Open **Results → Verify saved evidence**, choose a saved experiment and select
**Load methods and saved checks**. Choose distinct incumbent/baseline and candidate
methods, then explicitly select **Check swap** or **Check baseline gate**.

* Swap reuses the saved effective acceptance policy and existing scientific swap
  evaluator. It does not silently substitute today's packaged defaults.
* Baseline gate uses the saved primary metric and your nonnegative minimum gain.
  It asks a narrower improvement question, not replacement approval.
* Both require complete full-run evidence for all recorded methods, a matching
  available offline source, fitted recipe provenance, aligned test rows and
  labels, prediction artifacts and invocation coverage. Smoke, interrupted,
  failed, excluded or incomplete evidence is refused, not converted into PASS.
* Checks run in a separate analysis process with a 120-second limit; they do not
  fit predictors, download model weights, call a planning provider or change
  the original report, policy or model records. Wait for active training to finish.

Each executed check is saved separately under
`results/web/<job-id>/verifications/<check-id>.json`, including its inputs,
coverage, policy provenance and input digest. Reopen saved checks or download
the selected JSON. Scientific outcomes remain **PASS**, **FAIL** or
**INCONCLUSIVE**; process errors are **ERROR**, with no scientific verdict.
PASS is not a production-safety guarantee. The saved effective policy is supplied
explicitly during replay, so the evaluator's policy-origin flags describe the
replay; the artifact's `policy_provenance` describes the original selection.

### Compare exactly two saved experiments

Open **Results → Compare two experiments**, select two different saved jobs and
explicitly compare. The side-by-side evidence includes recorded configuration,
devices, method cells, metrics, coverage and policies. Compatibility checks
source fingerprints, task, actual saved test rows/labels/splits and recipes;
matching dataset names or fold counts alone are insufficient.

This surface is **descriptive only**, even when compatibility is **MATCHED**.
It never declares a winner, scientific equivalence or paired statistical verdict.
Different policies, recipes, execution settings or coverage remain visible.
Different sources or targets restrict presentation to configuration and report
links; shared quality ranking is withheld. Missing source provenance is
**UNKNOWN**, not approval. Supported defaults-versus-tuned tree cells additionally
reuse the existing cross-tier pairing validator without inventing a paired verdict.
The JSON download records the compatibility contract and caveats.

### Explicit diagnostics

**System → Backends and verification** shows the existing backend registry and
verification metadata. **Collect local package diagnostics** is a separate,
explicit action, with a timestamp and optional JSON download. Navigation and
Home refresh do not collect diagnostics. Only Python/OS identifiers, allowlisted
package versions and backend metadata are returned: no environment variables,
credentials, private rows, registry keys, cache paths, GPU initialization or
network probes. A missing optional package is unavailable, not an experiment
failure. There are no install, credential-editing or download controls.

## Why generate a fresh dataset

Familiar public benchmark tables may have appeared in a pretrained predictor's training
data. Quibble cannot inspect every predictor's pretraining corpus or prove that a
benchmark is unseen. Newly sampled rows reduce direct reuse of those benchmark examples
and make the input and outcome rules inspectable.

That is an experimental control, not a guarantee of no contamination. A predictor may
already know similar task structures or generating rules. A language-model planner can
also propose familiar assumptions, and a synthetic process may favor some methods.
Changing only a dataset's name does not create new examples; the same specification and
seed reproduce the same rows.

Use the generator to ask controlled questions: what changes when noise increases, an
input disappears, or irrelevant fields are added? Repeat across independently generated
tables and several task structures. Keep development variants separate from confirmatory
evaluation, preserve the recipe and hashes, and test useful findings on representative
real data. Synthetic tests complement real-data evaluation; they do not replace it.

The browser adds an interactive route through these steps. It uses the existing benchmark
runner and report pipeline; CLI commands remain available for scripting and settings
outside the UI's deliberately bounded controls.

## Start locally

Use the existing environment from [Getting started](getting-started.md#install):

```powershell
& .\.venv\Scripts\Activate.ps1
python -m quibble_ml web --open
```

Open `http://127.0.0.1:8765`. Choose another port with `--port 8876`.
The server binds only to `127.0.0.1`, not a network interface.
The default address opens About. Choose **Open workbench** in its **What to do next**
section, or **Workbench** in the shared header, to open `/workbench`.

The prominent **Quibble** heading identifies the product; the smaller subtitle describes
its purpose. Every page shares **Workbench / Blog / About / GitHub / Docs** navigation,
with the current page highlighted. These are product pages, not experiment steps.
Blog, About, and Docs open separately from the workbench so its draft stays intact.
It explains workflow, methods, devices, privacy, limitations, and the running version.
Its **Predicting from Tables blog series** link opens the local reader at `/blog`;
both articles, diagrams, and curated evidence work without public hosting.
Docs and Blog work locally; GitHub links to the source repository.
Version and license details appear in About's content, not the shared header.
Read [About Quibble](about.md) and the [documentation home](README.md) locally meanwhile.

The shared header's **Color theme** control offers System, Light, and Dark. System follows
your device appearance; an explicit choice is saved in this browser and applies across
Workbench, About, Docs, and Blog, including the static documentation site. If browser
storage is unavailable, the selected appearance still works for the current page.
No fonts, theme services, or other appearance assets are fetched from third parties.
Light uses neutral white and gray surfaces; Dark uses charcoal rather than navy.
Both have a flat header and one restrained blue accent for actions and selections.
Green, red, and amber identify result and execution states, not navigation choices.

The layout adapts to narrow screens and honors reduced-motion preferences. Keyboard users
can skip to the main content and follow visible focus indicators. Report verdict badges
make PASS, FAIL, and INCONCLUSIVE easier to scan without changing the report text, checks,
or caveats. A completed experiment status means execution finished, not that a swap passed.

The local Docs reader uses a documentation-site layout: grouped sidebar navigation,
the current guide highlighted, breadcrumbs, an **On this page** section index, and
Previous/Next guide links. Blog articles use the same layout and shared product header,
but their sidebar and Previous/Next links stay within the blog series, not the docs.
On narrow screens the navigation stacks above the article and can be collapsed using
**Documentation navigation**. Heading links work without JavaScript.

Circled `i` icons beside fields and prediction methods explain their purpose, limits,
and trade-offs. Hover or focus an icon with the keyboard to read its help; click or tap
to keep it open. Press Escape, tap the icon again, or click elsewhere to dismiss it.

The unified **Dataset** selector includes already-created grocery and retail clothing
tables without an account, provider call, or regeneration. Both domains have binary
classification and regression examples, each with 1,000 rows and eight predictors.
Saved recipes, input hashes, and explanations accompany all four tables.

### One review flow, three experiment steps

Home offers Start experiment, upload, recent browser jobs and a small setup summary.
A missing synthetic-planning provider does not prevent existing-data or upload work.
The sidebar groups Data, Run experiment, Results, System and Help. On narrow screens,
Menu opens a full labeled drawer with keyboard focus containment; Escape closes it
and returns focus. Public Docs, Blog and About remain separate reading pages.

Run experiment has three steps:
**Choose data -> Methods and settings -> Start and monitor**.
Data and Choose data share the same review controls, not separate drafts.
Start at Choose data:

- **Use existing dataset** shows ready-made and other saved datasets. Choose one,
  inspect its actual rows and assumptions, confirm review, then choose
  **Continue to experiment**. No specification creation is required.
- **Design new dataset** shows the scenario and recipe editor, not the existing-data
  controls. Propose a design, preview, confirm review, then choose
  **Save and continue** to save the dataset and open experiment configuration.
- **Upload CSV / Parquet** stages a local file, requires explicit column-role and
  split review, then saves the unchanged source before continuing.

Preview generation stays disabled after a successful preview until the recipe changes.
An unchanged saved recipe also needs no generation: discard the variant to use its
existing rows, or change the name or recipe to enable a new preview.

Methods and settings shows the chosen dataset as a summary, not another picker.
Use Change dataset to return to review. Next opens Start and monitor without
training anything. The matching top and bottom action bars use the same validation
and handlers, with visible explanations when an action is disabled.

Start and monitor summarizes the request. Full requires explicit approval;
changing data, methods, mode, device, splits or draft policy clears that approval.
Start sends one request; uncertain responses require Refresh status, not an
automatic retry. The global running indicator stays visible on other pages.
Opening an older report cannot redirect monitoring to that older job or let a
heartbeat overwrite the selected report.

Experiment history lists up to 30 recent browser jobs, not the complete CLI
archive. Saved report links and legacy `#job=...` bookmarks reopen without
reviewing new data. Back, Forward and direct page links preserve the in-memory
draft; reloading does not silently restore review approval. Missing jobs and
unknown routes give explicit errors and recovery directions.

Backends and verification shows existing registry metadata, not proof of package
installation. Local diagnostics require the separate explicit collection action.
Settings and theme includes independent mode, accent, contrast and desktop rail
preferences. Results also provides functional saved-evidence Verify and descriptive
two-experiment Compare pages; neither action starts training.

For a new design, enter your own description and choose
**Propose a scenario with my provider**.
Starting a proposal clears the previous recipe and preview, so a failed request cannot
leave an old example looking like the new design. Invalid provider specifications get
one validation-guided correction request, then an explicit error if still invalid.
Both requests may incur provider charges; malformed specifications are never saved.

The selected outcome type takes precedence over scenario wording. For an inventory
coverage yes/no question, select classification. Selecting regression asks for a
continuous synthetic score instead; the planner must disclose that reinterpretation
and its meaning in the assumptions. The score can be negative and is not automatically
demand units or a calibrated coverage probability. Classification assumptions should
explain labels 0 and 1.

Select a ready-made dataset to inspect its saved recipe, sample rows, and collapsed
explanation in the existing-data path. Review and continue; saving or generating it
again is unnecessary.
Changing the outcome filter clears the previous selection and review.
The single dataset list shows only datasets matching its outcome filter, with
classification entries labeled binary or multiclass. Groups identify bundled synthetic
examples, built-in benchmarks, and other saved datasets. They are all usable existing
data, not different readiness levels. Existing local folders override bundled names.
Choose **Design new dataset** for a blank design. Its outcome selector is separate
from the ready-made filter; custom scenario text is preserved on task changes.

The bundled names are `grocery_sample`, `grocery_regression`, `retail_clothing_binary`,
and `retail_clothing_regression`. CLI dataset commands can use them too.
An existing local folder with the same name takes precedence and is never overwritten.
Changing controls does not change saved rows; use **Create variant from this recipe**
to create a reviewed variant. These examples demonstrate task shapes, not calibrated
domain simulations or four independent sources of real-world evidence.
The existing-data path hides creation controls instead of showing an unrelated editor.
Choose **Design new dataset** to design a new dataset. To copy an existing recipe,
expand **Advanced: create a dataset variant** in the saved-data review and choose
**Create variant from this recipe**. The optional action appears only when a recipe exists;
datasets without one do not show variant controls. Built-in benchmarks show a plain
source note instead of empty metadata JSON. Imported datasets show their available
source details without an empty generation entry.
While editing a variant, **Discard changes and continue** sits beside **Save and continue**.
Saving uses the newly saved variant; discarding abandons the unsaved recipe and uses the
original saved rows. Both open experiment configuration without starting a run.
Discarding never modifies or deletes saved files. After saving a variant, the discard
action is no longer shown.

Expand **About this dataset** after a proposal, preview, or save to read its scenario,
target interpretation, input descriptions, and assumptions. This explanation comes
from the current recipe, not another provider request. Editing the recipe hides the
old explanation until a validated preview refreshes it. Reviewing a saved generated
dataset also provides a collapsed explanation from its saved recipe.

The `web` extra includes the planning/authentication SDKs. Select it with
`uv sync --locked --extra web --extra data`, retaining any wanted model extras.
Local generation and the local example do not call those SDKs.

## Configure your planning provider

Copy the template only if you do not already have settings:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

The planner supports the OpenAI-compatible Chat Completions protocol with JSON-object
responses. Choose a provider, its inference base URL, a model/deployment name, and
authentication. Azure Foundry is one option, not a requirement.

Restart the workbench after editing `.env`. The settings are server-side; credentials
are not entered into or stored in the browser.

| Provider | Example base URL | Model setting | Authentication |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1/` | Your available model identifier | `api_key` |
| Azure OpenAI / compatible Foundry inference | `https://YOUR-RESOURCE.openai.azure.com/openai/v1/` | Your actual deployment name | `aad` or `api_key` |
| Other compatible hosted service | Its documented OpenAI-compatible inference URL | Its available model identifier | `api_key` |
| Local compatible service | `http://127.0.0.1:11434/v1/` | Your installed local model identifier | `none`, or `api_key` if configured locally |

These are configuration examples, not claims that every service/model has been tested.
Native Anthropic, Gemini, Bedrock, or other non-compatible protocols need an adapter or
compatible gateway; changing a URL does not translate an incompatible API. Your chosen
model must support this protocol and JSON-object output. No automatic provider fallback occurs.

### OpenAI or another compatible hosted service

Set these entries in `.env`:

```dotenv
LLM_BASE_URL=https://api.openai.com/v1/
LLM_MODEL=YOUR-MODEL
LLM_AUTH_MODE=api_key
LLM_API_KEY=YOUR-PROVIDER-KEY
LLM_PROVIDER_NAME=My planning provider
```

For another compatible service, replace the URL, model, and provider-specific key.
Remote endpoints require HTTPS; URL credentials, query strings, and fragments are
rejected. Do not put keys in URLs.

Standard `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY` names also work;
the base URL defaults to OpenAI when no endpoint is configured. The `LLM_*` settings
take precedence. A custom `LLM_BASE_URL` outside `api.openai.com` requires its own
`LLM_API_KEY`: it never implicitly receives an unrelated OpenAI or old Azure key.
When using aliases for a non-OpenAI compatible service, explicitly pair
`OPENAI_BASE_URL` with that service's `OPENAI_API_KEY`.

### Microsoft Entra ID

For an Azure deployment, configure:

```dotenv
LLM_BASE_URL=https://YOUR-RESOURCE.openai.azure.com/openai/v1/
LLM_MODEL=YOUR-DEPLOYMENT
LLM_AUTH_MODE=aad
LLM_AAD_CREDENTIAL=cli
LLM_TOKEN_SCOPE=https://ai.azure.com/.default
AZURE_TENANT_ID=
AZURE_SUBSCRIPTION_ID=
```

Use the inference base URL, not a Foundry portal or `/api/projects/...` management URL.
An Azure deployment name is not necessarily the model family's name.
Install Azure CLI if needed and sign in explicitly:

```powershell
python -m quibble_ml auth
python -m quibble_ml auth --check
python -m quibble_ml web --open
```

For browserless/device-code login use `python -m quibble_ml auth --device-code`.
Alternatively, `python -m quibble_ml web --login --open` combines explicit login and
startup; add `--device-code` if required. Optional tenant and subscription settings
select the intended account context.

`cli` uses `AzureCliCredential`. `default` uses noninteractive
`DefaultAzureCredential`, allowing an existing environment or managed-identity setup.
The `auth` login helper is only for `cli`; for `default`, establish credentials
externally and use `auth --check`. Token acquisition does not prove that inference
authorization works. The identity also needs the inference role appropriate to the
resource, such as Cognitive Services OpenAI User for an Azure OpenAI resource.
Follow your organization's role-assignment policy.

### Azure API key

Keep the Azure URL/deployment and use this instead of AAD if your resource permits key access:

```dotenv
LLM_AUTH_MODE=api_key
LLM_API_KEY=YOUR-RESOURCE-KEY
```

Do not commit keys. `auth --check` in key mode checks configuration presence, not a
live request. Planning errors are surfaced without printing provider response bodies
or credentials. Real environment variables take precedence over `.env`.

### Local compatible service

Start and configure the local service separately, including any model download:

```dotenv
LLM_BASE_URL=http://127.0.0.1:11434/v1/
LLM_MODEL=YOUR-INSTALLED-MODEL
LLM_AUTH_MODE=none
LLM_PROVIDER_NAME=Local service
```

HTTP and unauthenticated mode are restricted to loopback addresses. In `none` mode,
the SDK uses a non-secret placeholder, never a stored OpenAI/Azure key. Entra tokens
are not sent to loopback services. A local service must still implement the required
API/output shape; Quibble does not install or start it.

Older `FOUNDRY_PROJECT_ENDPOINT`, `AZURE_AI_MODEL_DEPLOYMENT_NAME`,
`FOUNDRY_AUTH_MODE`, and `FOUNDRY_API_KEY` configurations remain supported when no
generic/OpenAI endpoint is set. They retain the old AAD default; `key` maps to
`api_key`. Prefer the provider-neutral names for new configuration.

See Microsoft's [Azure v1 API guide](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle)
and [keyless authentication guidance](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/managed-identity).
Mocked OpenAI/custom-provider/AAD/local-service contracts have automated coverage; live provider access must be
verified with your own configuration.

## Design and review

1. Describe the prediction decision, what a row represents, what information is available
   at prediction time, and what outcome should be predicted. Do not enter sensitive data.
2. Set rows, predictor fields, task, and generation seed. The target is an additional
   column: 8 predictors means 9 total columns.
3. For an arbitrary natural-language scenario, explicitly consent to sending that text
   and controls to your configured provider, then choose **Propose a scenario with my provider**. This may incur provider
   charges. No existing dataset rows, private references, or local documents are sent.
4. Review the JSON specification, particularly assumptions, timing, field ranges,
   weights, interactions, noise, and the task. The draft is a hypothesis, not a factual
   description of the industry.
5. Preview. Quibble generates the whole table, displays the first 20 rows, and summarizes
   target distribution and missing values.
6. Edit if needed and preview again. Confirm review and choose **Save and continue**. Saving requires the
   current specification's preview from this server session; edits invalidate it.
7. Configuration opens with the saved dataset already chosen. Choose prediction methods
   and device policy. Return to **Choose data** if you need the saved CSV download.
8. Run a smoke check, inspect failures and the interpreted report, then explicitly
   confirm a full comparison if appropriate.

The grocery example is fictional. It uses independent decisions rather than repeated
store histories. Do not treat its results as real demand forecasts or deployment evidence.

### Names, saved datasets, and variants

Ready-made datasets already have names; the planning provider proposes a name for
custom scenarios. For a new design or editable variant, change **New dataset name** without editing JSON, or edit the JSON's
`name` directly. Both stay synchronized. Changing the name requires another preview
and review confirmation. Names must pass the generator's portable-name validation.

To reuse data, select it in **Dataset**, then choose **Review selected dataset**.
The browser shows the actual saved first 20 rows, counts, outcome summary, missingness,
and available source/generation metadata. Confirm review and choose **Continue to experiment**
without generating or modifying rows. Source fingerprints are rechecked on confirmation;
if the source changes, review it again. Existing benchmark validation still applies.

For generated data, **Create variant from this recipe** under **Advanced: create a dataset variant** loads the saved recipe into
the editor. Choose a new unused name, make any changes, preview, confirm, and choose
**Save and continue**, or choose **Discard changes and continue** to use the original.
This generates a new table from the reviewed recipe; it does not rename or overwrite
the original. Provenance identifies the original source fingerprint. Imported tables
and built-in datasets without generation metadata can be used as-is, but cannot be
regenerated through a missing recipe.

There is no overwrite operation. Existing data and previous experiment evidence remain
unchanged. Selecting a different dataset clears the current review.

## Specification and generation mathematics

Version 1 accepts only known schema keys. Start from the UI or
`python -m quibble_ml generate --example`; unsupported keys or executable code are rejected.

| Section | Required content |
|---|---|
| Identity | `version: 1`, safe unique `name`, `scenario`, `task`, `target`, `rows`, `seed` |
| Timing | `row_unit`, `prediction_time`, `outcome_horizon`, nonempty `assumptions` |
| Features | Exact predictor list; each has `name`, `description`, `distribution`, `minimum`, `maximum`, `categories`, `effect`, `weight`, `missing_rate` |
| Outcome | `intercept`, `scale`, `noise` |
| Interactions | A list of `{left, right, weight}` feature-product terms; empty is valid |

Supported feature distributions:

- `uniform`: continuous samples over the chosen bounds.
- `integer`: equally likely integers, inclusive bounds.
- `normal`: midpoint mean, standard deviation one sixth of the range, clipped to bounds.
- `categorical`: equally likely listed strings, at least two unique categories. Numeric
  bounds are placeholders; category order encodes evenly spaced values from -1 to 1.
  Reordering categories changes their outcome effects.

All features are independently sampled. This version does not simulate correlated
inputs, time series, panel data, exact category proportions, or arbitrary business rules.
Numeric features are rescaled from their bounds to -1 through 1.

Effects transform each encoded value: `linear` leaves it unchanged, `quadratic` squares
it, `threshold` tests whether it exceeds zero, and `none` contributes zero (weight
must also be zero). A weighted sum plus the listed products forms a score. The latent
outcome is `intercept + scale * score + Normal(0, noise)`.

- Regression uses that latent outcome directly. It is continuous and may be negative;
  it does not promise nonnegative integer demand or bounded business quantities.
- Binary classification samples a Bernoulli outcome with logistic probability of the
  latent value. Even zero Gaussian noise does not remove Bernoulli randomness.

Feature missingness is independently applied *after* outcome generation. Hidden
pre-missing values, latent outcomes, and oracle probabilities are not predictor columns.
This separation prevents exposing generator internals as features; it does not
automatically detect semantic leakage in arbitrary datasets or validate domain realism.

Limits are 50 to 50,000 rows, 1 to 100 predictors, at most 2,000,000 predictor cells,
and seeds from 0 to 4,294,967,295. Missingness is limited to 0.5 per field.
Every generated classification class must realize at least 10 rows; regression must
be nonconstant. Violations fail explicitly instead of silently resampling or changing
requested counts. Balance is stochastic, not guaranteed exact.

## Files and reproducibility

Saved datasets use `QUIBBLE_DATA_DIR` (default `data`):

```text
data/<dataset-name>/
  data.csv          generated table, target last
  dataset.json      existing local-data loader manifest
  generation.json   reviewed specification, seed, generator version, hash, package versions
```

Provider-assisted records additionally preserve endpoint, requested/reported model, response
identifier, token usage when provided, and planning instructions. Edits are represented
by the saved final specification; these fields identify the drafting request, not
provider approval of subsequent edits. Preserve all three files. Generation metadata
is included in source fingerprints; reports label the experiment synthetic. Changing
the CSV without its generation record is rejected. Save a newly designed variant
under a new name. Existing folders are never overwritten.

A specification and matching software environment reproduce the local table.
Hosted model drafts are not guaranteed reproducible. Generation seeds differ from
split seeds and prediction-method seeds; the UI currently fixes model seed 0.
Repeat comparisons across fresh generation seeds before generalizing.

For CLI use, save a reviewed example/specification:

```powershell
python -m quibble_ml generate --example
# Save the JSON as reviewed-spec.json using a UTF-8 editor, then:
python -m quibble_ml generate --spec reviewed-spec.json
python -m quibble_ml datasets --verify --names grocery_sample
```

Use the actual saved name. Existing [local-data commands](testing-your-data.md)
can evaluate the same dataset.

## Experiments and reports

The UI launches the existing benchmark command in a subprocess, not arbitrary scripts.
Choose one to four public prediction methods and `cpu`, `cuda`, or `auto`. Installing
an SDK does not download all model weights. Selected pretrained methods may need their
own packages, authentication, and [model downloads](datasets.md).

Smoke uses the established 128-training/32-test maximum and is not a swap approval.
Full runs use `holdout`, `3x1`, or `3x2`, fixed model seed 0, and one fit worker.
One experiment can run at a time; each gets an isolated output directory.

For full runs, expand **Swap acceptance policy (full runs)** to inspect the
[packaged defaults](../quibble_ml/swap_thresholds.json). These are the limits Quibble uses,
not universally safe acceptance criteria. Choose **Custom policy** to edit JSON or import
a local JSON file. Omitted keys retain defaults; the preview shows the merged effective limits.
The server rejects invalid policies before training. **Reset to defaults** discards edits,
and **Download effective policy** saves a reusable copy. Smoke runs disable these controls
and do not save or evaluate a swap policy.

Each full job saves its effective settings and default/custom designation. Reopen its
**Saved swap acceptance policy** to inspect or download the exact saved values. A later
run does not automatically inherit them; import the saved JSON or select defaults explicitly.
Reports include all effective settings alongside swap results.
See [Swap checks](swap.md#changing-the-limits) for units and CLI policy options.

Read [what Full does and what the results establish](running-benchmarks.md#what-full-does-and-what-the-results-establish)
for the split, training-only preprocessing, fitting, held-out prediction, analysis, and
reporting sequence. The page also explains why fast `fit()` does not imply fast inference,
and why these results do not establish deployment readiness.

CPU pins all selected methods to CPU. CUDA requires availability; Auto chooses CUDA when
available, otherwise CPU, and pins that choice once per experiment. CUDA selections
disable incompatible methods with explanations: HistGB/Weak have no CUDA implementation,
this LightGBM adapter is CPU-only, and AutoGluon can mix CPU components into its ensemble.
Select at least one enabled method. API/CLI selections are filtered under the same policy,
with exclusions recorded; an empty eligible selection is rejected. Models are not substituted.
Data preparation, metrics, and report generation still run on CPU. GPU failure never
triggers a CPU retry. Existing reports retain their original device records.

```text
results/web/<job-id>/
  job.json
  thresholds.json                         effective policy, full runs only
  run.log
  output/report/report.md                 full report
  output/smoke/<smoke-id>/report/report.md  smoke report
```

The browser shows execution logs, sanitized formatted Markdown with inline PNG figures,
and a Markdown download. The underlying reports retain interpretation, source
fingerprints, participation/failure details, and synthetic limitations. A report can
exist for a failed or partial job; only subprocess exit code 0 marks a job completed.
Download/copy the associated figures as well when moving a Markdown report.

Headings, tables, lists, emphasis, and code blocks render in the browser. Embedded HTML
is treated as text and the generated HTML is sanitized. Only the job's known report
figures load as images; external images are not fetched. External HTTP/HTTPS links and
fragment links are allowed. Other local artifact/history links are inactive in the
browser; use the original downloaded Markdown and results directory for those files.

Jobs and datasets persist on disk. Draft/preview permissions are session-local and
are lost on server restart. Previous unfinished jobs are marked interrupted; restart
does not resume them. There is no cancellation button. Normal shutdown waits for the
active experiment; a forced process termination may leave child work running.

After **Start experiment**, a running indicator shows the dataset, mode, and experiment
ID while the browser checks status every 2.5 seconds. Another experiment cannot start
while one is active. The indicator is an activity signal, not a percentage or time estimate.
Open **Execution log** to see completed fits and diagnostics.

The report dropdown starts with no experiment selected and an empty report viewer.
Choose an experiment to load its status and report. Starting a new experiment or
opening a bookmarked experiment link selects that specific run automatically.
Choosing the empty dropdown entry clears the report details again.

You can close the browser tab without stopping the server-side experiment. Keep the
Quibble terminal/server running. Bookmark the experiment link under the result status,
or return to the same web address and select the dataset/mode/experiment ID in
**Current and previous experiments**. The browser reconnects to running work and shows
the saved report when it finishes. The saved-files path is displayed alongside the link.
After a server restart, use the same output directory and port/address to reopen completed
reports. Unfinished jobs are marked interrupted, not automatically resumed. If the
status connection is lost, the UI says so rather than claiming completion; use Refresh
after restoring the server connection.

On Windows, keep data/output roots and dataset names short. Deeply nested working
directories can exceed filesystem path limits during dated result snapshots.
Select a shorter `QUIBBLE_OUT_DIR` before starting the server if needed; such failures
remain failed jobs with diagnostic logs, not completed comparisons.

## Local-only boundary

Host checks, same-origin POST checks, a per-start request token, bounded JSON, fixed
routes, and safe text rendering reduce accidental browser-to-localhost access.
The request token is not user authentication. Do not expose this port through a proxy,
tunnel, shared machine, or public network. Local processes and users can access the
same filesystem. One workbench instance owns each output root.

Deployment would require authentication, authorization, tenant/job isolation, quotas,
secret management, and controlled storage. Those are not provided here.
