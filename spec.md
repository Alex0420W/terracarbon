# Hackathon Build Spec — 22-Hour Window

Two prompts, two builds, one team. This document is the reference spec for both. Read once end-to-end before the hackathon, then use as a lookup during the build.

Design principles across both builds:

- **One unforgettable visual moment per build.** Everything else is scaffolding around producing it.
- **Real methodology under the hood.** Every visual claim must be defensible in Q&A with domain vocabulary.
- **Abstract data ingestion.** Input format is unknown until the event; everything downstream must operate on canonical internal schemas so a thin adapter layer is the only thing that changes when data arrives.
- **Ship-by-hour-14 rule.** The end-to-end demo path must work by hour 14, even if ugly. Polish happens after. Stop coding at hour 21.

---

# PART 1 — LAND USE & SUSTAINABILITY

## 1.1 Winning Thesis

Build a tool that classifies land use from imagery, detects year-over-year transitions, and produces IPCC-aligned GHG emission estimates with honest uncertainty — visualized through an interactive time-slider reveal that makes carbon emission from land-use change physically legible.

**Target user:** A corporate sustainability officer or federal land-sector auditor producing Scope 3 or AFOLU-sector GHG inventory disclosures under the GHG Protocol Land Sector Guidance, CSRD, or SBTi FLAG frameworks.

**Value proposition:** What currently takes a geospatial analyst plus a carbon accountant three weeks takes this tool thirty seconds, with credible uncertainty bounds.

## 1.2 The Moment

An interactive map. The user draws a polygon over any region in the pre-staged AOI. A time slider runs across the available year range. As the slider moves:

1. Satellite basemap cross-fades through years
2. Classification overlay morphs — forests shrink, cropland expands, urban creeps
3. A running tCO₂e counter updates with a visibly widening or narrowing uncertainty band

Judge watches carbon being emitted under their cursor, in real time, offline, on the DGX Spark.

## 1.3 Abstract Data Input Layer

Data will arrive in unknown form. Design the system so a thin adapter is the only change needed to accept it. Canonical internal schemas below; any input format gets mapped to these on ingestion.

### 1.3.1 Input categories the system expects

**Imagery Source** — multi-band, multi-temporal raster data over the AOI. May arrive as:
- Cloud-optimized GeoTIFFs (COGs)
- A STAC catalog URL
- A directory of per-scene tiles
- Pre-composited annual median mosaics
- Any combination

**Boundary Source** — vector geometries defining AOIs, administrative units, or parcels. May arrive as GeoJSON, Shapefile, GeoPackage, or PostGIS table.

**Reference Labels (optional)** — pre-classified land cover products for validation or weak supervision. May be ESA WorldCover, Dynamic World, NLCD, MapBiomas, or a custom product provided by the sponsor.

**Auxiliary Layers (optional)** — climate zones, DEM, soil data, host-crop maps. Raster or vector.

### 1.3.2 Canonical internal schema

After ingestion, every input is normalized to:

```python
# Canonical AOI
@dataclass
class AOI:
    geometry: shapely.Polygon        # in EPSG:4326
    id: str
    metadata: dict                    # name, region, admin codes

# Canonical imagery stack
@dataclass
class ImageStack:
    aoi_id: str
    year: int
    bands: xr.DataArray               # dims: (band, y, x), 10m or native
    band_names: list[str]             # normalized: ['B','G','R','NIR','SWIR1','SWIR2']
    cloud_mask: xr.DataArray | None
    acquisition_dates: list[date]     # source scenes in composite
    crs: str                          # proj string

# Canonical classification output
@dataclass
class Classification:
    aoi_id: str
    year: int
    labels: xr.DataArray              # IPCC class codes 1-6
    confidence: xr.DataArray          # softmax max, per pixel
    class_probabilities: xr.DataArray # full softmax, (class, y, x)
    class_legend: dict[int, str]      # {1: 'Forest Land', ...}

# Canonical emissions output
@dataclass
class EmissionsReport:
    aoi_id: str
    year_start: int
    year_end: int
    transition_matrix: pd.DataFrame   # from_class × to_class, hectares
    per_transition_emissions: pd.DataFrame  # tCO2e with low/mean/high
    total_tCO2e: float
    total_tCO2e_ci95: tuple[float, float]
    methodology: str                  # e.g., "IPCC 2006 Tier 1"
```

### 1.3.3 Adapter pattern

One adapter per plausible input format. Each exposes a uniform interface:

```python
class ImagerySourceAdapter(Protocol):
    def list_available_years(self, aoi: AOI) -> list[int]: ...
    def load_stack(self, aoi: AOI, year: int) -> ImageStack: ...

# Implementations:
class STACAdapter(ImagerySourceAdapter): ...
class LocalCOGAdapter(ImagerySourceAdapter): ...
class PreComposedAnnualAdapter(ImagerySourceAdapter): ...
class SponsorProvidedAdapter(ImagerySourceAdapter): ...  # written at event
```

At hackathon start, whoever picks up data integration writes the one adapter needed for the actual format. Every downstream component is already wired against the Protocol.

## 1.4 System Architecture

```
[Input Data (unknown format)]
       │
       ▼
[Adapter Layer] ─────► canonical schemas
       │
       ▼
[Preprocessing]  ─ cloud masking, compositing, reprojection
       │
       ▼
[Classification]  ─ Prithvi-EO-2.0 fine-tuned, per-year
       │
       ▼
[Tile Generator]  ─ produces COG pyramid + web tiles for frontend
       │
       ▼
[Change Detection] ─ pixel-wise transition matrix between year pairs
       │
       ▼
[Emissions Engine] ─ IPCC Tier 1 EF lookup × transition matrix
       │
       ▼
[Uncertainty Engine] ─ Monte Carlo over classification probabilities
       │
       ▼
[FastAPI Backend] ─ exposes /classify, /emissions, /tiles, /summary
       │
       ▼
[MapLibre + deck.gl Frontend]
       │
       ▼
[Optional: Local LLM Orchestration Layer]
```

## 1.5 Component Specs

### 1.5.1 Preprocessing

- Cloud masking using Scene Classification Layer (SCL) if Sentinel-2, or Fmask for Landsat, or whatever arrives
- Annual median composite if raw scenes provided
- Reprojection to equal-area projection for accurate hectare calculations (e.g., EPSG:5070 for CONUS, EPSG:6933 globally)
- Optional: topographic correction if DEM available

Output: one `ImageStack` per (AOI, year).

### 1.5.2 Classification

**Model:** Prithvi-EO-2.0-300M (fallback to 600M only if fine-tune converges early and time allows).

**Head:** Lightweight UNet decoder producing 6-class segmentation aligned to IPCC AFOLU categories:
1. Forest Land
2. Cropland
3. Grassland
4. Wetlands
5. Settlements
6. Other Land

**Training regime for hackathon:**
- Frozen encoder, only decoder trained (or encoder with low LR)
- 1–2 epochs, 500–2000 patches
- Patches from a pre-published label source remapped to IPCC classes
- BF16 mixed precision (Spark native)
- Batch size driven by available memory; target 95%+ GPU utilization

**Output:** per-pixel class labels plus full softmax probabilities (needed for uncertainty).

**Fallback if fine-tuning fails:** Use Dynamic World probabilistic labels directly, map to IPCC classes, skip model training entirely. Pitch framing: "we focused engineering on the emissions pipeline, leveraging Google's published classification." Judges respect deliberate scope.

### 1.5.3 Tile Generator

Pre-generate COG pyramids for each classified year at multiple zoom levels. Serve via `titiler` or pre-render PNG tiles to disk. **This is the critical path for the slider animation** — on-demand classification during the demo will lag and the magic dies.

### 1.5.4 Change Detection

Straightforward pixel-wise:

```python
def transition_matrix(cls_t0: xr.DataArray, cls_t1: xr.DataArray, pixel_area_m2: float) -> pd.DataFrame:
    pairs = np.stack([cls_t0.values.ravel(), cls_t1.values.ravel()], axis=1)
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    hectares = counts * pixel_area_m2 / 10_000
    return pd.DataFrame({
        'from_class': unique[:, 0],
        'to_class': unique[:, 1],
        'hectares': hectares
    })
```

### 1.5.5 Emissions Engine

**Core idea:** Emissions from a land-use change are the difference in carbon stocks between the two states, summed across five carbon pools (above-ground biomass, below-ground biomass, dead wood, litter, soil organic carbon). IPCC 2006 Volume 4 publishes Tier 1 default emission factors for every category transition, stratified by climate zone.

**Implementation:** A lookup table keyed by `(from_class, to_class, climate_zone)` returning a tCO₂e/ha value with a low/mean/high range.

```python
# Example row schema
{
    "from_class": "Forest Land",
    "to_class": "Cropland",
    "climate_zone": "Tropical Wet",
    "tCO2e_per_ha_mean": 345.2,
    "tCO2e_per_ha_low": 280.0,
    "tCO2e_per_ha_high": 410.0,
    "source": "IPCC 2006 Vol 4 Ch 5 Table 5.1",
    "pools_included": ["AGB", "BGB", "DOM", "SOC"]
}
```

Total transitions matrix × EF table → per-transition emissions → total AOI emissions.

Pre-type this table **before the hackathon**. It's roughly 6×6×~5 climate zones = ~180 rows, most of which are zero (same-to-same transitions) or minor. Focus on the 15-20 transitions that actually matter.

### 1.5.6 Uncertainty Engine

Two sources of uncertainty to propagate:

**Classification uncertainty** — per-pixel softmax gives class probabilities. Sample class labels from the probability distribution N times (target: 500 MC iterations), recompute the transition matrix each time, get a distribution over total emissions.

**Emission factor uncertainty** — use IPCC's published low/high bounds as a uniform or triangular distribution per EF.

Combine via Monte Carlo:

```python
def monte_carlo_emissions(prob_t0, prob_t1, ef_table, n_iter=500, pixel_area_m2=100):
    totals = []
    for _ in range(n_iter):
        # Sample class per pixel from softmax
        cls_t0 = sample_categorical(prob_t0)
        cls_t1 = sample_categorical(prob_t1)
        # Sample EFs from their uncertainty ranges
        sampled_ef = sample_ef_table(ef_table)
        # Compute total emissions for this draw
        tm = transition_matrix_fast(cls_t0, cls_t1, pixel_area_m2)
        total = (tm['hectares'] * tm.apply(lambda r: sampled_ef[(r.from_class, r.to_class)], axis=1)).sum()
        totals.append(total)
    return np.mean(totals), np.percentile(totals, [2.5, 97.5])
```

**Performance note:** 500 iterations over a large AOI can be slow. Pre-compute the transition matrix once per sampled classification, vectorize the EF application, and cache aggressively. On the Spark, target <5 seconds for the full Monte Carlo on a typical demo AOI.

## 1.6 API Contract

FastAPI backend exposes:

```
GET  /api/aois                              List available pre-staged AOIs
GET  /api/aois/{id}/years                    List available years
GET  /api/aois/{id}/years/{year}/tiles/{z}/{x}/{y}.png   Classification tile
GET  /api/aois/{id}/years/{year}/imagery/{z}/{x}/{y}.png Basemap tile (optional)
POST /api/analyze                            { aoi_id | custom_geom, year_start, year_end }
                                             → EmissionsReport JSON
GET  /api/pixel-detail                       ?aoi_id=&year=&lat=&lon=
                                             → per-pixel classification + confidence
POST /api/llm/summarize                      { emissions_report } → disclosure paragraph
```

Response payloads follow the canonical schemas from §1.3.2 serialized to JSON.

## 1.7 Frontend Spec

**Stack:** MapLibre GL JS for basemap + raster tiles, deck.gl for overlays and animations, vanilla TypeScript or React (choose whichever is faster for your team — React if you're using component-based state).

**Layout:**

```
┌──────────────────────────────────────────────────────┐
│ [Logo]  Land Carbon Explorer        [AOI Selector ▾] │
├──────────────────────────────────────────────────────┤
│                                                      │
│                                                      │
│              [ Full-screen Map ]                     │
│                                                      │
│                                                      │
│  ┌─────────────────────────────────┐                 │
│  │ Year: 2019 ────●──────── 2024   │                 │
│  └─────────────────────────────────┘                 │
├──────────────────────────────────────────────────────┤
│ tCO₂e emitted: 48,230 [±6,100]  |  Details ▸         │
└──────────────────────────────────────────────────────┘
```

**Interactions:**

- Draw polygon tool in top-right: judge draws AOI → triggers `/api/analyze`
- Time slider at bottom: dragging animates between pre-rendered classification tiles and updates the running emissions total
- Legend and class toggles in a collapsible panel
- Click any pixel → pixel detail panel with classification, confidence, emission contribution
- "Generate disclosure" button → invokes LLM endpoint, streams paragraph into a modal

**Aesthetic:** Scientific register. Muted earth tones (Google Earth Engine / Global Forest Watch palette), not neon. Sans-serif UI chrome (Inter or system-ui). Generous whitespace. The map is the hero; UI fades into the background.

**Performance requirement:** Slider must animate smoothly. Pre-render tiles, prefetch adjacent years, debounce tCO₂e recomputation.

## 1.8 Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Model | Prithvi-EO-2.0-300M | Best foundation model for HLS imagery, ARM-compatible |
| Fine-tuning | terratorch | YAML config, no boilerplate |
| Raster I/O | rioxarray, stackstac | Lazy loading, xarray ecosystem |
| Vector I/O | geopandas, shapely | Standard |
| Tile serving | titiler or pre-rendered PNG | titiler if time allows, PNG pre-render as fallback |
| Backend | FastAPI + uvicorn | Fast, typed, async |
| Frontend | MapLibre GL + deck.gl + React | Best-in-class geo viz |
| LLM (optional) | Ollama + Llama-3.1-8B or Qwen-2.5-14B | Runs on Spark, function-calling supported |
| Orchestration | Docker Compose | One command to spin up demo |

## 1.9 22-Hour Execution Plan

| Hours | Milestone | Owner |
|---|---|---|
| 0–1 | Kickoff, data format confirmed, adapter sketched | Whole team |
| 1–3 | Adapter implemented, canonical ImageStacks flowing | Data |
| 1–4 | Frontend skeleton with map + slider + fake overlay | Frontend |
| 3–8 | Classification pipeline running end-to-end on one AOI, one year pair | ML |
| 4–8 | Emissions engine + EF table + transition logic | Backend |
| 8–10 | Tile generation for all years in slider range | ML + Backend |
| 8–12 | Frontend wired to real API, slider animates real tiles | Frontend |
| 10–14 | Uncertainty engine integrated, running tCO₂e counter live | Backend |
| 14 | **SHIP-IT CHECKPOINT** — demo path works end-to-end | Whole team |
| 14–17 | Pixel-detail panel, polish, color ramp tuning | Frontend |
| 14–18 | Optional: LLM disclosure generation | ML |
| 17–20 | Pitch deck, demo rehearsal | Whole team |
| 20–21 | Final polish, dry run with judges-style Q&A | Whole team |
| 21 | **STOP CODING** | — |
| 21–22 | Sleep or final prep | — |

## 1.10 Cut Order

If behind schedule, cut in this order:

1. LLM disclosure generation (last-in, first-out)
2. Pixel detail panel
3. Full Monte Carlo (replace with static ±20% uncertainty band)
4. 600M model (stick with 300M)
5. Multiple year pairs (show one pair only)
6. Custom polygon drawing (use a hardcoded demo AOI)

Never cut: the map + slider + classification overlay + total emissions number. That is the demo.

## 1.11 Demo Script (3 minutes)

**0:00–0:30** "Corporate sustainability officers reporting under CSRD need to quantify land-sector emissions across supply chains. Today, that takes a geospatial analyst, a carbon accountant, and three weeks per region. Here's thirty seconds."

**0:30–1:30** Live demo. Select pre-staged AOI, drag slider across years, total tCO₂e counts up visibly. Click a pixel, show the decomposition. If LLM is wired: click "generate disclosure," paragraph appears citing your actual numbers.

**1:30–2:15** "Under the hood: NASA/IBM's Prithvi foundation model, fine-tuned on our AOI. IPCC 2006 Tier 1 emission factors keyed by climate zone. Monte Carlo uncertainty propagation across classification confidence and EF bounds. Aligned with the GHG Protocol Land Sector Guidance."

**2:15–2:45** "Entire system runs on this six-inch DGX Spark, offline, which matters for any auditor handling proprietary supply-chain data."

**2:45–3:00** "We demoed on Colorado because we had 22 hours. The code is AOI-agnostic — point it at Mato Grosso tomorrow and the pipeline runs. Thanks."

## 1.12 Q&A Preparation

Likely judge questions:

- **"How do you handle cloud cover?"** SCL masking, annual median compositing, gap-fill from adjacent years.
- **"What's your classification accuracy?"** State it honestly. "On our holdout set, overall accuracy was X%, with weakest performance on wetlands — a known hard class for optical imagery."
- **"Why Tier 1 and not Tier 2 or 3?"** Tier 1 is defensible for screening; Tier 2/3 require region-specific EFs that aren't possible in 22 hours but the pipeline accepts them as drop-in replacements.
- **"How does this compare to existing tools?"** Global Forest Watch covers forest loss only. MapBiomas is Brazil-only. Pachama and Sylvera are commercial and black-box. This is open, multi-sector, and auditable.
- **"What's the business model?"** If asked — "API access for corporate reporting platforms, with institutional pricing."

---

# PART 2 — AGRICULTURAL RISK & BIOSECURITY

## 2.1 Winning Thesis

Build a pathway-risk assessment and surveillance-targeting tool for invasive fruit flies entering the US, stratified by species, seasonality, and destination vulnerability — visualized through a dynamic 3D globe that makes the seasonal, multi-species structure of invasion risk immediately legible.

**Target user:** A USDA APHIS port risk analyst or state-level agricultural commissioner deciding where to deploy limited inspection and trap resources next quarter.

**Value proposition:** Currently, surveillance deployment is driven by historical precedent and expert judgment. This tool adds a dynamic, quantitative, species-aware risk layer that responds to seasonality and trade-flow shifts.

## 2.2 The Moment

3D globe with animated pathway arcs from origin countries to US ports of entry. Arcs are:

- **Colored** by fruit fly species (distinct palette per species)
- **Thickness** scaled by passenger or cargo volume on that route
- **Opacity** pulsing with seasonal risk at the currently-selected month
- **Clickable** — each arc opens a sidebar with the decomposed risk score

A time slider runs across the calendar year. As it advances, the globe's risk geography shifts — Pakistan lights up in mango season, the Mediterranean flares during summer citrus, Southeast Asia stays hot year-round.

Second view, one click away: continental US county map shaded by establishment vulnerability (climate × hosts × inbound risk). Optionally, a recommended trap placement overlay from the facility-location optimizer.

## 2.3 Abstract Data Input Layer

Input data will arrive in unknown form. Design around canonical internal schemas.

### 2.3.1 Input categories the system expects

**Pathway Flow Source** — origin-destination volumes over time. Could be air passenger volumes (US DOT T-100, IATA), maritime cargo (CBP AMS, PIERS), mail pathways, or any combination.

**Interception Record Source** — confirmed incidents of fruit fly interceptions at US ports. Likely USDA APHIS AQIM or equivalent; may be per-incident or pre-aggregated.

**Species Presence Source** — which species occurs in which geography. CABI Invasive Species Compendium, EPPO Global Database, GBIF occurrence records, or a custom sponsor-provided file.

**Host Crop Source** — US host crop distribution and area. USDA Cropland Data Layer, NASS Census of Agriculture, or sponsor-provided.

**Climate Suitability Source** — per-species climate envelope rasters, either pre-published from literature or computed at event from WorldClim.

**Economic Value Source (optional)** — dollar value of at-risk agricultural production by county/crop.

### 2.3.2 Canonical internal schema

```python
@dataclass
class PathwayFlow:
    origin_country: str            # ISO-3
    destination_port: str          # IATA code or CBP port code
    month: int                     # 1-12
    pathway_type: str              # 'air_passenger' | 'air_cargo' | 'maritime_cargo' | 'mail'
    volume: float                  # normalized units (passengers, tonnes, etc.)
    commodity: str | None          # for cargo

@dataclass
class Interception:
    species: str                   # standardized binomial
    date: date
    origin_country: str
    port: str
    pathway_type: str
    commodity: str | None
    life_stage: str | None         # egg/larva/adult

@dataclass
class SpeciesPresence:
    species: str
    country: str                   # ISO-3
    presence_status: str           # 'established' | 'detected' | 'absent'
    intensity_score: float | None  # 0-1 if available
    seasonality: dict[int, float] | None  # month -> relative abundance

@dataclass
class HostPresence:
    county_fips: str
    crop: str
    hectares: float
    economic_value_usd: float | None

@dataclass
class ClimateSuitability:
    species: str
    raster: xr.DataArray           # 0-1 suitability, CONUS
    source: str                    # 'MaxEnt_WorldClim' | 'literature' | ...

@dataclass
class RiskScore:
    species: str
    origin: str
    destination_port: str
    month: int
    p_origin_prevalence: float
    p_contamination_given_pathway: float
    p_survival_in_transit: float
    p_establishment_given_arrival: float
    composite_risk: float
    uncertainty_ci95: tuple[float, float]
```

### 2.3.3 Adapter pattern

Same pattern as Part 1. One Protocol per input category, implementations swapped in based on what data actually arrives.

```python
class PathwayFlowAdapter(Protocol):
    def load_flows(self, year_range: tuple[int, int]) -> Iterable[PathwayFlow]: ...

class InterceptionAdapter(Protocol):
    def load_interceptions(self, year_range: tuple[int, int]) -> Iterable[Interception]: ...

# ... etc per category
```

## 2.4 System Architecture

```
[Input Data (unknown format)]
       │
       ▼
[Adapter Layer] ─────► canonical schemas
       │
       ▼
[Risk Model]  ─ 4-factor pathway risk per (species, origin, port, month)
       │
       ▼
[Establishment Model] ─ county-level vulnerability = climate × hosts × inbound
       │
       ▼
[Optimization Layer (optional)] ─ trap placement, budget constraints
       │
       ▼
[FastAPI Backend]
       │
       ▼
[deck.gl Globe + MapLibre Frontend]
       │
       ▼
[Optional: LLM Briefing Generator]
```

## 2.5 Component Specs

### 2.5.1 Risk Model

The core formula, per (species × origin × destination × month):

```
Risk = P(origin_prevalence)
     × P(contamination | pathway, commodity)
     × P(survival_in_transit | pathway, transit_time)
     × P(establishment | destination_climate, host_availability)
```

Each factor:

**P(origin_prevalence)** — derived from SpeciesPresence records. If `presence_status == 'established'` and seasonality is available, use month-specific intensity. Otherwise fallback to a step function (1.0 if established, 0.3 if detected, 0 if absent).

**P(contamination | pathway)** — empirical rate from interception records. For each (origin, pathway, commodity) combination, compute `interceptions / volume` as a hit rate. Smooth with a Bayesian prior to handle sparse cells.

**P(survival_in_transit)** — simple decay model by pathway type:
- Air passenger: 0.95 (fast, fruit in carry-on)
- Air cargo: 0.80
- Maritime cargo (ambient): 0.30
- Maritime cargo (cold chain): 0.05
- Mail: 0.60

Tune based on literature if time allows.

**P(establishment | arrival)** — per destination, computed as: climate suitability at destination × host availability within dispersal radius. Weighted by species-specific climate envelope and host preference.

### 2.5.2 Establishment Model

County-level vulnerability:

```python
def county_vulnerability(county_fips, species, month):
    climate = climate_suitability[species].at(county_fips)
    hosts = host_area_weighted_by_preference(county_fips, species)
    inbound = sum(risk for risk in all_inbound_risks if risk.destination_serves(county_fips))
    return climate * hosts * inbound
```

Ports serve counties based on proximity and historical distribution patterns. For hackathon simplicity, use straight-line distance with a cutoff (e.g., ports serve counties within 500km with exponential decay).

### 2.5.3 Optimization Layer

Facility location problem: given N traps and M candidate counties, maximize total vulnerability covered subject to:

- Budget constraint
- Minimum coverage per high-risk species
- Diversity constraint (don't cluster all traps in one state)

Solve with a simple greedy heuristic or PuLP if time allows. Output: ranked list of counties with recommended trap counts.

This is the "surveillance optimization" deliverable that turns the dashboard into a decision tool.

### 2.5.4 LLM Briefing Generator (optional)

Input: structured risk summary for the selected month, top N routes, top N vulnerable counties, notable changes vs last month.

Prompt template:

```
You are a USDA APHIS analyst preparing a monthly pathway risk briefing.
Data for {month}:
- Top 3 origin-species risks: {...}
- Top 5 vulnerable counties: {...}
- Month-over-month changes: {...}

Produce a 2-paragraph briefing in formal government register. Cite specific routes and species by binomial name. Recommend surveillance actions.
```

Output streams into a sidebar panel.

## 2.6 API Contract

```
GET  /api/species                             List species tracked
GET  /api/months/{month}/flows                Pathway flows for a month
GET  /api/months/{month}/risks                Composite risk scores
GET  /api/routes/{species}/{origin}/{port}    Decomposed risk for a specific route
GET  /api/counties/vulnerability              County vulnerability scores
POST /api/optimize                            { budget, constraints } → trap placements
POST /api/llm/briefing                        { month, filters } → briefing text
```

## 2.7 Frontend Spec

**Stack:** deck.gl (GlobeView with ArcLayer), MapLibre for the county map, React for state management.

**Primary view — globe:**

```
┌──────────────────────────────────────────────────────┐
│ [Logo]  Fruit Fly Pathway Risk    [Species Filter ▾] │
├──────────────────────────────────────────────────────┤
│                                                      │
│                                                      │
│              [ 3D Globe with Arcs ]                  │
│                                                      │
│                                                      │
│  ┌─────────────────────────────────┐                 │
│  │ Month: Jan ───●──── Dec         │                 │
│  └─────────────────────────────────┘                 │
├──────────────────────────────────────────────────────┤
│ Top risk this month: Oriental FF from Thailand → LAX │
│ [ View US County Risk ▸ ]                            │
└──────────────────────────────────────────────────────┘
```

**Secondary view — US counties:**

Choropleth shaded by vulnerability. Toggle layers:
- Climate suitability
- Host crop area
- Inbound pathway risk
- Composite vulnerability (default)
- Recommended trap placements (overlay)

Click county → detail panel with contributing origins, dominant species threat, recommended action.

**Interactions:**

- Month slider: animates arc opacities and county shading in sync
- Species filter: multi-select toggle in legend
- Arc click: opens decomposed risk panel
- "Generate briefing" button: invokes LLM endpoint
- "Optimize trap placement" panel: budget slider + constraints → updates trap overlay

**Aesthetic:** Dark globe background (space-like, lets arcs glow). Species colors from a colorblind-safe categorical palette. Counties in a sequential heat scale. Typography: Inter or similar, serious register.

## 2.8 Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Data processing | pandas, geopandas, xarray | Standard geo stack |
| Climate modeling | elapid or scikit-learn MaxEnt | Only if pre-published suitability unavailable |
| Optimization | PuLP or custom greedy | Depends on time |
| Backend | FastAPI | Consistent with Part 1 |
| Frontend | React + deck.gl + MapLibre | Globe and choropleth both |
| LLM (optional) | Ollama + Llama-3.1-8B | Runs on Spark |

**Note:** No heavy GPU training needed. The Spark is overkill for the compute; its value here is running the LLM locally for the briefing generator.

## 2.9 22-Hour Execution Plan

| Hours | Milestone | Owner |
|---|---|---|
| 0–1 | Kickoff, data format confirmed, adapter priorities set | Whole team |
| 1–4 | Adapter layer, canonical records flowing | Data |
| 1–4 | Frontend skeleton: globe renders, placeholder arcs, slider works | Frontend |
| 4–8 | Risk model: 4-factor pipeline produces RiskScore records | Backend |
| 4–8 | Globe wired to real data, arcs animate by month | Frontend |
| 8–11 | County vulnerability model, choropleth view | Backend + Frontend |
| 11–14 | Route click-through panels, species filtering | Frontend |
| 14 | **SHIP-IT CHECKPOINT** — globe + county view working with real data | Whole team |
| 14–17 | Optimization layer + trap overlay | Backend + Frontend |
| 14–18 | Optional: LLM briefing | ML |
| 17–20 | Visual polish, color palette, typography | Frontend |
| 17–20 | Pitch deck, demo rehearsal | Whole team |
| 20–21 | Final polish, Q&A dry run | Whole team |
| 21 | **STOP CODING** | — |

## 2.10 Cut Order

1. LLM briefing
2. Optimization layer with constraints (keep simple greedy)
3. Month-over-month delta analysis
4. Maritime cargo pathway (keep only air passenger + air cargo)
5. Species count (drop to 3 if started with 5)
6. County-level optimization (stay at state level)

Never cut: the globe with animated arcs + month slider + county vulnerability map. That is the demo.

## 2.11 Demo Script (3 minutes)

**0:00–0:30** "USDA APHIS intercepts invasive fruit flies at US ports every day. A single Medfly establishment in California's stone fruit belt is a multi-billion-dollar event. Surveillance resources are finite. Our tool tells you where to point them, by species, by month."

**0:30–1:30** Live demo. Start on globe. Drag month slider through the calendar year — arcs flare and fade with seasonality. Click an arc mid-summer from Thailand to LAX. Decomposition panel: "Oriental fruit fly, 0.78 composite risk, dominant factor is P(contamination | air cargo mangoes)." Switch to county view. Point at California's Central Valley. Click "optimize trap placement" — overlay appears showing recommended trap distribution. If LLM is wired: click "briefing," paragraph appears.

**1:30–2:15** "Risk model is a four-factor decomposition: origin prevalence, pathway contamination rate from real APHIS interception data, transit survival, and destination establishment probability from climate envelopes and host crop distribution. Species-stratified across Medfly, Oriental fruit fly, and Mexican fruit fly. Seasonal at monthly resolution."

**2:15–2:45** "This is a screening tool — it prioritizes where to deploy fuller investigation, not a substitute for expert risk analysis. We're validating our top-ranked routes against historical APHIS interceptions; correlation is [X]."

**2:45–3:00** "Full pipeline runs offline on the Spark, which matters for any agency working with sensitive trade and surveillance data. Code is species-agnostic — adding Queensland fruit fly next is a data add, not a rewrite."

## 2.12 Q&A Preparation

- **"How did you validate the model?"** Compare top-ranked routes to historical interceptions. Show a ranked list and the overlap.
- **"What data are you using?"** Name the sources: APHIS AQIM, DOT T-100, USDA CDL, CABI, WorldClim. Judges from USDA or related agencies will immediately recognize and trust these names.
- **"What about emerging pathways like e-commerce parcels?"** Valid and growing — the adapter layer accepts new pathway types without model changes. Demonstrating it is a data acquisition problem, not a modeling one.
- **"How would you extend to other pests?"** Schema is species-agnostic. Brown marmorated stink bug, spotted lanternfly, Khapra beetle — same pipeline, different presence + host data.
- **"What's the accuracy?"** Be honest: "This is a relative ranking tool, not an absolute predictor. Its job is to point resources, not to predict specific incidents."

---

# PART 3 — CROSS-CUTTING

## 3.1 Hardware & Environment

- DGX Spark (128GB unified memory, GB10 Grace Blackwell)
- Full software stack pre-installed and tested before event: CUDA, cuDNN, PyTorch (ARM aarch64 wheels), terratorch, torchgeo, rasterio, geopandas, deck.gl via npm
- Pre-downloaded: Prithvi-EO-2.0-300M weights, chosen LLM weights (Llama-3.1-8B or Qwen-2.5-14B at minimum)
- Pre-staged: demo AOI imagery, IPCC emission factor table, any non-sensitive reference data for fruit fly prompt
- Backup: cloud environment spun up with credits loaded, as hot standby if hardware fails

## 3.2 AWS Integration (if credits available)

- **S3 for data staging** — 30 minutes of setup, legitimizes "we used AWS" for sponsor prize eligibility
- **Optional: SageMaker for parallel hyperparameter sweeps** — only if Spark workload leaves time for experimentation
- **Do not** attempt live training on AWS under 22-hour pressure unless pre-configured

Ask the AWS sponsor rep at the event: "What would make us a strong candidate for the AWS prize?" Their answer is your integration target.

## 3.3 Team Division (assumed 3-4 people)

| Role | Primary | Secondary |
|---|---|---|
| Data / adapters | Person A | — |
| ML / modeling | Person B | Person A after adapters ship |
| Backend / API | Person B or C | — |
| Frontend | Person C or D | — |
| Pitch / narrative | Whole team | Lead picked by hour 17 |

If solo or pair: drop the LLM layer and the optimization layer. Keep the core demo.

## 3.4 Pitch Fundamentals (both prompts)

- **Open with a specific user and a specific problem.** Not "industry" — a person with a job title.
- **Demo before methodology.** Judges buy in visually; then you earn their attention for the technical depth.
- **Name things specifically.** "Prithvi-EO-2.0," "IPCC 2006 Tier 1," "APHIS AQIM interception records," "MaxEnt climate envelope." Vocabulary is credibility.
- **Acknowledge one limitation honestly.** Judges trust teams that know what their tool doesn't do.
- **Close with a forward path.** "Adding [X] is a data add, not a rewrite." Shows you built something extensible.

## 3.5 The Single Most Important Rule

**Ship a narrow thing that works end-to-end by hour 14.** Everything after hour 14 is polish. Teams that keep adding features past hour 18 have a 10x higher chance of demoing a broken product. Every hackathon post-mortem confirms this.

---

*End of spec. Good luck.*
