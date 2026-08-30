# 🏀 Basketball Tactics

An MVP computer-vision pipeline for extracting basketball player movement, team assignments, ball possession, and tactical events from game footage.

The project converts raw basketball video into structured data that can later be used for tactical analysis and AI-generated coaching insights.

## 🚀 MVP V1 Pipeline

The current pipeline performs the following steps:

```text
Input Video
    │
    ▼
FPS Normalization
    │
    ▼
Player Detection + Tracking
    │
    ▼
Team Assignment
    │
    ▼
Court Coordinate Transformation
    │
    ▼
Ball Tracking + Possession
    │
    ▼
Tactical Event Extraction
```

### Pipeline stages

1. **Video FPS normalization**

   * Videos above 30 FPS are normalized to 30 FPS.
   * Videos at or below 30 FPS are kept at their original FPS.

2. **Player detection and tracking**

   * Uses Roboflow for player detection.
   * Uses ByteTrack through Supervision for persistent player IDs.
   * Extracts bounding boxes and player centers.

3. **Team assignment**

   * Classifies players into `Team_A` and `Team_B`.
   * Uses jersey-color information to distinguish teams.
   * Excludes non-player detections where appropriate.

4. **Court coordinate transformation**

   * Detects basketball-court landmarks.
   * Calculates a homography between camera coordinates and a canonical court coordinate system.
   * Transforms player positions into court coordinates.

5. **Ball possession**

   * Tracks the ball.
   * Finds the nearest player.
   * Applies temporal smoothing to estimate possession.
   * Identifies possession changes and possession segments.

6. **Tactical events**

   * Combines player, team, ball, and possession data.
   * Produces team-aware tactical events.

---

## 📁 Project Structure

```text
basketball-tactics/
│
├── pipeline.py
│
├── convert_video.py
├── extract_coordinates.py
├── team_assignment.py
├── transform_player_coordinates.py
├── possession_v6.py
├── build_tactical_events.py
│
├── court_points.py
├── court_transform.py
├── build_court_template.py
│
├── run_detection.py
├── extract_frame.py
│
├── visualize_court.py
├── visualize_detection.py
├── visualize_possession.py
├── plot_movement.py
├── plot_topdown.py
│
├── test_court.py
├── test_real_homography.py
├── test_topdown.py
│
├── data/
│   ├── videos/
│   └── processed/
│
├── outputs/
│
├── .env
├── .gitignore
└── README.md
```

Generated videos, CSV files, raw data, virtual environments, and secrets are intentionally excluded from Git.

---

## ⚙️ Requirements

Python 3.12 is currently used for development.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install opencv-python numpy pandas supervision python-dotenv inference-sdk
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key
```

The `.env` file is ignored by Git and should never be committed.

---

## 🎥 Input Video

Place the source video inside:

```text
data/videos/
```

For example:

```text
data/videos/game.mp4
```

The pipeline automatically creates:

```text
data/processed/video_30fps.mp4
```

when FPS normalization is required.

---

## ▶️ Run the Complete Pipeline

Run:

```bash
python pipeline.py data/videos/game.mp4
```

The pipeline executes all V1 processing stages automatically.

Expected stages:

```text
1. FPS normalization
2. Player coordinate extraction
3. Team assignment
4. Court coordinate transformation
5. Possession estimation
6. Tactical event extraction
```

---

## 📊 Generated Data

The main V1 outputs are:

### Player coordinates

```text
outputs/player_coordinates_v2.csv
```

Contains tracked player bounding boxes and center coordinates.

### Team assignments

```text
outputs/player_team_assignments.csv
```

Contains player-level team classification and jersey-color information.

### Player court coordinates

```text
outputs/player_coordinates_with_teams.csv
```

Contains player coordinates transformed into the canonical court coordinate system.

Example fields include:

```text
frame
player_id
center_x
center_y
team
jersey_color
court_x
court_y
```

### Possession

```text
outputs/possession_v6.csv
```

Contains estimated ball possession and possession-related information.

### Tactical events

```text
outputs/tactical_events.csv
```

Contains team-aware tactical events generated from the tracking and possession data.

---

## 📐 Court Coordinate System

The current MVP uses a canonical **640 × 640 court coordinate system** based on the court landmark template.

Court landmarks are stored in:

```text
court_points.py
```

The homography implementation is contained in:

```text
court_transform.py
```

The transformation maps camera-space coordinates into the canonical court-space representation.

---

## 🧪 Testing

Individual stages can be tested independently.

For example:

```bash
python test_court.py
```

Real court homography testing:

```bash
python test_real_homography.py
```

Top-down visualization testing:

```bash
python test_topdown.py
```

---

## 🎞️ Visualization

The repository contains scripts for inspecting intermediate results:

```bash
python visualize_detection.py
python visualize_court.py
python visualize_possession.py
python plot_movement.py
python plot_topdown.py
```

These are useful for validating the computer-vision pipeline before using the generated data for higher-level tactical analysis.

---

## ⚠️ Current MVP Limitations

This is an early V1 system and should not yet be treated as production-grade basketball analytics.

Current limitations include:

* Player identity tracking can change when detections are lost.
* Team classification is based primarily on visual jersey information.
* Ball detection can contain missed observations.
* Possession is estimated from spatial proximity and temporal smoothing.
* Court homography depends on reliable court landmark detection.
* The canonical court coordinates are currently used for geometric consistency rather than a fully calibrated real-world NBA court model.
* Tactical event detection is still an early rule-based layer.
* Pass detection is intentionally not part of V1.

---

## 🛣️ Roadmap

### V1 — Computer Vision Foundation

* [x] Video FPS normalization
* [x] Player detection
* [x] Player tracking
* [x] Team assignment
* [x] Court landmark detection
* [x] Homography transformation
* [x] Ball tracking
* [x] Possession estimation
* [x] Team-aware tactical events

### V2 — Tactical Intelligence

Planned improvements include:

* [ ] Reliable pass detection
* [ ] Dribble detection
* [ ] Shot detection
* [ ] Offensive/defensive possession classification
* [ ] Formation detection
* [ ] Player spacing analysis
* [ ] Ball movement analysis
* [ ] Possession-based tactical sequences
* [ ] Improved court calibration

### V3 — AI Coaching Layer

Future work will connect the structured tactical data to an instruction-following language model to generate natural-language tactical analysis.

Potential output:

```text
Team A repeatedly created space on the weak side
before attacking the right half-court.

The primary ball handler initiated the possession
from the top of the key, while two teammates moved
toward the weak side.

The sequence suggests a recurring spacing pattern
that could be analyzed across multiple possessions.
```

A future Streamlit interface will make these insights accessible through an interactive dashboard.

---

## 🔒 Data and Security

The following are intentionally excluded from the repository:

```text
.env
.venv/
data/
outputs/
__pycache__/
.DS_Store
```

Do not commit API keys, raw game footage, or generated datasets unless they are intentionally prepared for public release.

---

## 📜 Status

**Current status: MVP V1**

The computer-vision foundation is operational end-to-end on the current test footage.

The next major development layer is tactical intelligence built on top of the structured tracking, possession, and event data.
