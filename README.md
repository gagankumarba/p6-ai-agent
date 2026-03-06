P6-Agent: Integrated Planning System (IPS) for High-Rise Construction
This repository contains the P6-Agent, a specialized component of an Integrated Planning System (IPS) designed to automate and optimize construction scheduling for high-rise residential and commercial buildings. The agent is trained on a comprehensive baseline from the EMIRAL Residential Towers project in Algiers, consisting of four towers (3 Basements + G + M + 22 Floors + Saloon Terrace) totaling 211,768 $m^2$.

🚀 Overview
The P6-Agent leverages Machine Learning (ML), Natural Language Processing (NLP), and Bayesian Inference to "learn" the engineering logic behind complex P6 schedules. By analyzing a baseline of 19,202 activities, the agent differentiates between user-defined inputs and system-generated outputs, enabling it to act as a reference for new high-rise projects.
Key Project Specifications (EMIRAL Towers)
Total Activities: 19,202.
Project Phases: 4 distinct phases with structural dependencies (e.g., Tower 4 depends on Tower 2/3 podium blinding)
Calendar: "Emiral - Algeria" (Friday weekend) with specialized 7-day calendars for curing and testing.
WBS Depth: 8-level decomposition (Project down to Dry/Wet/Common Area).

🛠 Features
1. Training Data Extractor: An independent script that reverse-engineers quantities and productivity from P6 Excel exports (TASK, TASKRSRC, TASKPRED).Reverse Engineering: Calculates hidden quantities using $Quantity = \frac{Budgeted\ Labor\ Units}{Labor\ Constant\ from\ Norms}$.Vertical Multiplier: Learns productivity degradation patterns from the Ground Floor to the 22nd floor .
2. Hybrid Logic Engine (NLP)Semantic Labeling: Uses NLP to classify activities as Norm-Based (resource-driven) or Standard-Based (fixed durations like curing/testing).Typicality Index: Clusters identical tasks across 22 floors to identify "Typical Floor Cycles" versus "Non-Typical" exceptions (Basements, Saloon Terrace) .
3. Bayesian Probabilistic ReferenceConverts activity "words" (metadata) into mathematical expressions to predict durations for new projects.Posterior Inference: $P(Duration | Attributes) = \frac{P(Attributes | Duration) \times P(Duration)}{P(Attributes)}$Updates the probability of success based on Discipline, Location, and Resource Density.
  
📂 Repository Structure
├── data/
│   ├── norms/             # Productivity sheets (e.g., Div 02: Site Work)
│   ├── p6_exports/        # TASK, TASKRSRC, and TASKPRED Excel sheets
│   └── training/          # Annotated JSON training pairs
├── src/
│   ├── extractor.py       # Independent Data Extractor Script
│   ├── nlp_classifier.py  # Semantic work-class labeling
│   ├── logic_graph.py     # Graph analysis for predecessors and lags
│   └── bayesian_model.py  # Probabilistic duration prediction
└── README.md

📝 Methodology
The agent follows a five-phase construction scheduling workflow:
Scope Definition: Mapping WBS levels 1–8 and project-specific calendars.
Activity Identification: Decoding standardized IDs (e.g., E-T-T2-FN-GFML0040) for location and trade.
Logic Mining: Extracting FS/SS relationships and site-access lags (e.g., Ramp closure constraints).
Resource Analysis: Loading man-hours and calculating resource intensity patterns.
Critical Path & Risk: Identifying critical activities and filtering "Planner Errors" (Negative Float) from the training set.

🚦 Getting Started
P6 Export: Generate an Excel export from Primavera P6 including Activity ID, WBS Code, Original Duration, Predecessor Details, and Budgeted Labor Units.
Annotation: Run the nlp_classifier.py to flag "Standard" vs "Norm" activities.
Train: Use the extractor.py to generate the relational JSON dataset for Bayesian inference.
