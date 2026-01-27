# Two-Stage Fraud Detection Production MLOps Pipeline

Two-Stage Fraud Detection is a production-oriented MLOps project designed to simulate how fraud detection systems are built, deployed, and monitored in real-world environments.  
The system uses a two-stage inference architecture to balance low latency and high recall by routing only uncertain transactions to a more expressive model.

This project focuses on system design, trade-offs, and deployment rather than just model accuracy.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Design Decisions](#key-design-decisions)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Logging and monitoring](#logging-and-monitoring)
- [Docker Deployment](#docker-deployment)
- [Requirements](#requirements)

---

## Project Overview

The goal of this project is to understand how machine learning pipelines move from experimentation to production.

The system simulates a real-world fraud detection workflow where:
- Transaction data is stored in a database
- Models are trained offline
- Inference is served via an API
- Predictions are logged for auditing and monitoring

Instead of using a single model, the pipeline uses a two-stage design:
- Stage 1 handles most traffic quickly
- Stage 2 improves recall on uncertain predictions

---

## Key Design Decisions

### Why Two-Stage Inference

Using a single model with threshold tuning introduces a trade-off:
- Lower thresholds improve recall but increase false positives
- Higher thresholds improve precision but increase false negatives

To avoid degrading user experience while still improving recall, this system:
- Keeps the first-stage model conservative and fast
- Escalates only uncertain predictions to a second-stage model

This design improves system-level recall without compromising precision for confident predictions.

---

## System Architecture

Client / Frontend
|
v
FastAPI Inference Service
|
v
Stage 1 Model (Logistic Regression)
|
|-- Confident Prediction -> Final Decision
|
|-- Uncertain (0.3–0.7 Probability)
|
v
Stage 2 Model (XGBoost)
|
v
Final Decision
|
v
Logging and PostgreSQL Storage


---

## Features

- Two-stage inference architecture
- Uncertainty-based routing using prediction probabilities
- Low-latency inference for majority of traffic
- System-level metrics including recall and escalation rate
- PostgreSQL-backed data ingestion and inference logging
- Structured logging for auditability
- FastAPI-based REST inference service
- Simple HTML frontend for testing
- Dockerized deployment for reproducibility

---

## Installation

```bash
git clone https://github.com/Amith0707/Two-stage-fraud-MLOps.git
cd Two-stage-fraud-MLOps
pip install -r requirements.txt
```
---

## Usage

#### Run the FastAPI Server

```bash
uvicorn src.api.app:app --reload
```

#### Open in browser

```bash
http://localhost:8000
```
---

## API Endpoints

| Endpoint  | Method | Description               |
| --------- | ------ | ------------------------- |
| /         | GET    | Simple frontend interface |
| /predict  | POST   | Fraud prediction endpoint |
| /health   | GET    | Service health check      |
| /metadata | GET    | Model and system metadata |

---

## Logging and monitoring

Each inference request logs:

*   Input transaction features

*   Model prediction

*   Prediction probability

*   Stage used for inference

*   Timestamp

Logs are written to both:

*   Structured log files

*   PostgreSQL database

This enables debugging, monitoring, and future retraining.

---

## Docker Deployment

### Build the Image

```bash
docker build -t two-stage-fraud-api .
```

### Run the container

```bash
docker run -p 5000:5500 --env-file .env two-stage-fraud-api
```

Access the service at:

```bash
http://localhost:5000
```

### How it works

*   Historical fraud data is ingested into PostgreSQL

*   Stage 1 Logistic Regression model is trained offline

*   Prediction probability distribution is analyzed to identify uncertainty range

*   Uncertain samples are routed to Stage 2 XGBoost model

*   Models are saved as artifacts

*   FastAPI loads models once at startup

*   Incoming requests are processed via two-stage inference

*   Predictions and metadata are logged for auditing

---

## Requirements

*   Python 3.9 or higher

*   FastAPI

*   Uvicorn

*   Scikit-learn

*   XGBoost

*   Pandas and NumPy

*   PostgreSQL

*   SQLAlchemy

*   Docker