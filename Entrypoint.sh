#!/bin/bash
# entrypoint.sh
# Runs on every container start:
# 1. Seeds PT-8839-CR data into the DB
# 2. Starts the KLM endpoint server

set -e

echo "=== Patient KLM Startup ==="
echo "Database: $PATIENT_KLM_DB_PATH"

# Seed Demo #2 patient (safe to run every time — uses INSERT OR REPLACE)
echo ""
echo "Seeding PT-8839-CR data..."
python seed_demo2.py

echo ""
echo "Starting endpoint server..."
python patient_klm_endpoint.py
