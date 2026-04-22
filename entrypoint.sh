#!/bin/bash
set -e

echo "=== Patient KLM Startup ==="
echo "Database: $PATIENT_KLM_DB_PATH"

echo ""
echo "Seeding PT-8839-CR (Demo 2)..."
python seed_demo2.py

echo ""
echo "Seeding P-003 (Demo 3)..."
python demo_3/seed_p003.py

echo ""
echo "Seeding PT-9921 (Dermatology)..."
python seed_demo3.py

echo ""
echo "Starting endpoint server..."
python patient_klm_endpoint.py
