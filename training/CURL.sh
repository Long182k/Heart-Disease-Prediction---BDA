## lOW RISK
curl -X POST http://localhost:5002/api/predict -H "Content-Type: application/json" -d "{\"age_years\": 25, \"gender\": 1, \"height\": 175, \"weight\": 65, \"ap_hi\": 110, \"ap_lo\": 70, \"cholesterol\": 1, \"gluc\": 1, \"smoke\": 0, \"alco\": 0, \"active\": 1, \"bmi\": 21.2, \"bp_category\": \"Normal\"}"

## Moderate Risk
curl -X POST http://localhost:5002/api/predict -H "Content-Type: application/json" -d "{\"age_years\": 45, \"gender\": 1, \"height\": 170, \"weight\": 80, \"ap_hi\": 125, \"ap_lo\": 75, \"cholesterol\": 2, \"gluc\": 1, \"smoke\": 0, \"alco\": 1, \"active\": 0, \"bmi\": 27.7, \"bp_category\": \"Elevated\"}"

## High Risk
curl -X POST http://localhost:5002/api/predict -H "Content-Type: application/json" -d "{\"age_years\": 65, \"gender\": 1, \"height\": 160, \"weight\": 90, \"ap_hi\": 160, \"ap_lo\": 100, \"cholesterol\": 3, \"gluc\": 3, \"smoke\": 1, \"alco\": 1, \"active\": 0, \"bmi\": 35.2, \"bp_category\": \"Hypertension Stage 2\"}"
