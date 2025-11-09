🌍 Predicting Exoplanet Habitability using Probabilistic Models

A **Statistical Machine Learning Lab Project** aimed at predicting whether an exoplanet is **habitable** or **non-habitable** using NASA’s Exoplanet Archive data.  
This project demonstrates the use of **probabilistic models** (Naïve Bayes and Logistic Regression) and basic astrophysical criteria (Kopparapu et al., 2013) to explore data-driven habitability classification.

---

## 🧠 Overview

The project classifies planets as **habitable (1)** or **non-habitable (0)** based on simple physical conditions:

\[
\text{Habitable if: } (pl\_rade \le 1.8) \land (0.35 \le pl\_insol \le 1.5)
\]

### Features Used
| Feature | Description | Units |
|----------|--------------|-------|
| `pl_rade` | Planet Radius | Earth radii |
| `pl_bmasse` | Planet Mass | Earth masses |
| `pl_orbsmax` | Orbit Semi-Major Axis | AU |
| `pl_insol` | Insolation Flux | Earth flux |
| `st_teff` | Stellar Temperature | Kelvin |
| `st_rad` | Stellar Radius | Solar radii |

---

## 📂 Project Structure

ml project/
│
├── data/
│ ├── exoplanet_data.csv # Raw NASA dataset
│ ├── exoplanet_cleaned.csv # Preprocessed dataset
│ ├── feature_scaler.pkl # StandardScaler
│ ├── naive_bayes_model.pkl # Trained Naïve Bayes model
│ ├── log_reg_calibrated.pkl # Calibrated Logistic Regression model
│
├── notebook/
│ ├── data_prep.ipynb # Data cleaning + model training
│
├── results/
│ ├── edge_case_results.csv # Model performance on test planets
│
├── demo_app.py # Streamlit app for live demo
├── test_edge_cases.py # Automated test script
├── requirements.txt # Required Python packages
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/exoplanet-habitability.git
cd exoplanet-habitability
2️⃣ Create and activate virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Streamlit app
bash
Copy code
streamlit run demo_app.py
5️⃣ Test edge cases (automated evaluation)
bash
Copy code
python test_edge_cases.py
🧩 Models Used
Model	Description	Type
Gaussian Naïve Bayes	Simple baseline model assuming feature independence	Probabilistic
Logistic Regression (Balanced + Calibrated)	Regularized logistic classifier with class balancing and Platt calibration	Probabilistic

📊 Evaluation Metrics
Accuracy

Precision

Recall

F1-score

ROC-AUC

Brier Score

Logistic Regression (Calibrated) Results:
Metric	Score
Accuracy	0.943
Precision	0.118
Recall	1.000
F1-score	0.211
ROC-AUC	0.986

🪐 Edge Case Testing
Case	Naïve Bayes	Logistic (Calibrated)	Verdict
Earth-like	1.000	0.168	✅ Habitable
Hot Jupiter	0.000	0.000	❌ Non-habitable
Frozen Giant	1.000	1.000	⚠️ Misclassified (Data bias)
Close Rocky	0.000	0.001	❌ Non-habitable
M-dwarf Zone	1.000	0.193	✅ Habitable
A-type Star	0.000	0.000	❌ Non-habitable
Borderline Planet	1.000	0.301	✅ Possibly habitable

🔍 Observation:
The models correctly identify most planets, but both misclassify extremely cold “Frozen Giants” as habitable — likely due to dataset bias and absence of physical constraints.

📖 Key Learnings
Naïve Bayes tends to overestimate probabilities due to independence assumptions.

Logistic Regression (Calibrated) produces smoother, realistic probabilities.

Dataset noise and missing values can lead to misleading signals (data bias).

Physically informed post-rules (e.g., insolation limits) can improve reliability.

🧮 References
Kopparapu, R. K. et al. (2013). Habitable Zones around Main-sequence Stars: New Estimates.

NASA Exoplanet Archive — https://exoplanetarchive.ipac.caltech.edu

scikit-learn Documentation — https://scikit-learn.org

✍️ Author
Aditya Kumar Jaiswal
B.Tech CSE — Bennett University
📚 Statistical Machine Learning Lab (2025)
🌐 GitHub Profile

yaml
Copy code

---

✅ After pasting:
- Save it as `README.md` in your main folder.  
- Replace `<your-username>` with your actual GitHub username before committing.  

Once done, your repo will look **super clean and professional** on GitHub.

Want me to help you write your **final 4-page project report (PDF-ready)** next? I can generate it w