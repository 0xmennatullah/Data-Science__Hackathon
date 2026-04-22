# 📊 Data Science Projects — Hackathon

A total of 12 distinct data issues were identified and remediated, including critical column-shift errors (CS-001, CS-002), brand-name fragmentation (MF-001), a dual-scale encoding error in ConditionValue, and four bugs in the original flag-generation logic. Miss-
ing values were classified under the MCAR/MAR/MNAR statistical framework, and
each column received a statistically appropriate imputation strategy. Eight additional issues (P-001 through P-008) were identified and patched, the most impactful being the replacement of a O(n2) KNN imputation (runtime ≈10 min) with a vectorised grouped probabilistic approach (≈5 sec, statistically equivalent for this MAR structure)
Post-remediation: 558,837 input records → ≈548,000 usable records
