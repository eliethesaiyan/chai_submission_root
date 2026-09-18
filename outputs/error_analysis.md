# Extraction error analysis

These are observed errors against weak labels. No clinician review is claimed.

Model: distilbert-clinical-v1-473c169b4cb6

## N003 (test)

60 year old male with chest pain radiating to left arm. Diagnosed myocardial infarction. Started aspirin and atorvastatin.

Missing: [{"label": "symptoms", "start": 22, "end": 54, "text": "chest pain radiating to left arm"}]

Spurious: [{"label": "diagnosis", "start": 46, "end": 50, "text": "left"}, {"label": "symptoms", "start": 22, "end": 42, "text": "chest pain radiating"}, {"label": "symptoms", "start": 51, "end": 54, "text": "arm"}]

Interpretation: Overlapping spans indicate a boundary or BIO fragmentation error; strict span scoring counts both a false positive and a false negative.

Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.

Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.

## N006 (test)

70 year old female with shortness of breath and wheezing. Diagnosed asthma exacerbation. Given salbutamol nebulization.

Missing: [{"label": "medications", "start": 95, "end": 105, "text": "salbutamol"}]

Spurious: [{"label": "medications", "start": 95, "end": 118, "text": "salbutamol nebulization"}]

Interpretation: Overlapping spans indicate a boundary or BIO fragmentation error; strict span scoring counts both a false positive and a false negative.

Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.

Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.

## N016 (test)

33 year old female with wheezing and shortness of breath. Asthma exacerbation. Given salbutamol.

Missing: [{"label": "diagnosis", "start": 58, "end": 77, "text": "Asthma exacerbation"}]

Spurious: [{"label": "diagnosis", "start": 65, "end": 77, "text": "exacerbation"}, {"label": "symptoms", "start": 58, "end": 64, "text": "Asthma"}]

Interpretation: Overlapping spans indicate a boundary or BIO fragmentation error; strict span scoring counts both a false positive and a false negative.

Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.

Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.

## N030 (test)

34 year old female with cough and chest pain. Diagnosed pneumonia. Started amoxicillin.

Missing: [{"label": "diagnosis", "start": 56, "end": 65, "text": "pneumonia"}]

Spurious: [{"label": "symptoms", "start": 56, "end": 65, "text": "pneumonia"}]

Interpretation: The model omitted or mislabeled a field. Sparse training examples and unseen context can explain this, but the cause is not proven.

Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.

Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.

## N004 (validation)

25 year old female with dysuria and lower abdominal pain. Diagnosed urinary tract infection. Prescribed ciprofloxacin.

Missing: [{"label": "symptoms", "start": 36, "end": 56, "text": "lower abdominal pain"}]

Spurious: [{"label": "symptoms", "start": 36, "end": 41, "text": "lower"}, {"label": "symptoms", "start": 42, "end": 56, "text": "abdominal pain"}]

Interpretation: Overlapping spans indicate a boundary or BIO fragmentation error; strict span scoring counts both a false positive and a false negative.

Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.

Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.

## N010 (validation)

40 year old female with severe headache and blurred vision. Diagnosed migraine. Given paracetamol.

Missing: [{"label": "symptoms", "start": 24, "end": 39, "text": "severe headache"}]

Spurious: [{"label": "symptoms", "start": 24, "end": 30, "text": "severe"}, {"label": "symptoms", "start": 31, "end": 39, "text": "headache"}]

Interpretation: Overlapping spans indicate a boundary or BIO fragmentation error; strict span scoring counts both a false positive and a false negative.

Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.

Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.

## N026 (validation)

71 year old male with wheezing. Asthma exacerbation. Given salbutamol.

Missing: [{"label": "diagnosis", "start": 32, "end": 51, "text": "Asthma exacerbation"}]

Spurious: [{"label": "diagnosis", "start": 39, "end": 51, "text": "exacerbation"}, {"label": "symptoms", "start": 32, "end": 38, "text": "Asthma"}]

Interpretation: Overlapping spans indicate a boundary or BIO fragmentation error; strict span scoring counts both a false positive and a false negative.

Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.

Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.

## N037 (validation)

46 year old female with fever and cough. Diagnosed pneumonia. Given azithromycin.

Missing: [{"label": "diagnosis", "start": 51, "end": 60, "text": "pneumonia"}]

Spurious: [{"label": "symptoms", "start": 51, "end": 60, "text": "pneumonia"}]

Interpretation: The model omitted or mislabeled a field. Sparse training examples and unseen context can explain this, but the cause is not proven.

Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.

Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.

## N044 (validation)

53 year old male with cough, fever. Diagnosed pneumonia. Given amoxicillin.

Missing: [{"label": "diagnosis", "start": 46, "end": 55, "text": "pneumonia"}]

Spurious: [{"label": "symptoms", "start": 46, "end": 55, "text": "pneumonia"}]

Interpretation: The model omitted or mislabeled a field. Sparse training examples and unseen context can explain this, but the cause is not proven.

Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.

Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.

## N011 (train)

28 year old male with fever and cough. Chest exam suggest pneumonia. Started azithromycin.

Missing: [{"label": "diagnosis", "start": 58, "end": 67, "text": "pneumonia"}]

Spurious: [{"label": "symptoms", "start": 58, "end": 67, "text": "pneumonia"}]

Interpretation: The model omitted or mislabeled a field. Sparse training examples and unseen context can explain this, but the cause is not proven.

Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.

Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.

## Common failure modes

Subword fragmentation; symptom-boundary ambiguity; unseen diagnosis/medication names; negation and historical mentions; multiple diagnoses. The rule-based assertion filter is separate from learned span detection and does not establish general negation accuracy.

Tests inferred from mentions use conservative sentence exclusions. 'Not documented' is not proof that a test was not performed.

Grouped splitting removes exact demographic-only duplicates, but near-duplicate templates and weak-label bias remain.
