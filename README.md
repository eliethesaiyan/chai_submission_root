# chai_submission_root
# Clinical-AI
### About
This projects consists of processing clinical notes from text into structured entities as data annoations.
The following steps were followed:
#### 1. Preprocessing.
1. Using regex to extract age and sex entities.
2. Using regex to infers diagnosis using keywords like `with` and `diagnosed`.
3. Drop the concepts of the duration of the symptoms.
4. Building a vocabulary f the diagonses terms from clincal notes.
5. Building a  medications from guidelines.

This step can be reproduced as follows:

```console 
$ ./run.sh annotate
```
It creates model data annotation provides a data analsysis summary as follow:
```console
[2026-09-08 00:10:15,464][__main__][INFO] - Dataset Analysis Summary
[2026-09-08 00:10:15,464][__main__][INFO] - ==================================================
[2026-09-08 00:10:15,464][__main__][INFO] - Total Clinical Notes:50
[2026-09-08 00:10:15,464][__main__][INFO] - Total Guidelines: 19
[2026-09-08 00:10:15,464][__main__][INFO] - Demographics Summary:
[2026-09-08 00:10:15,464][__main__][INFO] -   Max Age: 72
[2026-09-08 00:10:15,464][__main__][INFO] -   Minimum Age: 5
[2026-09-08 00:10:15,464][__main__][INFO] -   Male Count: 26
[2026-09-08 00:10:15,464][__main__][INFO] -   Female Count: 24
[2026-09-08 00:10:15,464][__main__][INFO] - Diagnosis in clinical Counts Summary: 3 diags examples
[2026-09-08 00:10:15,464][__main__][INFO] -   pneumonia: 7 recorded in clinical notes.
[2026-09-08 00:10:15,464][__main__][INFO] -   meningitis: 5 recorded in clinical notes.
[2026-09-08 00:10:15,464][__main__][INFO] -   myocardial infarction: 5 recorded in clinical notes.
[2026-09-08 00:10:15,464][__main__][INFO] - Medications in clinical note Counts Summary: 3 meds examples
[2026-09-08 00:10:15,464][__main__][INFO] -   nitrofurantoin: 2 recorded in clinical notes .
[2026-09-08 00:10:15,464][__main__][INFO] -   ciprofloxacin: 3 recorded in clinical notes .
[2026-09-08 00:10:15,464][__main__][INFO] -   azithromycin: 7 recorded in clinical notes .
```