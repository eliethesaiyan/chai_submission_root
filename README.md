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
this step can be reproduced as follows:
```bash 
./run.sh annotate
```