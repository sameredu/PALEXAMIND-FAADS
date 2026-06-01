import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob

def parse_duration(duration_str):
    if pd.isna(duration_str) or duration_str == 'Not yet evaluated':
        return 0
    
    parts = str(duration_str).split()
    total_seconds = 0
    for i in range(0, len(parts), 2):
        try:
            val = int(parts[i])
            unit = parts[i+1].lower()
            if 'hour' in unit:
                total_seconds += val * 3600
            elif 'min' in unit:
                total_seconds += val * 60
            elif 'sec' in unit:
                total_seconds += val
        except:
            continue
    return total_seconds

def extract_features(file_path):
    df = pd.read_excel(file_path)
    exam_name = os.path.basename(file_path).replace('_Anonymized.xlsx', '')
    
    # Identify question columns
    q_cols = [c for c in df.columns if c.startswith('Q')]
    
    features_list = []
    
    for idx, row in df.iterrows():
        # F1: Inactivity Gap Variance (Simplified as variance of non-blank responses across timeline if available)
        # Since we don't have per-question timestamps, we use a proxy: distribution of unanswered questions
        unanswered_indices = [i for i, val in enumerate(row[q_cols]) if val == 'Unanswered']
        if len(unanswered_indices) > 1:
            igv = np.var(np.diff(unanswered_indices))
        else:
            igv = 0
            
        # F2: Initial Latency (Proxy: if Start_Time is significantly after exam opening - using 2:00 PM as baseline from README)
        try:
            start_time = pd.to_datetime(row['Start_Time'])
            # Standard start is usually at the hour, let's assume 2:00 PM for this dataset
            baseline = start_time.replace(minute=0, second=0)
            il = (start_time - baseline).total_seconds()
        except:
            il = 0
            
        # F3: Submission Efficiency (Ratio of answered questions to total duration)
        duration_sec = parse_duration(row['Duration'])
        answered_count = sum([1 for val in row[q_cols] if val != 'Unanswered' and val != 'Needs_Grading'])
        ser = answered_count / (duration_sec / 60) if duration_sec > 0 else 0
        
        # Labels for training (based on Taxonomy in README)
        # 1. Complete Disconnection: Finished but all questions blank
        is_p1 = (row['Attempt_Status'] == 'Finished' and all([val == 'Unanswered' for val in row[q_cols]]))
        # 2. Delayed Entry: Duration < 5 mins
        is_p2 = (duration_sec < 300 and duration_sec > 0)
        # 3. Partial Interruption: > 50% blank
        unanswered_count = sum([1 for val in row[q_cols] if val == 'Unanswered'])
        is_p3 = (unanswered_count / len(q_cols) > 0.5 and not is_p1)
        
        label = 'Normal'
        if is_p1: label = 'P1'
        elif is_p2: label = 'P2'
        elif is_p3: label = 'P3'
        
        features_list.append({
            'Student_Label': row['Student_Label'],
            'Exam': exam_name,
            'IGV': igv,
            'IL': il,
            'SER': ser,
            'Label': label
        })
        
    return pd.DataFrame(features_list)

# Process all files
all_features = []
files = glob.glob('/home/ubuntu/dataset/*.xlsx')
for f in files:
    print(f"Processing {f}...")
    all_features.append(extract_features(f))

final_df = pd.concat(all_features)

# Add BFD (Behavioral Fairness Divergence) and RIF (Infrastructure Vulnerability)
# RIF: Frequency of past technical issues (P1, P2, P3 labels)
rif_map = final_df.groupby('Student_Label')['Label'].apply(lambda x: sum([1 for l in x if l != 'Normal'])).to_dict()
final_df['RIF'] = final_df['Student_Label'].map(rif_map)

# BFD: Divergence from historical profile (L2 norm of current vector vs mean vector for that student)
# We calculate mean vector per student across exams
student_means = final_df.groupby('Student_Label')[['IGV', 'IL', 'SER']].transform('mean')
final_df['BFD'] = np.sqrt(((final_df[['IGV', 'IL', 'SER']] - student_means)**2).sum(axis=1))

final_df.to_csv('/home/ubuntu/processed_features.csv', index=False)
print("Feature engineering complete. Saved to /home/ubuntu/processed_features.csv")
