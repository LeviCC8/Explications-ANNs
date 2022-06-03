import pandas as pd
import os

# comparing repair algorithms
path = 'heuristic_results'
datasets = []
repair1_len = []
repair2_len = []
repair1_time = []
repair2_time = []
for dataset in os.listdir(path):
    if dataset.endswith('.csv'):
        continue
    results = os.listdir(os.path.join(path, dataset))
    for r in results:
        if r.endswith('global.csv'):
            df = pd.read_csv(os.path.join(path, dataset, r), usecols=['repair_time', 'repair2_time', 'len_list', 'len_list2'])
            df_repair = df.loc[df['repair_time'] != 0]
            mins = df_repair.min()
            maxs = df_repair.max()
            means = df_repair.mean()
            stds = df_repair.std()
            repair1_len.extend([mins['len_list'], f"{means['len_list']} +- {stds['len_list']}", maxs['len_list']])
            repair2_len.extend([mins['len_list2'], f"{means['len_list2']} +- {stds['len_list2']}", maxs['len_list2']])
            repair1_time.extend([mins['repair_time'], f"{means['repair_time']} +- {stds['repair_time']}", maxs['repair_time']])
            repair2_time.extend([mins['repair2_time'], f"{means['repair2_time']} +- {stds['repair2_time']}", maxs['repair2_time']])
            datasets.extend([f'{dataset}_m', f'{dataset}_a', f'{dataset}_M'])

df = {'repair1_len': repair1_len, 'repair1_time': repair1_time, 'repair2_len': repair2_len, 'repair2_time': repair2_time}

df = pd.DataFrame(data=df, index=datasets)
df.to_csv(f"heuristic_results\\results_repair_global.csv")


# comparing pipelines
path = 'heuristic_results'
datasets = []
global_abductive_len = []
global_abductive_time = []
local_abductive_len = []
local_abductive_time = []
global_pipeline_len = []
global_pipeline_time = []
local_pipeline_len = []
local_pipeline_time = []
for dataset in os.listdir(path):
    if dataset.endswith('.csv'):
        continue
    datasets.extend([f'{dataset}_m', f'{dataset}_a', f'{dataset}_M'])
    results = os.listdir(os.path.join(path, dataset))
    for r in results:
        if r.endswith('global.csv'):
            df = pd.read_csv(os.path.join(path, dataset, r))
            mins = df.min()
            maxs = df.max()
            means = df.mean()
            stds = df.std()
            global_abductive_len.extend([mins['len_abductive'], f"{means['len_abductive']} +- {stds['len_abductive']}", maxs['len_abductive']])
            global_abductive_time.extend([mins['time_abductive'], f"{means['time_abductive']} +- {stds['time_abductive']}", maxs['time_abductive']])
            global_pipeline_len.extend([mins['len_list2'], f"{means['len_list2']} +- {stds['len_list2']}", maxs['len_list2']])
            pipeline_time = df['time_heuristic'] + df['valid_time'] + df['repair2_time'] + df['refine_time']
            global_pipeline_time.extend([pipeline_time.min(), f"{pipeline_time.mean()} +- {pipeline_time.std()}", pipeline_time.max()])
        elif r.endswith('local.csv'):
            df = pd.read_csv(os.path.join(path, dataset, r))
            mins = df.min()
            maxs = df.max()
            means = df.mean()
            stds = df.std()
            local_abductive_len.extend([mins['len_abductive'], f"{means['len_abductive']} +- {stds['len_abductive']}", maxs['len_abductive']])
            local_abductive_time.extend([mins['time_abductive'], f"{means['time_abductive']} +- {stds['time_abductive']}", maxs['time_abductive']])
            local_pipeline_len.extend([mins['len_list2'], f"{means['len_list2']} +- {stds['len_list2']}", maxs['len_list2']])
            pipeline_time = df['time_heuristic'] + df['valid_time'] + df['repair2_time'] + df['refine_time']
            local_pipeline_time.extend([pipeline_time.min(), f"{pipeline_time.mean()} +- {pipeline_time.std()}", pipeline_time.max()])

df = {'global_abductive_len': global_abductive_len, 'global_abductive_time': global_abductive_time, 'local_abductive_len': local_abductive_len, 'local_abductive_time': local_abductive_time,
      'global_pipeline_len': global_pipeline_len, 'global_pipeline_time': global_pipeline_time, 'local_pipeline_len': local_pipeline_len, 'local_pipeline_time': local_pipeline_time}

df = pd.DataFrame(data=df, index=datasets)
df.to_csv(f"heuristic_results\\results_pipelines.csv")
