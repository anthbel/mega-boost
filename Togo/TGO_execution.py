# Databricks notebook source
# Builds the budget-execution tables for Togo, in pandas without Spark: the silver
# lines grouped by the dimension columns below, with the allocation and execution
# amounts summed. tgo_execution holds the dashboard's breakdowns; tgo_execution_detail
# keeps each budget line (imputation) with its action, services (admin3 to admin5),
# function (func3) and economic classification (econ1 to econ4). The only check is
# that every line gets a source of funding.
#
# Inputs
#   SILVER_CSV  tgo_2021_onward_boost_silver.csv written by TGO_ETL.py
#   OUTPUT_DIR  where tgo_execution.csv and tgo_execution_detail.csv go
# With DB_BACKEND=postgres neither is used: the silver table is read from
# boost_intermediate.tgo_2021_onward_boost_silver of the PostgreSQL database in
# POSTGRES_DSN and the outputs replace boost.tgo_execution and
# boost.tgo_execution_detail (postgres_tables.py).
import os

import numpy as np
import pandas as pd

COUNTRY = 'Togo'
DB_BACKEND = os.environ.get('DB_BACKEND', 'csv')
if DB_BACKEND == 'postgres':
    import postgres_tables
elif DB_BACKEND == 'csv':
    SILVER_CSV = os.environ.get('SILVER_CSV') or input("Path to tgo_2021_onward_boost_silver.csv: ").strip()
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR') or input("Output folder: ").strip()
else:
    raise RuntimeError(f"Unknown DB_BACKEND {DB_BACKEND!r}; expected csv or postgres.")

# Silver columns kept as dimensions, under their output names
DIMENSION_SOURCES = {
    'CODE_TYPE_BUDGET': 'budget_type_code', 'TYPE_BUDGET': 'budget_type',
    'CODE_ADMIN2': 'admin2_code', 'ADMIN2': 'admin2', 'ADMIN2_COURT': 'admin2_short',
    'CODE_PROG': 'prog_code', 'PROG': 'prog',
    'CODE_FUNC1': 'func_code', 'func': 'func', 'func_sub': 'func_sub',
    'CODE_ECON1': 'econ1_code', 'ECON1': 'econ1',
    'econ': 'econ',
    'CODE_GENRE': 'gender_code', 'GENRE': 'gender',
    'CODE_BUDGET_VERT': 'green_code', 'BUDGET_VERT': 'green',
    'CODE_DEPENSES_SOCIALES': 'social_code', 'DEPENSES_SOCIALES': 'social',
}
MEASURE_SOURCES = {
    'DOTATION_AE': 'dotation_ae', 'DOTATION_INITIALE': 'dotation_initiale',
    'DOTATION_FINALE': 'dotation_finale', 'ENGAGEMENT': 'engagement',
    'LIQUIDATION': 'liquidation', 'ORDONNANCER_RI': 'ordonnancer_ri',
    'ORDONNANCER_RE': 'ordonnancer_re', 'ORDONNANCER': 'ordonnancer',
}
DIMENSIONS = [
    'country_name', 'year',
    'budget_type_code', 'budget_type',
    'admin2_code', 'admin2', 'admin2_short',
    'prog_code', 'prog',
    'func_code', 'func', 'func_sub',
    'econ1_code', 'econ1',
    'econ',
    'funding_code', 'funding', 'pip', 'funder',
    'gender_code', 'gender', 'green_code', 'green', 'social_code', 'social',
    'project_code', 'project',
]
MEASURES = list(MEASURE_SOURCES.values())

# Silver columns the detail table adds to the fact table's keys
DETAIL_SOURCES = {
    'CODE_ACTION': 'action_code', 'ACTION': 'action',
    'CODE_ADMIN3': 'admin3_code', 'ADMIN3': 'admin3',
    'CODE_ADMIN4': 'admin4_code', 'ADMIN4': 'admin4',
    'CODE_ADMIN5': 'admin5_code', 'CODE_FUNC3': 'func3_code',
    'CODE_ECON2': 'econ2_code', 'ECON2': 'econ2',
    'CODE_ECON3': 'econ3_code', 'ECON3': 'econ3',
    'CODE_ECON4': 'econ4_code', 'ECON4': 'econ4',
    'CODE_IMPUTATION': 'imputation',
}
DETAIL_DIMENSIONS = [
    'country_name', 'year',
    'budget_type_code', 'budget_type',
    'admin2_code', 'admin2', 'admin2_short',
    'prog_code', 'prog', 'action_code', 'action',
    'admin3_code', 'admin3', 'admin4_code', 'admin4', 'admin5_code', 'func3_code',
    'econ1_code', 'econ1', 'econ2_code', 'econ2', 'econ3_code', 'econ3', 'econ4_code', 'econ4',
    'funding_code', 'funder', 'gender_code', 'green_code', 'social_code',
    'imputation',
]

# Source of funding, from the ADMIN5 label of internal resources and is_foreign
INTERNAL = 'FINANCEMENTS INTERNES'
INTERNAL_FUNDING = {
    'FINANCEMENTS INTERNES - HORS PROJETS': 'RI_HORS_PROJETS',
    'FINANCEMENTS INTERNES - PROJETS': 'RI_PROJETS',
    'FINANCEMENTS INTERNES - FONDS DE CONTREPARTIE': 'RI_CONTREPARTIE',
}
FUNDING_LABELS = {
    'RI_HORS_PROJETS': 'Ressources internes hors projets',
    'RI_PROJETS': 'Ressources internes, projets',
    'RI_CONTREPARTIE': 'Fonds de contrepartie',
    'RI_AUTRES': 'Ressources internes, autres',
    'RE': 'Ressources externes',
}
CAPITAL_ECON1 = '5'

# COMMAND ----------

source_columns = (list(DIMENSION_SOURCES) + list(MEASURE_SOURCES) + list(DETAIL_SOURCES)
                  + ['year', 'is_foreign', 'ADMIN5'])
if DB_BACKEND == 'postgres':
    silver = postgres_tables.read_table('prd_mega', 'boost_intermediate', 'tgo_2021_onward_boost_silver',
                                        columns=source_columns)
else:
    # codes are read as text so leading zeros are kept
    header = pd.read_csv(SILVER_CSV, nrows=0).columns
    silver = pd.read_csv(SILVER_CSV, usecols=source_columns, na_values=['null'],
                         dtype={c: str for c in header if c.startswith('CODE_')})

t = silver[list(DIMENSION_SOURCES) + list(MEASURE_SOURCES)].rename(columns={**DIMENSION_SOURCES, **MEASURE_SOURCES})
t['country_name'] = COUNTRY
t['year'] = silver['year'].astype(int)

admin5 = silver['ADMIN5']
is_foreign = silver['is_foreign'].astype(bool)
funding_code = pd.Series(np.select(
    [is_foreign]
    + [admin5 == label for label in INTERNAL_FUNDING]
    + [admin5.fillna('').str.startswith(INTERNAL)],
    ['RE'] + list(INTERNAL_FUNDING.values()) + ['RI_AUTRES'],
    default=''), index=silver.index)
unclassified = admin5[funding_code == '']
assert unclassified.empty, f'ADMIN5 values with no funding class: {sorted(unclassified.astype(str).unique())}'
t['funding_code'] = funding_code
t['funding'] = funding_code.map(FUNDING_LABELS)
t['pip'] = funding_code != 'RI_HORS_PROJETS'
t['funder'] = admin5.where(funding_code == 'RE')

# Projects are kept on capital lines only
capital = silver['CODE_ECON1'] == CAPITAL_ECON1
t['project_code'] = silver['CODE_ADMIN4'].where(capital)
t['project'] = silver['ADMIN4'].where(capital)

# COMMAND ----------

detail = pd.concat([t, silver[list(DETAIL_SOURCES)].rename(columns=DETAIL_SOURCES)], axis=1)

# COMMAND ----------

def group(frame, dimensions):
    # dropna=False keeps NULL dimension values as groups; sum(min_count=1) gives NULL
    # for a group with no amounts
    return (frame.groupby(dimensions, dropna=False)[MEASURES].sum(min_count=1)
            .reset_index()[dimensions + MEASURES])


outputs = {
    'tgo_execution': group(t, DIMENSIONS),
    'tgo_execution_detail': group(detail, DETAIL_DIMENSIONS),
}
for name, table in outputs.items():
    if DB_BACKEND == 'postgres':
        postgres_tables.replace_table(table, 'prd_mega', 'boost', name)
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        table.to_csv(os.path.join(OUTPUT_DIR, f'{name}.csv'), index=False)
    print(f'{len(table):>7} rows  {name} (from {len(silver)} silver rows)')
