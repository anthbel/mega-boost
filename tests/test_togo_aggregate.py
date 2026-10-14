"""TGO_aggregate's expenditure_and_outcome_by_country_geo1_func_year, through the CSV path.

TGO_ETL and TGO_aggregate run end to end on a synthetic workbook with education and
health lines in every region for 2021 and 2022. The Global Data Lab fixture is shaped
like the real Togo series: rows run to 2023, school attendance stops in 2020 and the
health index is set in later years, with a gap in one region and a late start in
another. GDL's Lome and Total regions have no BOOST region. All values are synthetic.
"""
import os
import shutil
import unittest

import numpy as np
import pandas as pd

from helpers import REPO_ROOT, run_script, write_workbook
from test_togo_etl import COLS, ROWS_2021, ROWS_2022, SHEET_NAME
from test_togo_postgres import INDICATORS, REGIONS

TMP = os.path.join(REPO_ROOT, "tests", ".fixtures_tmp", "togo_aggregate")
TABLE = "expenditure_and_outcome_by_country_geo1_func_year"

WORKBOOK_REGIONS = ["REGION CENTRALE", "REGION DE LA KARA", "REGION MARITIME",
                    "REGION DES PLATEAUX", "REGION DES SAVANES"]


def line(year, region, func1, func2, executed):
    return [year, "A20", "Unit 20", "FINANCEMENTS INTERNES", region, "Pref 20", func1, func2, func2 + "0",
            "3", "E0", "E000", "E00000", "D000000000020", executed, executed, executed]


# Education spending falls and health spending rises from Centrale to Savanes, so the
# per-capita ranks run in opposite directions.
REGIONAL_ROWS = [
    row
    for year in (2021, 2022)
    for i, region in enumerate(WORKBOOK_REGIONS)
    for row in (line(year, region, "09", "091", (5 - i) * 1_000_000),
                line(year, region, "07", "073", (i + 1) * 1_000_000))
]

GDL_YEARS = [2019, 2020, 2021, 2022, 2023]
NAN = np.nan
ATTENDANCE = {
    "Centrale": [0.55, 0.60, NAN, NAN, NAN],
    "Kara":     [0.66, 0.70, NAN, NAN, NAN],
    "Lome":     [0.90, 0.95, NAN, NAN, NAN],
    "Maritime": [0.75, 0.80, NAN, NAN, NAN],
    "Plateaux": [0.62, 0.65, NAN, NAN, NAN],
    "Savanes":  [0.45, 0.50, NAN, NAN, NAN],
    "Total":    [0.65, 0.70, NAN, NAN, NAN],
}
HEALTH = {
    "Centrale": [0.50, 0.51, 0.52, 0.53, 0.99],
    "Kara":     [0.60, 0.61, 0.62, NAN, 0.99],
    "Lome":     [0.90, 0.91, 0.92, 0.93, 0.99],
    "Maritime": [0.55, 0.56, 0.57, 0.58, 0.99],
    "Plateaux": [0.45, 0.46, 0.47, 0.48, 0.99],
    "Savanes":  [NAN, NAN, NAN, 0.70, 0.99],
    "Total":    [0.55, 0.56, 0.57, 0.58, 0.99],
}
HD_INDEX = pd.DataFrame([
    {"country_name": "Togo", "adm1_name": region, "year": year,
     "attendance_6to17yo": ATTENDANCE[region][i], "health_index": HEALTH[region][i]}
    for region in ATTENDANCE for i, year in enumerate(GDL_YEARS)
])

# (func, year, region): (outcome_year, outcome_index, rank_per_capita_real_exp, rank_outcome_index)
EXPECTED = {
    **{("Education", year, region): (2020, ATTENDANCE[region][1], pc_rank, outcome_rank)
       for year in (2021, 2022)
       for region, pc_rank, outcome_rank in [("Centrale", 1, 4), ("Kara", 2, 2), ("Maritime", 3, 1),
                                             ("Plateaux", 4, 3), ("Savanes", 5, 5)]},
    ("Health", 2021, "Centrale"): (2021, 0.52, 4, 3),
    ("Health", 2021, "Kara"):     (2021, 0.62, 3, 1),
    ("Health", 2021, "Maritime"): (2021, 0.57, 2, 2),
    ("Health", 2021, "Plateaux"): (2021, 0.47, 1, 4),
    ("Health", 2021, "Savanes"):  (None, None, None, None),
    ("Health", 2022, "Centrale"): (2022, 0.53, 5, 4),
    ("Health", 2022, "Kara"):     (2021, 0.62, 4, 2),
    ("Health", 2022, "Maritime"): (2022, 0.58, 3, 3),
    ("Health", 2022, "Plateaux"): (2022, 0.48, 2, 5),
    ("Health", 2022, "Savanes"):  (2022, 0.70, 1, 1),
}

COLUMNS = [
    "country_name", "adm1_name", "func", "year", "expenditure", "budget", "cpi_factor", "population",
    "real_expenditure", "real_budget", "latest_year", "earliest_year",
    "per_capita_expenditure", "per_capita_real_expenditure", "per_capita_budget", "per_capita_real_budget",
    "adm1_name_for_map", "spent_in_region",
    "outcome_index", "rank_per_capita_real_exp", "rank_outcome_index", "outcome_year",
]


class TogoAggregateOutcomeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)
        input_dir = os.path.join(TMP, "input")
        etl_out = os.path.join(TMP, "etl")
        indicator_dir = os.path.join(TMP, "indicator")
        cls.out = os.path.join(TMP, "aggregate")
        for d in (input_dir, etl_out, indicator_dir):
            os.makedirs(d)
        write_workbook(os.path.join(input_dir, "BUDGET.xlsx"),
                       {SHEET_NAME: [COLS] + ROWS_2021 + ROWS_2022 + REGIONAL_ROWS})
        for name, df in {**INDICATORS, "global_data_lab_hd_index": HD_INDEX}.items():
            df.to_csv(os.path.join(indicator_dir, f"{name}.csv"), index=False)

        env = {"DB_BACKEND": "csv"}
        run_script("Togo/TGO_ETL.py", env={**env, "INPUT_DIR": input_dir, "OUTPUT_DIR": etl_out})
        run_script("Togo/TGO_aggregate.py", env={
            **env, "GOLD_CSV": os.path.join(etl_out, "tgo_boost_gold.csv"),
            "INDICATOR_DIR": indicator_dir, "OUTPUT_DIR": cls.out})
        path = os.path.join(cls.out, f"{TABLE}.csv")
        cls.table = pd.read_csv(path)
        cls.text = pd.read_csv(path, dtype=str, keep_default_na=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_columns_keep_their_order_with_outcome_year_last(self):
        self.assertEqual(list(self.table.columns), COLUMNS)

    def test_each_spending_year_takes_latest_outcome_year_with_a_value(self):
        t = self.table[self.table["func"].isin(["Education", "Health"])
                       & self.table["adm1_name"].isin(REGIONS)]
        actual = {}
        for row in t.itertuples():
            values = (row.outcome_year, row.outcome_index, row.rank_per_capita_real_exp, row.rank_outcome_index)
            actual[(row.func, row.year, row.adm1_name)] = tuple(None if pd.isna(v) else v for v in values)
        self.assertEqual(actual, EXPECTED)

    def test_outcome_year_is_an_integer_column(self):
        self.assertEqual(sorted(set(self.text["outcome_year"])), ["", "2020", "2021", "2022"])

    def test_rows_without_an_outcome_are_kept_without_one(self):
        no_outcome = ~(self.table["func"].isin(["Education", "Health"]) & self.table["adm1_name"].isin(REGIONS))
        self.assertTrue(no_outcome.any())
        outcome_columns = ["outcome_index", "outcome_year", "rank_per_capita_real_exp", "rank_outcome_index"]
        self.assertTrue(self.table.loc[no_outcome, outcome_columns].isna().all().all())
        self.assertIn("Central Scope", set(self.table.loc[no_outcome, "adm1_name"]))

    def test_gdl_regions_without_a_boost_region_are_left_out(self):
        self.assertEqual(set(self.table["adm1_name"]), set(REGIONS) | {"Central Scope"})
        self.assertFalse(self.table.duplicated(["adm1_name", "func", "year"]).any())


if __name__ == "__main__":
    unittest.main()
