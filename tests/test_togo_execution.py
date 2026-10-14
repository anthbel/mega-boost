"""TGO_execution builds the budget-execution fact table from the silver table.

The script runs end to end through its CSV path on a small hand-made silver table
(all values synthetic) and the output is checked for its columns, the measure
totals, the funding classification and the grouping.
"""
import csv
import os
import shutil
import unittest

import pandas as pd

from helpers import REPO_ROOT, run_script

NB = "Togo/TGO_execution.py"
TMP = os.path.join(REPO_ROOT, "tests", ".fixtures_tmp", "togo_execution")

DIMENSIONS = [
    "country_name", "year",
    "budget_type_code", "budget_type",
    "admin2_code", "admin2", "admin2_short",
    "prog_code", "prog",
    "func_code", "func", "func_sub",
    "econ1_code", "econ1",
    "econ",
    "funding_code", "funding", "pip", "funder",
    "gender_code", "gender", "green_code", "green", "social_code", "social",
    "project_code", "project",
]
MEASURES = ["dotation_ae", "dotation_initiale", "dotation_finale", "engagement",
            "liquidation", "ordonnancer_ri", "ordonnancer_re", "ordonnancer"]
SOURCE_MEASURES = ["DOTATION_AE", "DOTATION_INITIALE", "DOTATION_FINALE", "ENGAGEMENT",
                   "LIQUIDATION", "ORDONNANCER_RI", "ORDONNANCER_RE", "ORDONNANCER"]

HP = "FINANCEMENTS INTERNES - HORS PROJETS"
PROJ = "FINANCEMENTS INTERNES - PROJETS"
FC = "FINANCEMENTS INTERNES - FONDS DE CONTREPARTIE"

BASE = {
    "CODE_TYPE_BUDGET": "1", "TYPE_BUDGET": "BUDGET GENERAL",
    "CODE_ADMIN2": "010", "ADMIN2": "Ministry A", "ADMIN2_COURT": "MA",
    "CODE_ADMIN4": "0000000000001", "ADMIN4": "Project A",
    "CODE_PROG": "003", "PROG": "Programme 3",
    "CODE_FUNC1": "01", "FUNC1": "SERVICES GENERAUX", "func": "General public services", "func_sub": None,
    "CODE_ECON1": "2", "ECON1": "PERSONNEL", "econ": "Wage bill",
    "ADMIN5": HP, "is_foreign": False,
    "CODE_GENRE": "1", "GENRE": "Gender 1",
    "CODE_BUDGET_VERT": "0", "BUDGET_VERT": "Green 0",
    "CODE_DEPENSES_SOCIALES": "1", "DEPENSES_SOCIALES": "Social 1",
    "year": 2025,
}
CAPITAL = {"CODE_ECON1": "5", "ECON1": "DEPENSES EN CAPITAL", "econ": "Capital expenditures"}


def row(measure_base, **overrides):
    values = dict(BASE, **overrides)
    for i, name in enumerate(SOURCE_MEASURES):
        values[name] = measure_base + i
    return values


# Rows 0 and 1 share every dimension and must merge; row 2 differs from them only
# by a NULL gender and row 3 only by a func_sub, so both stay separate groups.
SILVER_ROWS = [
    row(100),
    row(1000),
    row(10_000, CODE_GENRE=None, GENRE=None),
    row(20_000, func_sub="Judiciary"),
    row(30_000, CODE_FUNC1="09", func="Education", func_sub="Primary Education", year=2024),
    row(40_000, **CAPITAL),
    row(50_000, **CAPITAL, ADMIN5=PROJ, CODE_ADMIN4="0000000000002", ADMIN4="Project B"),
    row(60_000, **CAPITAL, ADMIN5=FC),
    row(70_000, **CAPITAL, ADMIN5="FINANCEMENTS INTERNES"),
    row(80_000, **CAPITAL, ADMIN5="BANQUE MONDIALE/DONS", is_foreign=True),
    row(90_000, ADMIN5="BANQUE MONDIALE/EMPRUNTS", is_foreign=True),
]


def code_dtypes(columns, suffix):
    return {c: str for c in columns if c.endswith(suffix)}


class TogoExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)
        os.makedirs(TMP)
        cls.silver = pd.DataFrame(SILVER_ROWS)
        silver_csv = os.path.join(TMP, "tgo_2021_onward_boost_silver.csv")
        # written the way TGO_ETL writes the silver CSV
        cls.silver.to_csv(silver_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
        run_script(NB, env={"DB_BACKEND": "csv", "SILVER_CSV": silver_csv, "OUTPUT_DIR": TMP})
        out = os.path.join(TMP, "tgo_execution.csv")
        cls.out = pd.read_csv(out, dtype=code_dtypes(pd.read_csv(out, nrows=0).columns, "_code"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def lines(self, **filters):
        mask = pd.Series(True, index=self.out.index)
        for column, value in filters.items():
            mask &= self.out[column].isna() if value is None else self.out[column] == value
        return self.out[mask]

    def test_columns_in_order(self):
        self.assertEqual(list(self.out.columns), DIMENSIONS + MEASURES)

    def test_measure_totals_match_silver(self):
        for target, source in zip(MEASURES, SOURCE_MEASURES):
            with self.subTest(measure=target):
                self.assertEqual(self.out[target].sum(), self.silver[source].sum())
        by_year = self.out.groupby("year")["ordonnancer"].sum().to_dict()
        self.assertEqual(by_year, self.silver.groupby("year")["ORDONNANCER"].sum().to_dict())

    def test_identical_dimensions_merge_and_null_keys_stay_separate(self):
        self.assertEqual(len(self.out), len(SILVER_ROWS) - 1)
        merged = self.lines(year=2025, econ1_code="2", funding_code="RI_HORS_PROJETS",
                            gender_code="1", func_sub=None)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged["dotation_ae"].iloc[0], 100 + 1000)
        self.assertEqual(merged["ordonnancer"].iloc[0], 107 + 1007)
        null_gender = self.lines(year=2025, econ1_code="2", funding_code="RI_HORS_PROJETS",
                                 gender_code=None)
        self.assertEqual(null_gender["dotation_ae"].tolist(), [10_000])
        self.assertTrue(null_gender["gender"].isna().all())
        judiciary = self.lines(year=2025, econ1_code="2", func_sub="Judiciary")
        self.assertEqual(judiciary["dotation_ae"].tolist(), [20_000])

    def test_codes_keep_leading_zeros(self):
        self.assertEqual(set(self.out["admin2_code"]), {"010"})
        self.assertEqual(set(self.out["prog_code"]), {"003"})
        self.assertEqual(set(self.out["func_code"]), {"01", "09"})
        self.assertEqual(set(self.out["budget_type_code"]), {"1"})
        self.assertEqual(set(self.out["green_code"]), {"0"})

    def test_renamed_dimensions(self):
        line = self.lines(year=2024).iloc[0]
        self.assertEqual(line["country_name"], "Togo")
        self.assertEqual(
            line[["budget_type", "admin2", "admin2_short", "prog", "func", "func_sub",
                  "econ1", "econ", "gender", "green", "social"]].tolist(),
            ["BUDGET GENERAL", "Ministry A", "MA", "Programme 3", "Education", "Primary Education",
             "PERSONNEL", "Wage bill", "Gender 1", "Green 0", "Social 1"])
        self.assertEqual(line["social_code"], "1")

    def test_funding_classification(self):
        capital = self.lines(econ1_code="5").set_index("funding_code")
        self.assertEqual(sorted(capital.index),
                         ["RE", "RI_AUTRES", "RI_CONTREPARTIE", "RI_HORS_PROJETS", "RI_PROJETS"])
        self.assertEqual(capital["funding"].to_dict(), {
            "RI_HORS_PROJETS": "Ressources internes hors projets",
            "RI_PROJETS": "Ressources internes, projets",
            "RI_CONTREPARTIE": "Fonds de contrepartie",
            "RI_AUTRES": "Ressources internes, autres",
            "RE": "Ressources externes",
        })
        self.assertEqual(capital["dotation_ae"].to_dict(), {
            "RI_HORS_PROJETS": 40_000, "RI_PROJETS": 50_000, "RI_CONTREPARTIE": 60_000,
            "RI_AUTRES": 70_000, "RE": 80_000})
        self.assertEqual(capital["pip"].to_dict(), {
            "RI_HORS_PROJETS": False, "RI_PROJETS": True, "RI_CONTREPARTIE": True,
            "RI_AUTRES": True, "RE": True})
        self.assertEqual(capital.loc["RE", "funder"], "BANQUE MONDIALE/DONS")
        self.assertTrue(capital.drop(index="RE")["funder"].isna().all())

    def test_foreign_recurrent_line_has_funder_and_no_project(self):
        line = self.lines(econ1_code="2", funding_code="RE")
        self.assertEqual(line["funder"].tolist(), ["BANQUE MONDIALE/EMPRUNTS"])
        self.assertEqual(line["pip"].tolist(), [True])
        self.assertTrue(line["project_code"].isna().all())

    def test_project_only_on_capital_lines(self):
        self.assertTrue(self.lines(econ1_code="2")[["project_code", "project"]].isna().all().all())
        capital = self.lines(econ1_code="5").set_index("funding_code")
        self.assertEqual(capital.loc["RI_PROJETS", ["project_code", "project"]].tolist(),
                         ["0000000000002", "Project B"])
        self.assertEqual(capital.loc["RE", ["project_code", "project"]].tolist(),
                         ["0000000000001", "Project A"])


if __name__ == "__main__":
    unittest.main()
