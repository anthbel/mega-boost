"""TGO_execution builds the budget-execution fact table and its line-level detail
table from the silver table.

The script runs end to end through its CSV path on a small hand-made silver table
(all values synthetic) and both outputs are checked for their columns, the measure
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
DETAIL_DIMENSIONS = [
    "country_name", "year",
    "budget_type_code", "budget_type",
    "admin2_code", "admin2", "admin2_short",
    "prog_code", "prog", "action_code", "action",
    "admin3_code", "admin3", "admin4_code", "admin4", "admin5_code", "func3_code",
    "econ1_code", "econ1", "econ2_code", "econ2", "econ3_code", "econ3", "econ4_code", "econ4",
    "funding_code", "funder", "gender_code", "green_code", "social_code",
    "imputation",
]
SOURCE_MEASURES = ["DOTATION_AE", "DOTATION_INITIALE", "DOTATION_FINALE", "ENGAGEMENT",
                   "LIQUIDATION", "ORDONNANCER_RI", "ORDONNANCER_RE", "ORDONNANCER"]

HP = "FINANCEMENTS INTERNES - HORS PROJETS"
PROJ = "FINANCEMENTS INTERNES - PROJETS"
FC = "FINANCEMENTS INTERNES - FONDS DE CONTREPARTIE"

BASE = {
    "CODE_TYPE_BUDGET": "1", "TYPE_BUDGET": "BUDGET GENERAL",
    "CODE_ADMIN2": "010", "ADMIN2": "Ministry A", "ADMIN2_COURT": "MA",
    "CODE_ADMIN3": "11", "ADMIN3": "Cabinet structures",
    "CODE_ADMIN4": "0000000000001", "ADMIN4": "Project A",
    "CODE_PROG": "003", "PROG": "Programme 3", "CODE_ACTION": "0031", "ACTION": "Action 31",
    "CODE_FUNC1": "01", "FUNC1": "SERVICES GENERAUX", "func": "General public services", "func_sub": None,
    "CODE_FUNC3": "0160",
    "CODE_ECON1": "2", "ECON1": "PERSONNEL", "econ": "Wage bill",
    "CODE_ECON2": "66", "ECON2": "Wages", "CODE_ECON3": "661", "ECON3": "Salaries in cash",
    "CODE_ECON4": "661112", "ECON4": "Base salary",
    "CODE_ADMIN5": "1100", "ADMIN5": HP, "is_foreign": False,
    "CODE_GENRE": "1", "GENRE": "Gender 1",
    "CODE_BUDGET_VERT": "0", "BUDGET_VERT": "Green 0",
    "CODE_DEPENSES_SOCIALES": "1", "DEPENSES_SOCIALES": "Social 1",
    "year": 2025,
}
CAPITAL = {"CODE_ECON1": "5", "ECON1": "DEPENSES EN CAPITAL", "econ": "Capital expenditures",
           "CODE_ECON2": "23", "ECON2": "Buildings", "CODE_ECON3": "231", "ECON3": "Office buildings",
           "CODE_ECON4": "231112", "ECON4": "Office building works"}


def row(measure_base, **overrides):
    values = dict(BASE, **overrides)
    for i, name in enumerate(SOURCE_MEASURES):
        values[name] = measure_base + i
    # the imputation is action, admin4, func3, admin5 and econ4, as in the source system
    values["CODE_IMPUTATION"] = "".join(
        values[c] or "" for c in ("CODE_ACTION", "CODE_ADMIN4", "CODE_FUNC3", "CODE_ADMIN5", "CODE_ECON4"))
    return values


# Rows 0 and 1 share every dimension and must merge; row 2 differs from them only
# by a NULL gender and row 3 only by its function, so both stay separate groups in
# the fact table. Row 11 differs from row 0 only by its econ4 line, so it merges into
# the fact table's group and stays a line of its own in the detail table.
SILVER_ROWS = [
    row(100),
    row(1000),
    row(10_000, CODE_GENRE=None, GENRE=None),
    row(20_000, func_sub="Judiciary", CODE_FUNC3="0330"),
    row(30_000, CODE_FUNC1="09", func="Education", func_sub="Primary Education", year=2024),
    row(40_000, **CAPITAL),
    row(50_000, **CAPITAL, ADMIN5=PROJ, CODE_ADMIN4="0000000000002", ADMIN4="Project B"),
    row(60_000, **CAPITAL, ADMIN5=FC),
    row(70_000, **CAPITAL, ADMIN5="FINANCEMENTS INTERNES"),
    row(80_000, **CAPITAL, ADMIN5="BANQUE MONDIALE/DONS", is_foreign=True),
    row(90_000, ADMIN5="BANQUE MONDIALE/EMPRUNTS", is_foreign=True),
    row(100_000, CODE_ECON4="661114", ECON4="Allowances"),
]


def code_dtypes(columns, suffix):
    return {c: str for c in columns if c.endswith(suffix)}


class _ExecutionRun(unittest.TestCase):
    """Runs the script once on SILVER_ROWS and reads both outputs."""

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
        detail = os.path.join(TMP, "tgo_execution_detail.csv")
        detail_dtypes = code_dtypes(pd.read_csv(detail, nrows=0).columns, "_code")
        cls.detail = pd.read_csv(detail, dtype={**detail_dtypes, "imputation": str})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    @staticmethod
    def _select(frame, filters):
        mask = pd.Series(True, index=frame.index)
        for column, value in filters.items():
            mask &= frame[column].isna() if value is None else frame[column] == value
        return frame[mask]


class TogoExecutionTest(_ExecutionRun):
    def lines(self, **filters):
        return self._select(self.out, filters)

    def test_columns_in_order(self):
        self.assertEqual(list(self.out.columns), DIMENSIONS + MEASURES)

    def test_measure_totals_match_silver(self):
        for target, source in zip(MEASURES, SOURCE_MEASURES):
            with self.subTest(measure=target):
                self.assertEqual(self.out[target].sum(), self.silver[source].sum())
        by_year = self.out.groupby("year")["ordonnancer"].sum().to_dict()
        self.assertEqual(by_year, self.silver.groupby("year")["ORDONNANCER"].sum().to_dict())

    def test_identical_dimensions_merge_and_null_keys_stay_separate(self):
        self.assertEqual(len(self.out), len(SILVER_ROWS) - 2)
        merged = self.lines(year=2025, econ1_code="2", funding_code="RI_HORS_PROJETS",
                            gender_code="1", func_sub=None)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged["dotation_ae"].iloc[0], 100 + 1000 + 100_000)
        self.assertEqual(merged["ordonnancer"].iloc[0], 107 + 1007 + 100_007)
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


class TogoExecutionDetailTest(_ExecutionRun):
    """The line-level table: the fact table's keys plus action, admin3/4/5, func3,
    econ2 to econ4 and the imputation, with the same measures."""

    def detail_lines(self, **filters):
        return self._select(self.detail, filters)

    def test_detail_columns_in_order(self):
        self.assertEqual(list(self.detail.columns), DETAIL_DIMENSIONS + MEASURES)

    def test_detail_totals_match_silver_and_fact(self):
        for target, source in zip(MEASURES, SOURCE_MEASURES):
            with self.subTest(measure=target):
                self.assertEqual(self.detail[target].sum(), self.silver[source].sum())
                self.assertEqual(self.detail[target].sum(), self.out[target].sum())
        by_year = self.detail.groupby("year")["ordonnancer"].sum().to_dict()
        self.assertEqual(by_year, self.silver.groupby("year")["ORDONNANCER"].sum().to_dict())

    def test_detail_keeps_each_line(self):
        # rows 0 and 1 share their imputation and every key; each other row is a line
        self.assertEqual(len(self.detail), len(SILVER_ROWS) - 1)
        base = self.detail_lines(imputation="0031000000000000101601100661112", year=2025,
                                 funding_code="RI_HORS_PROJETS", gender_code="1")
        self.assertEqual(base["dotation_ae"].tolist(), [100 + 1000])
        allowances = self.detail_lines(econ4_code="661114")
        self.assertEqual(allowances["dotation_ae"].tolist(), [100_000])
        self.assertEqual(allowances["imputation"].tolist(), ["0031000000000000101601100661114"])

    def test_detail_line_keys(self):
        line = self.detail_lines(econ4_code="661114").iloc[0]
        self.assertEqual(
            line[["action_code", "action", "admin3_code", "admin3", "admin4_code", "admin4",
                  "admin5_code", "func3_code", "econ2_code", "econ2", "econ3_code", "econ3",
                  "econ4", "admin2_short", "budget_type"]].tolist(),
            ["0031", "Action 31", "11", "Cabinet structures", "0000000000001", "Project A",
             "1100", "0160", "66", "Wages", "661", "Salaries in cash",
             "Allowances", "MA", "BUDGET GENERAL"])

    def test_detail_funding_matches_fact(self):
        capital = self.detail_lines(econ1_code="5").set_index("funding_code")
        self.assertEqual(capital["dotation_ae"].to_dict(), {
            "RI_HORS_PROJETS": 40_000, "RI_PROJETS": 50_000, "RI_CONTREPARTIE": 60_000,
            "RI_AUTRES": 70_000, "RE": 80_000})
        self.assertEqual(capital.loc["RE", "funder"], "BANQUE MONDIALE/DONS")
        self.assertTrue(capital.drop(index="RE")["funder"].isna().all())
        self.assertEqual(capital.loc["RI_PROJETS", "admin4"], "Project B")

    def test_detail_null_keys_stay_separate(self):
        null_gender = self.detail_lines(gender_code=None)
        self.assertEqual(null_gender["dotation_ae"].tolist(), [10_000])


if __name__ == "__main__":
    unittest.main()
