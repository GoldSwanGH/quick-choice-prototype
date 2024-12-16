import unittest
import pandas as pd
from t_ordering import Criterion, Preference, DecisionModel

class TestCycleDetection(unittest.TestCase):
    def setUp(self):
        # Определение критериев
        self.price_criterion = Criterion(
            name="Price",
            absolute=True,
            maximize=False,
            min_value=100,
            max_value=1000,
        )

        self.quality_criterion = Criterion(
            name="Quality",
            absolute=False,
            maximize=True,
            valid_values=["low", "medium", "high"],  # Упорядоченные значения
        )

        self.brand_reputation_criterion = Criterion(
            name="Brand Reputation",
            absolute=False,
            maximize=True,
            valid_values=["unknown", "known", "famous"],  # Упорядоченные значения
        )

        self.criterion_1 = Criterion(
            name="criterion1",
            absolute=True,
            maximize=True,
            min_value=0,
            max_value=10000,
        )

        self.criterion_2 = Criterion(
            name="criterion2",
            absolute=True,
            maximize=True,
            min_value=0,
            max_value=1000,
        )

        self.criterion_3 = Criterion(
            name="criterion3",
            absolute=True,
            maximize=True,
            min_value=0,
            max_value=100000,
        )

        self.criteria_list = [
            self.price_criterion,
            self.quality_criterion,
            self.brand_reputation_criterion,
            self.criterion_1,
            self.criterion_2,
            self.criterion_3,
        ]

        # Создание DataFrame с альтернативами
        data = {
            "Alternative": ["Alternative A", "Alternative B", "Alternative C"],
            "Price": [500, 800, 300],
            "Quality": ["medium", "high", "low"],
            "Brand Reputation": ["known", "famous", "unknown"],
            "criterion1": [100, 200, 300],
            "criterion2": [150, 210, 340],
            "criterion3": [120, 260, 370],
        }
        self.alternatives_df = pd.DataFrame(data)
        self.alternatives_df.set_index("Alternative", inplace=True)

        # Обеспечение корректных типов данных
        self.alternatives_df["Price"] = self.alternatives_df["Price"].astype(float)
        self.alternatives_df["Quality"] = self.alternatives_df["Quality"].astype(str)
        self.alternatives_df["Brand Reputation"] = self.alternatives_df["Brand Reputation"].astype(str)
        self.alternatives_df["criterion1"] = self.alternatives_df["criterion1"].astype(float)
        self.alternatives_df["criterion2"] = self.alternatives_df["criterion2"].astype(float)
        self.alternatives_df["criterion3"] = self.alternatives_df["criterion3"].astype(float)

        # Определение предпочтений
        self.preference1 = Preference(
            criterion1=self.quality_criterion,
            criterion2=self.price_criterion,
            equivalent=False
        )

        self.preference2 = Preference(
            criterion1=self.price_criterion,
            criterion2=self.brand_reputation_criterion,
            equivalent=True
        )

        self.preference3 = Preference(
            criterion1=self.brand_reputation_criterion,
            criterion2=self.quality_criterion,
            equivalent=False
        )

        self.preferences_list = [self.preference1, self.preference2, self.preference3]

    def test_decision_model_raises_value_error(self):
        # Проверка, что при создании DecisionModel выбрасывается ValueError
        with self.assertRaises(ValueError):
            DecisionModel(self.criteria_list, self.alternatives_df, self.preferences_list)

if __name__ == "__main__":
    unittest.main(verbosity=2)