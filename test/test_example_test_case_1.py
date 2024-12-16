import unittest
import pandas as pd
from t_ordering import Criterion, Preference, DecisionModel

class TestExampleTestCase1(unittest.TestCase):
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
            valid_values=["low", "medium", "high"],
        )

        self.brand_reputation_criterion = Criterion(
            name="Brand Reputation",
            absolute=False,
            maximize=True,
            valid_values=["unknown", "known", "famous"],
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
        self.preferences_list = [
            Preference(criterion1=self.price_criterion, criterion2=self.quality_criterion, equivalent=False),
            Preference(criterion1=self.quality_criterion, criterion2=self.brand_reputation_criterion, equivalent=True),
            Preference(criterion1=self.quality_criterion, criterion2=self.criterion_1, equivalent=True),
            Preference(criterion1=self.criterion_1, criterion2=self.criterion_2, equivalent=True),
            Preference(criterion1=self.criterion_2, criterion2=self.quality_criterion, equivalent=True),
            Preference(criterion1=self.criterion_2, criterion2=self.criterion_3, equivalent=False),
        ]

    def test_decision_model_initialization(self):
        # Проверка, что исключение не возникает
        try:
            decision_model = DecisionModel(self.criteria_list, self.alternatives_df, self.preferences_list)
        except ValueError as e:
            self.fail(f"Инициализация DecisionModel выбросила ValueError: {e}")

    def test_t_ordering_output(self):
        # Проверка, что метод t_ordering возвращает ожидаемый DataFrame
        decision_model = DecisionModel(self.criteria_list, self.alternatives_df, self.preferences_list)
        expected_df = pd.DataFrame(
            {
                "Price": [0.555556, 0.222222, 0.777778],
                "Quality": [0.5, 1.0, 0.0],
                "Brand Reputation": [0.5, 1.0, 0.0],
                "criterion1": [0.01, 0.02, 0.03],
                "criterion2": [0.15, 0.21, 0.34],
                "criterion3": [0.0012, 0.0026, 0.0037],
            },
            index=["Alternative A", "Alternative B", "Alternative C"],
        )
        expected_df.index.name = "Alternative"

        # Вызов метода t_ordering
        result_df = decision_model.t_ordering()
        print(result_df)
        # Сравнение DataFrame
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == "__main__":
    unittest.main(verbosity=2)
