import unittest
import pandas as pd
from t_ordering import Criterion, Preference, DecisionModel

class TestExampleFromBachelor(unittest.TestCase):
    def setUp(self):
        # Определение критериев f1 и f2
        self.criterion_cr1 = Criterion(
            name="cr1",
            absolute=True,
            maximize=True,
            min_value=0,
            max_value=10,
        )
        self.criterion_cr3 = Criterion(
            name="cr3",
            absolute=True,
            maximize=False,
            min_value=1,
            max_value=1,
        )
        self.criterion_cr4 = Criterion(
            name="cr4",
            absolute=False,
            maximize=True,
            valid_values=["Two", "One", "Three"],
        )
        self.criterion_cr5 = Criterion(
            name="cr5",
            absolute=False,
            maximize=True,
            valid_values=["Yellow", "Green", "Blue"],
        )

        self.criteria_list = [self.criterion_cr1, self.criterion_cr3, self.criterion_cr4, self.criterion_cr5]

        # Создание DataFrame с альтернативами: Z и W
        data = {
            "Alternative": ["Alternative 1", "Alternative 2", "Alternative 3"],
            "cr1": [1, 2, 3],
            "cr3": [1, 1, 1],
            "cr4": ["One", "Two", "Three"],
            "cr5": ["Blue", "Yellow", "Green"],
        }
        self.alternatives_df = pd.DataFrame(data)
        self.alternatives_df.set_index("Alternative", inplace=True)

        # Обеспечение корректных типов данных
        self.alternatives_df["cr1"] = self.alternatives_df["cr1"].astype(float)
        self.alternatives_df["cr3"] = self.alternatives_df["cr3"].astype(float)
        self.alternatives_df["cr4"] = self.alternatives_df["cr4"].astype(str)
        self.alternatives_df["cr5"] = self.alternatives_df["cr5"].astype(str)

        # Предпочтений нет
        # self.preferences_list = []

        # Определение предпочтений
        self.preferences_list = [
            Preference(criterion1=self.criteria_list[0], criterion2=self.criteria_list[1], equivalent=True),
            Preference(criterion1=self.criteria_list[1], criterion2=self.criteria_list[2], equivalent=True),
            Preference(criterion1=self.criteria_list[2], criterion2=self.criteria_list[3], equivalent=True),
        ]

    def test_decision_model_initialization(self):
        # Проверка, что исключение ValueError не выбрасывается
        try:
            decision_model = DecisionModel(self.criteria_list, self.alternatives_df, self.preferences_list)
        except ValueError as e:
            self.fail(f"Инициализация DecisionModel выбросила ValueError: {e}")

    def test_t_ordering_output(self):
        # Создание модели
        decision_model = DecisionModel(self.criteria_list, self.alternatives_df, self.preferences_list)

        # Ожидаемый результат: остаётся только Alternative 3
        expected_df = pd.DataFrame(
            {
                "cr1": [0.3],
                "cr3": [1.0],
                "cr4": [1.0],
                "cr5": [0.5],
            },
            index=["Alternative 3"],
        )
        expected_df.index.name = "Alternative"

        # Вызов метода t_ordering
        result_df = decision_model.t_ordering()
        print(result_df)
        # Проверка, что остаётся только альтернатива Alternative 3
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == "__main__":
    unittest.main()
