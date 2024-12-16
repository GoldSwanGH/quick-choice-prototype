import unittest
import pandas as pd
from t_ordering import Criterion, Preference, DecisionModel

class TestExampleFromArticle(unittest.TestCase):
    def setUp(self):
        # Определение критериев f1 и f2
        self.criterion_f1 = Criterion(
            name="f1",
            absolute=True,
            maximize=True,
            min_value=0,
            max_value=1,
        )
        self.criterion_f2 = Criterion(
            name="f2",
            absolute=True,
            maximize=True,
            min_value=0,
            max_value=1,
        )
        self.criterion_f3 = Criterion(
            name="f1",
            absolute=True,
            maximize=True,
            min_value=0,
            max_value=1,
        )
        self.criterion_f4 = Criterion(
            name="f2",
            absolute=True,
            maximize=True,
            min_value=0,
            max_value=1,
        )

        self.criteria_list = [self.criterion_f1, self.criterion_f2, self.criterion_f3, self.criterion_f4]

        # Создание DataFrame с альтернативами: Z и W
        data = {
            "Alternative": ["Z", "W"],
            "f1": [1, 0.4],
            "f2": [0.5, 0.9],
            "f3": [0.1, 0.1],
            "f4": [0.2, 0.2],
        }
        self.alternatives_df = pd.DataFrame(data)
        self.alternatives_df.set_index("Alternative", inplace=True)

        # Обеспечение корректных типов данных
        for column in self.alternatives_df.columns:
            self.alternatives_df[column] = self.alternatives_df[column].astype(float)

        # Определение предпочтения: f1 важнее, чем f2
        self.preferences_list = [
            Preference(criterion1=self.criterion_f1, criterion2=self.criterion_f2, equivalent=False)
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

        # Ожидаемый результат: остаётся только Z
        expected_df = pd.DataFrame(
            {
                "f1": [1.0],
                "f2": [0.5],
                "f3": [0.1],
                "f4": [0.2],
            },
            index=["Z"],
        )
        expected_df.index.name = "Alternative"

        # Вызов метода t_ordering
        result_df = decision_model.t_ordering()
        print(result_df)
        # Проверка, что остаётся только альтернатива Z
        pd.testing.assert_frame_equal(result_df.loc[["Z"]], expected_df)

if __name__ == "__main__":
    unittest.main()
