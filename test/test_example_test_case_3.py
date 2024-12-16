import unittest
import pandas as pd
from t_ordering import Criterion, Preference, DecisionModel

class TestExampleTestCase3(unittest.TestCase):
    def setUp(self):
        # Определение критериев
        self.criteria_list = [
            Criterion(name=f"criterion{i+1}", absolute=True, maximize=True, min_value=0, max_value=1)
            for i in range(7)
        ]

        # Создание DataFrame с альтернативами
        data = {
            "Alternative": ["Alternative A", "Alternative B", "Alternative C", "Alternative D"],
            "criterion1": [0.4, 0.2, 0.2, 0.2],
            "criterion2": [0.6, 0.8, 0.7, 0.7],
            "criterion3": [0.4, 0.4, 0.5, 0.4],
            "criterion4": [0.2, 0.2, 0.3, 0.3],
            "criterion5": [0.1, 0.2, 0.2, 0.2],
            "criterion6": [0.7, 0.1, 0.5, 0.4],
            "criterion7": [0.5, 0.9, 0.7, 0.2],
        }
        self.alternatives_df = pd.DataFrame(data)
        self.alternatives_df.set_index("Alternative", inplace=True)

        # Обеспечение корректных типов данных
        for column in self.alternatives_df.columns:
            self.alternatives_df[column] = self.alternatives_df[column].astype(float)

        # Определение предпочтений
        self.preferences_list = [
            Preference(criterion1=self.criteria_list[1], criterion2=self.criteria_list[2], equivalent=True),
            Preference(criterion1=self.criteria_list[3], criterion2=self.criteria_list[4], equivalent=True),
            Preference(criterion1=self.criteria_list[4], criterion2=self.criteria_list[5], equivalent=True),
            Preference(criterion1=self.criteria_list[0], criterion2=self.criteria_list[2], equivalent=False),
            Preference(criterion1=self.criteria_list[4], criterion2=self.criteria_list[6], equivalent=False),
        ]

    def test_decision_model_initialization(self):
        # Проверка, что исключение ValueError не выбрасывается
        try:
            decision_model = DecisionModel(self.criteria_list, self.alternatives_df, self.preferences_list)
        except ValueError as e:
            self.fail(f"Инициализация DecisionModel выбросила ValueError: {e}")

    def test_t_ordering_output(self):
        # Проверка корректности вывода метода t_ordering для Примера 3
        decision_model = DecisionModel(self.criteria_list, self.alternatives_df, self.preferences_list)
        expected_df = pd.DataFrame(
            {
                "criterion1": [0.4, 0.2],
                "criterion2": [0.6, 0.7],
                "criterion3": [0.4, 0.5],
                "criterion4": [0.2, 0.3],
                "criterion5": [0.1, 0.2],
                "criterion6": [0.7, 0.5],
                "criterion7": [0.5, 0.7],
            },
            index=["Alternative A", "Alternative C"],
        )
        expected_df.index.name = "Alternative"

        # Вызов метода t_ordering
        result_df = decision_model.t_ordering()

        # Сравнение DataFrame для двух альтернатив
        pd.testing.assert_frame_equal(result_df.loc[["Alternative A", "Alternative C"]], expected_df)

if __name__ == "__main__":
    unittest.main()
