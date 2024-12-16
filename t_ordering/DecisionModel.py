import pandas as pd
from typing import List, Dict, Tuple
from t_ordering import Criterion, Preference

class DecisionModel:
    def __init__(self, criteria_list: List[Criterion], alternatives_df: pd.DataFrame, preferences_list: List[Preference]):
        """
        Инициализирует объект DecisionModel.

        Параметры:
        - criteria_list: Список объектов Criterion.
        - alternatives_df: DataFrame с альтернативами и значениями критериев.
        - preferences_list: Список объектов Preference.
        """
        self.pareto_front = None
        self.criteria = {criterion.name: criterion for criterion in criteria_list}
        self.alternatives = alternatives_df.copy()
        self.preferences = preferences_list
        self.normalized_alternatives = None  # DataFrame с нормализованными значениями
        self.validate_model()
        self.normalize_data()

    def validate_model(self):
        """
        Выполняет валидацию модели: проверяет корректность данных и отсутствие циклов в предпочтениях.
        """
        # Проверка, что все критерии присутствуют в DataFrame альтернатив
        for criterion in self.criteria.values():
            if criterion.name not in self.alternatives.columns:
                raise ValueError(
                    f"Критерий '{criterion.name}' отсутствует в DataFrame альтернатив"
                )
            # Проверка типа данных столбца
            if criterion.is_absolute():
                if not pd.api.types.is_numeric_dtype(self.alternatives[criterion.name]):
                    raise ValueError(
                        f"Критерий '{criterion.name}' должен иметь числовой тип данных"
                    )
                # Проверка допустимых значений
                if not self.alternatives[criterion.name].between(
                    criterion.min_value, criterion.max_value
                ).all():
                    invalid_values = self.alternatives[
                        ~self.alternatives[criterion.name].between(
                            criterion.min_value, criterion.max_value
                        )
                    ][criterion.name]
                    raise ValueError(
                        f"Значения {invalid_values.tolist()} для критерия '{criterion.name}' выходят за допустимый диапазон [{criterion.min_value}, {criterion.max_value}]"
                    )
            elif criterion.is_ordinal():
                if not pd.api.types.is_object_dtype(self.alternatives[criterion.name]):
                    raise ValueError(
                        f"Критерий '{criterion.name}' должен иметь строковый тип данных для порядковых значений"
                    )
                # Проверка допустимых значений
                if not self.alternatives[criterion.name].isin(criterion.valid_values).all():
                    invalid_values = self.alternatives[
                        ~self.alternatives[criterion.name].isin(criterion.valid_values)
                    ][criterion.name]
                    raise ValueError(
                        f"Значения {invalid_values.tolist()} для критерия '{criterion.name}' не входят в допустимые значения {criterion.valid_values}"
                    )
        # Проверка, что все критерии из предпочтений присутствуют в списке критериев
        criterion_names = set(self.criteria.keys())
        for pref in self.preferences:
            if pref.criterion1.name not in criterion_names:
                raise ValueError(f"Критерий '{pref.criterion1.name}' из предпочтений отсутствует в списке критериев")
            if pref.criterion2.name not in criterion_names:
                raise ValueError(f"Критерий '{pref.criterion2.name}' из предпочтений отсутствует в списке критериев")
        # Проверка на циклы в предпочтениях
        self.check_for_cycles()

    def check_for_cycles(self):
        """
        Проверяет наличие циклов в предпочтениях критериев.
        Если цикл найден, выбрасывает исключение ValueError с подробной информацией о цикле.
        """
        # Построение графа предпочтений
        graph: Dict[str, List[Tuple[str, bool]]] = {}
        for criterion in self.criteria.values():
            graph[criterion.name] = []

        # Добавляем ребра в граф
        for pref in self.preferences:
            c1 = pref.criterion1.name
            c2 = pref.criterion2.name
            if pref.equivalent:
                # Добавляем двунаправленные ребра для эквивалентности
                graph[c1].append((c2, False))  # False означает, что ребро не строгое
                graph[c2].append((c1, False))
            else:
                # Добавляем направленное ребро для строгого предпочтения
                graph[c1].append((c2, True))  # True означает, что ребро строгое

        def dfs(node, parent, has_strict_edge, stack):
            stack.append(node)
            for neighbor, is_strict in graph[node]:
                path_has_strict_edge = has_strict_edge or is_strict
                if (neighbor != stack[0]) and (len(stack) < 3):
                    if dfs(neighbor, node, path_has_strict_edge, stack):
                        return True
                elif neighbor == stack[0] and path_has_strict_edge:
                    # Если найден цикл, и по пути есть хотя бы одно строгое предпочтение, то собираем цикл
                    cycle = []
                    idx = stack.index(neighbor)
                    cycle_nodes = stack[idx:] + [neighbor]
                    for i in range(len(cycle_nodes) - 1):
                        n1 = cycle_nodes[i]
                        n2 = cycle_nodes[i + 1]
                        # Найдем отношение между n1 и n2
                        for neighbor_name, is_strict_edge in graph[n1]:
                            if neighbor_name == n2:
                                relation = ">" if is_strict_edge else "="
                                cycle.append(f"{n1} {relation} {n2}")
                                break
                    error_message = "Обнаружен цикл в предпочтениях: " + " -> ".join(cycle)
                    raise ValueError(error_message)
            stack.pop()
            return False

        for node in graph:
            if dfs(node, None, False, []):
                return

    def normalize_data(self):
        """
        Нормализует исходные данные альтернатив по каждому критерию.
        """
        normalized_df = self.alternatives.copy()
        for criterion in self.criteria.values():
            if criterion.is_ordinal():
                if len(criterion.valid_values) == 1:
                    normalized_df[criterion.name] = 1.0
                    continue
                # Кодирование порядковых значений от 0 до n
                value_to_number = {value: idx for idx, value in enumerate(criterion.valid_values)}
                normalized_values = normalized_df[criterion.name].map(value_to_number)
                # Сохраняем минимальное и максимальное значение после кодирования
                K_min = 0
                K_max = len(criterion.valid_values) - 1
                Alt_star = normalized_values.astype(float)
            else:
                # Абсолютный критерий
                if criterion.min_value == criterion.max_value:
                    normalized_df[criterion.name] = 1.0
                    continue
                Alt_star = normalized_df[criterion.name].astype(float)
                K_min = criterion.min_value
                K_max = criterion.max_value

            # Применяем нормализацию
            if criterion.is_maximize():
                normalized_values = (Alt_star - K_min) / (K_max - K_min)
            else:
                normalized_values = (K_max - Alt_star) / (K_max - K_min)

            normalized_df[criterion.name] = normalized_values

        self.normalized_alternatives = normalized_df
        return normalized_df

    def find_pareto_front(self):
        """
        Находит множество Парето среди нормализованных альтернатив.

        Результат:
        - Обновляет self.pareto_front с альтернативами из множества Парето.
        """
        if self.normalized_alternatives is None:
            raise ValueError("Данные не нормализованы. Пожалуйста, выполните нормализацию перед поиском множества Парето.")

        pareto_front = []
        dominated = set()

        alternatives = self.normalized_alternatives
        alternatives_matrix = alternatives.values
        num_alternatives = alternatives.shape[0]
        indices = alternatives.index.tolist()

        for i in range(num_alternatives):
            if i in dominated:
                continue
            for j in range(num_alternatives):
                if i == j or j in dominated:
                    continue
                if self._dominates(alternatives_matrix[j], alternatives_matrix[i]):
                    dominated.add(i)
                    break
            else:
                pareto_front.append(indices[i])

        self.pareto_front = self.normalized_alternatives.loc[pareto_front]
        print(f"Найдено {len(self.pareto_front)} альтернатив в множестве Парето.\n")
        return self.pareto_front

    def _dominates(self, row1, row2):
        """
        Проверяет, доминирует ли row1 над row2 по критерию Парето.

        Параметры:
        - row1, row2: массивы значений критериев для двух альтернатив.

        Возвращает:
        - True, если row1 доминирует над row2, иначе False.
        """
        return all(r1 >= r2 for r1, r2 in zip(row1, row2)) and any(r1 > r2 for r1, r2 in zip(row1, row2))

    def _get_equivalent_groups(self):
        """
        Создает группы эквивалентных критериев на основе предпочтений.

        Возвращает:
        - Список наборов, каждый набор содержит имена эквивалентных критериев.
        """
        # Initialize variables
        criterion_to_group = {}
        groups = []

        # For each criterion
        for criterion_name in self.criteria.keys():
            # Skip if already in a group
            if criterion_name in criterion_to_group:
                continue

            # Find all equivalence preferences involving this criterion
            equivalent_criteria = set()
            to_process = [criterion_name]

            while to_process:
                current = to_process.pop()
                if current in equivalent_criteria:
                    continue
                equivalent_criteria.add(current)

                for pref in self.preferences:
                    if pref.equivalent:
                        if pref.criterion1.name == current:
                            if pref.criterion2.name not in equivalent_criteria:
                                to_process.append(pref.criterion2.name)
                        elif pref.criterion2.name == current:
                            if pref.criterion1.name not in equivalent_criteria:
                                to_process.append(pref.criterion1.name)

            # Assign these equivalent criteria to a group
            existing_groups = [criterion_to_group[c] for c in equivalent_criteria if c in criterion_to_group]
            if existing_groups:
                # Merge all existing groups and current equivalent_criteria
                merged_group = set().union(*existing_groups, equivalent_criteria)
                # Remove old groups
                groups = [g for g in groups if g not in existing_groups]
                groups.append(merged_group)
                # Update mapping
                for c in merged_group:
                    criterion_to_group[c] = merged_group
            else:
                # Create a new group
                new_group = equivalent_criteria
                groups.append(new_group)
                for c in new_group:
                    criterion_to_group[c] = new_group

        # Handle criteria not in any preferences
        unconnected_criteria = set(self.criteria.keys()) - set(criterion_to_group.keys())
        if unconnected_criteria:
            for criterion_name in unconnected_criteria:
                # Each criterion gets its own group
                new_group = set([criterion_name])
                groups.append(new_group)
                criterion_to_group[criterion_name] = new_group

        # Store the mapping
        self.criterion_to_group = criterion_to_group
        self.groups = groups  # Store groups for later use

        return groups

    def _assign_importance_relations(self):
        """
        Назначает отношения важности между группами критериев на основе предпочтений.

        - Создает и сохраняет граф отношений важности между группами критериев.
        - Учитывает транзитивность отношений важности.
        """
        # Initialize the graph
        group_importance_graph = {}

        # Get the groups and their IDs
        groups = self.groups
        group_ids = {id(group): group for group in groups}

        # Initialize the graph for each group
        for group in groups:
            group_id = id(group)
            group_importance_graph[group_id] = set()

        # For each preference where criterion1 is more important than criterion2
        for pref in self.preferences:
            if not pref.equivalent:
                c1_group = self.criterion_to_group[pref.criterion1.name]
                c2_group = self.criterion_to_group[pref.criterion2.name]
                c1_group_id = id(c1_group)
                c2_group_id = id(c2_group)
                if c1_group_id != c2_group_id:
                    # Group of criterion1 is more important than group of criterion2
                    group_importance_graph[c2_group_id].add(c1_group_id)  # Edge from less important to more important group

        # Compute transitive closure to include indirect importance relations
        # For each group, find all more important groups (direct and indirect)
        def dfs(group_id, visited):
            for more_important_group_id in group_importance_graph[group_id].copy():
                if more_important_group_id not in visited:
                    visited.add(more_important_group_id)
                    dfs(more_important_group_id, visited)
                    group_importance_graph[group_id].update(visited)

        for group_id in group_importance_graph:
            visited = set()
            dfs(group_id, visited)

        # Store the graph
        self.group_importance_graph = group_importance_graph
        self.group_ids = group_ids  # Store group IDs for reference

        # Print out the groups and their importance relations
        print("Группы и их отношения важности (включая транзитивные):")
        for group_id, more_important_group_ids in group_importance_graph.items():
            group = self.group_ids[group_id]
            criteria_in_group = ', '.join(group)
            more_important_groups = [', '.join(self.group_ids[mid]) for mid in more_important_group_ids]
            print(f"Группа [{criteria_in_group}] -> более важные группы: {more_important_groups if more_important_groups else 'Нет'}")
        print("\n")

    def _check_t_dominance(self, Z_values, W_values):
        """
        Проверяет, доминирует ли альтернатива Z над альтернативой W в t-упорядочении.

        Параметры:
        - Z_values: ряд с нормализованными значениями критериев для альтернативы Z.
        - W_values: ряд с нормализованными значениями критериев для альтернативы W.

        Возвращает:
        - True, если Z доминирует над W, иначе False.
        """
        # Compute group sums for Z and W
        Z_group_sums = {}
        W_group_sums = {}

        group_name_to_id = {}  # Map group names to IDs for easy access

        for group in self.groups:
            group_id = id(group)
            group_name = ','.join(sorted(group))  # Create a unique name for the group
            group_name_to_id[group_name] = group_id
            Z_sum = round(Z_values[list(group)].sum(), 8)
            W_sum = round(W_values[list(group)].sum(), 8)

            Z_group_sums[group_name] = Z_sum
            W_group_sums[group_name] = W_sum

        # Check dominance using group sums for WE
        if self._dominates_group_sums(Z_group_sums, W_group_sums):
            #print(f'[DEBUG] Dominated by WE, Z = {Z_group_sums}, W = {W_group_sums}')
            return True

        # Initialize W_adjusted_group_sums with original W_group_sums
        W_adjusted_group_sums = W_group_sums.copy()

        # Start transferring from groups with the most more important groups to those with none
        # This effectively starts from the least important groups
        groups_sorted = sorted(self.group_importance_graph.items(), key=lambda x: len(x[1]), reverse=True)
        group_id_to_name = {id(group): ','.join(sorted(group)) for group in self.groups}

        transferred = False

        # Start transferring from less important groups to more important ones
        for group_id, more_important_group_ids in groups_sorted:
            current_group_name = group_id_to_name[group_id]
            Z_current = Z_group_sums[current_group_name]
            W_current = W_adjusted_group_sums[current_group_name]

            # Scenario 1: W_current <= Z_current, no transfer needed
            if W_current <= Z_current:
                continue  # Skip this group

            # Scenario 2: W_current > Z_current, need to transfer excess to more important groups
            excess = round(W_current - Z_current, 8)  # Amount to transfer
            W_adjusted_group_sums[current_group_name] = Z_current  # Reduce W_current to Z_current

            # Try to transfer excess to more important groups
            remaining_excess = excess

            # Try to transfer excess to more important groups
            if not more_important_group_ids:
                # No more important groups to transfer to
                return False  # Cannot adjust W to be dominated by Z

            # Iterate over more important groups
            for more_important_group_id in more_important_group_ids:
                more_important_group_name = group_id_to_name[more_important_group_id]
                Z_more = Z_group_sums[more_important_group_name]
                W_more = W_adjusted_group_sums[more_important_group_name]

                # Calculate available capacity in the more important group
                capacity = round(Z_more - W_more, 8)
                if capacity <= 0:
                    continue  # No capacity, move to the next more important group

                # Transfer as much as possible
                transfer_amount = min(remaining_excess, capacity)
                W_adjusted_group_sums[more_important_group_name] = round(W_adjusted_group_sums[more_important_group_name] + transfer_amount, 8)
                remaining_excess = round(remaining_excess - transfer_amount, 8)

                if remaining_excess <= 0:
                    transferred = True
                    break  # Transferred all excess, no need to continue

            if remaining_excess > 0:
                # Unable to transfer all excess to more important groups
                # Therefore, cannot make W equivalent or dominated by Z
                return False

        #print(f'[DEBUG] After transfers, Z = {Z_group_sums}, W = {W_adjusted_group_sums}')
        #print(f'[DEBUG] transferred = {transferred}')
        # After transfers, check if Z dominates or is equivalent to adjusted W
        if transferred and self._dominates_or_equal_group_sums(Z_group_sums, W_adjusted_group_sums):
            #print(f'[DEBUG] Discarded: alternative Z {Z_values}, alternative W {W_values}')
            return True

        # If no dominance found
        #print(f'[DEBUG] Not discarded: alternative Z {Z_values}, alternative W {W_values}')
        return False

    def _dominates_group_sums(self, Z_sums, W_sums):
        """
        Проверяет, доминирует ли Z_sums над W_sums в смысле Парето.

        Параметры:
        - Z_sums: словарь групповых сумм для альтернативы Z.
        - W_sums: словарь групповых сумм для альтернативы W.

        Возвращает:
        - True, если Z_sums доминирует над W_sums, иначе False.
        """
        dominates = False
        for group_name in Z_sums.keys():
            if round(Z_sums[group_name], 8) < round(W_sums[group_name], 8):
                return False  # Z is worse in at least one group
            elif round(Z_sums[group_name], 8) > round(W_sums[group_name], 8):
                dominates = True  # Z is better in at least one group
        return dominates

    def _dominates_or_equal_group_sums(self, Z_sums, W_sums):
        """
        Проверяет, что Z_sums эквивалентен W_sums или Z_sums доминирует над W_sums в смысле Парето.

        Параметры:
        - Z_sums: словарь групповых сумм для альтернативы Z.
        - W_sums: словарь групповых сумм для альтернативы W.

        Возвращает:
        - True, если Z_sums доминирует или эквивалентен W_sums, в противном случае False.
        """
        for group_name in Z_sums.keys():
            if round(Z_sums[group_name], 8) < round(W_sums[group_name], 8):
                return False  # Z is worse in at least one group
        return True

    def t_ordering(self):
        """
        Применяет метод t-упорядочения для сокращения множества Парето на основе предпочтений пользователя.

        Результат:
        - Обновляет self.pareto_t альтернативами, оставшимися после t-упорядочения.
        """
        if self.pareto_front is None:
            self.find_pareto_front()

        # Assign importance relations
        self._get_equivalent_groups()
        self._assign_importance_relations()

        # Copy Pareto alternatives for processing
        pareto_alternatives = self.pareto_front.copy()

        # Set of alternatives to remove
        alternatives_to_remove = set()

        # List of alternative names
        alternative_names = list(pareto_alternatives.index)

        # For each pair of alternatives in the Pareto set
        for i in range(len(alternative_names)):
            alt_name_Z = alternative_names[i]
            if alt_name_Z in alternatives_to_remove:
                continue
            Z_values = pareto_alternatives.loc[alt_name_Z]
            #print(f'[DEBUG] comparing {alt_name_Z}')
            for j in range(len(alternative_names)):
                if i == j:
                    continue
                alt_name_W = alternative_names[j]
                if alt_name_W in alternatives_to_remove:
                    continue
                W_values = pareto_alternatives.loc[alt_name_W]
                #print(f'[DEBUG] to {alt_name_W}')
                # Check if Z dominates W
                if self._check_t_dominance(Z_values, W_values):
                    #print(f'[DEBUG] {alt_name_Z} dominated {alt_name_W}')
                    alternatives_to_remove.add(alt_name_W)

        # Update alternatives after t-ordering
        self.pareto_t = pareto_alternatives.drop(index=alternatives_to_remove)
        print(f"Количество альтернатив после t-упорядочивания: {len(self.pareto_t)}\n")
        return self.pareto_t

    def __str__(self):
        """
        Возвращает строковое представление объекта DecisionModel.
        """
        criteria_str = "\n".join([str(criterion) for criterion in self.criteria.values()])
        preferences_str = "\n".join([str(pref) for pref in self.preferences])
        normalized_str = self.normalized_alternatives.to_string() if self.normalized_alternatives is not None else "Данные не нормализованы"
        return (f"DecisionModel:\n\nКритерии:\n{criteria_str}\n\n"
                f"Альтернативы:\n{self.alternatives}\n\n"
                f"Нормализованные альтернативы:\n{normalized_str}\n\n"
                f"Предпочтения:\n{preferences_str}\n")