import pandas as pd
import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_info_gain=1e-4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain
        self.tree = None
        self.feature_names = None  # To store feature names for better readability

    def _calculate_entropy(self, y):
        """计算熵"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _calculate_information_gain(self, X_column, y, threshold=None, is_continuous=True):
        """计算信息增益"""
        parent_entropy = self._calculate_entropy(y)

        if is_continuous:
            if threshold is None:  # Should not happen if called correctly
                return 0
            left_indices = X_column <= threshold
            right_indices = X_column > threshold
            y_left, y_right = y[left_indices], y[right_indices]

            if len(y_left) == 0 or len(y_right) == 0:
                return 0  # No split or one branch is empty

            p_left = len(y_left) / len(y)
            p_right = len(y_right) / len(y)
            child_entropy = p_left * self._calculate_entropy(y_left) + \
                p_right * self._calculate_entropy(y_right)
        else:  # Discrete feature
            unique_values = np.unique(X_column)
            child_entropy = 0
            for value in unique_values:
                subset_indices = X_column == value
                y_subset = y[subset_indices]
                if len(y_subset) == 0:
                    continue
                p_subset = len(y_subset) / len(y)
                child_entropy += p_subset * self._calculate_entropy(y_subset)

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _find_best_split(self, X, y, features_indices):
        """寻找最佳划分特征和阈值/值"""
        best_gain = -1
        best_feature_index = -1
        best_threshold = None  # For continuous features
        best_is_continuous = None

        for feature_index in features_indices:
            X_column = X[:, feature_index]
            # 尝试判断特征是连续还是离散 (这是一个简化判断)
            # 如果唯一值数量大于某个比例（比如10个，或者总样本的1/5），认为是连续的，或者使用先验知识判断是否是连续值
            is_continuous_feature = False
            if self.feature_names:  # If feature names are available
                feature_name = self.feature_names[feature_index]
                # Explicitly define continuous features
                if feature_name in ['Age', 'Fare']:
                    is_continuous_feature = True
                # Pclass, Sex, SibSp, Parch, Embarked are treated as discrete
            else:  # Fallback if feature names not set (less ideal)
                if len(np.unique(X_column)) > 10 and (X_column.dtype == np.float64 or X_column.dtype == np.int64):
                    is_continuous_feature = True

            if is_continuous_feature:
                unique_values = np.unique(X_column)
                if len(unique_values) < 2:  # Not enough values to split
                    continue
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                for t in thresholds:
                    gain = self._calculate_information_gain(
                        X_column, y, threshold=t, is_continuous=True)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature_index = feature_index
                        best_threshold = t
                        best_is_continuous = True
            else:  # Discrete feature
                # For discrete features, we split on each unique value.
                # The information gain calculation already handles this by summing weighted child entropies.
                gain = self._calculate_information_gain(
                    X_column, y, is_continuous=False)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = None  # No threshold for discrete split, split is on unique values
                    best_is_continuous = False

        return best_feature_index, best_threshold, best_gain, best_is_continuous

    def _build_tree(self, X, y, current_depth, features_indices):
        """递归构建决策树"""
        n_samples, n_features = X.shape

        # 停止条件
        if len(np.unique(y)) == 1:  # 所有样本属于同一类别
            return {'label': y[0]}
        if current_depth >= self.max_depth:
            return {'label': Counter(y).most_common(1)[0][0]}  # 返回多数类
        if n_samples < self.min_samples_split:
            return {'label': Counter(y).most_common(1)[0][0]}
        if not features_indices:  # No more features to split on
            return {'label': Counter(y).most_common(1)[0][0]}

        best_feature_idx, best_threshold, best_gain, is_continuous = self._find_best_split(
            X, y, features_indices)

        if best_gain < self.min_info_gain or best_feature_idx == -1:
            return {'label': Counter(y).most_common(1)[0][0]}

        # 更新可用特征 (对于离散特征，一旦使用就不再使用；连续特征可以多次使用，但通常在不同节点)
        feature_name_to_display = self.feature_names[best_feature_idx] if self.feature_names else best_feature_idx

        if is_continuous:
            left_indices = X[:, best_feature_idx] <= best_threshold
            right_indices = X[:, best_feature_idx] > best_threshold

            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[right_indices], y[right_indices]

            if len(y_left) == 0 or len(y_right) == 0:  # Avoid creating nodes with no samples
                return {'label': Counter(y).most_common(1)[0][0]}

            # Recursively build left and right children
            # We pass all feature_indices down, the splitting logic will handle if they are useful
            left_child = self._build_tree(
                X_left, y_left, current_depth + 1, features_indices)
            right_child = self._build_tree(
                X_right, y_right, current_depth + 1, features_indices)

            return {
                'feature_index': best_feature_idx,
                'feature_name': feature_name_to_display,
                'threshold': best_threshold,
                'is_continuous': True,
                'left_child': left_child,
                'right_child': right_child,
                'info_gain': best_gain
            }
        else:  # Discrete feature
            # For discrete features, we create a branch for each unique value of the best feature
            unique_values = np.unique(X[:, best_feature_idx])
            branches = {}

            # Create a new list of feature indices for children, removing the current discrete feature
            # This is a common approach for ID3 with discrete features.
            remaining_features_indices = [
                fi for fi in features_indices if fi != best_feature_idx]
            # if no more features but can still split
            if not remaining_features_indices and len(unique_values) > 1:
                # if we must split but no more features, this node becomes a leaf based on majority
                pass  # The check for best_gain < min_info_gain should handle this mostly

            for value in unique_values:
                subset_indices = X[:, best_feature_idx] == value
                X_subset, y_subset = X[subset_indices], y[subset_indices]
                if len(y_subset) == 0:
                    # If a value has no samples, assign majority class of parent
                    branches[value] = {
                        'label': Counter(y).most_common(1)[0][0]}
                    continue

                # For discrete features, we typically remove the feature from consideration for children
                # in this branch of the tree.
                child_node = self._build_tree(
                    X_subset, y_subset, current_depth + 1, remaining_features_indices)
                branches[value] = child_node

            return {
                'feature_index': best_feature_idx,
                'feature_name': feature_name_to_display,
                'is_continuous': False,
                'branches': branches,
                'info_gain': best_gain
            }

    def fit(self, X, y, feature_names=None):
        """训练决策树"""
        if feature_names is not None and len(feature_names) == X.shape[1]:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]

        n_features = X.shape[1]
        initial_features_indices = list(range(n_features))
        self.tree = self._build_tree(
            X, y, current_depth=0, features_indices=initial_features_indices)

    def _predict_sample(self, x, node):
        """递归预测单个样本"""
        if 'label' in node:  # Leaf node
            return node['label']

        feature_index = node['feature_index']

        if node['is_continuous']:
            threshold = node['threshold']
            if x[feature_index] <= threshold:
                return self._predict_sample(x, node['left_child'])
            else:
                return self._predict_sample(x, node['right_child'])
        else:  # Discrete feature
            value = x[feature_index]
            if value in node['branches']:
                return self._predict_sample(x, node['branches'][value])
            else:
                labels_in_branches = [
                    b['label'] for b in node['branches'].values() if 'label' in b]
                if labels_in_branches:
                    return Counter(labels_in_branches).most_common(1)[0][0]
                else:
                    return 0  # Default prediction if label is truly unknown

    def predict(self, X):
        """预测多个样本"""
        if self.tree is None:
            raise ValueError(
                "Tree has not been trained yet. Call fit() first.")
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

# --- Data Preprocessing ---


def preprocess_data(df, is_train=True, train_mean_age=None, train_mode_embarked=None):
    # Drop irrelevant columns
    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'],
                 axis=1, errors='ignore')

    # Handle 'Age' missing values
    if is_train:
        train_mean_age = df['Age'].mean()
    df['Age'] = df['Age'].fillna(train_mean_age)

    # Handle 'Embarked' missing values
    if is_train:
        train_mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(train_mode_embarked)

    # Handle 'Fare' missing values (if any, typically in test set)
    if 'Fare' in df.columns:
        # Use mean of current set, or pass train_mean_fare
        mean_fare = df['Fare'].mean()
        df['Fare'] = df['Fare'].fillna(mean_fare)

    # Convert categorical to numerical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    embarked_map = {label: idx for idx,
                    label in enumerate(df['Embarked'].unique())}
    df['Embarked'] = df['Embarked'].map(embarked_map).astype(int)

    if is_train:
        y = df['Survived'].values
        X = df.drop('Survived', axis=1)
        feature_names = X.columns.tolist()
        return X.values, y, feature_names, train_mean_age, train_mode_embarked, embarked_map
    else:
        X = df
        feature_names = X.columns.tolist()
        return X.values, feature_names


if __name__ == '__main__':
    # Load data
    train_df = pd.read_csv('train.csv')

    # Preprocess training data
    X_train, y_train, feature_names, mean_age, mode_embarked, train_embarked_map = preprocess_data(
        train_df.copy(), is_train=True)

    print("Feature names:", feature_names)
    # Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

    # Initialize and train the Decision Tree
    dt_classifier = DecisionTree(
        max_depth=5, min_samples_split=5, min_info_gain=0.01)
    dt_classifier.fit(X_train, y_train, feature_names=feature_names)

    print("\nDecision Tree Structure (simplified view):")
    # A simple way to visualize or inspect the tree (can be complex for large trees)

    def print_tree_structure(node, indent=""):
        if not node:
            return
        if 'label' in node:
            print(f"{indent}Leaf: Predict {node['label']}")
            return

        f_name = node.get('feature_name', node.get(
            'feature_index', 'Unknown Feature'))
        info_gain = node.get('info_gain', 0)
        print(f"{indent}Node: Split on {f_name} (Gain: {info_gain:.4f})")

        if node.get('is_continuous'):
            threshold = node['threshold']
            print(f"{indent}  Threshold: <= {threshold:.2f}")
            print(f"{indent}  Left Child:")
            print_tree_structure(node.get('left_child'), indent + "    ")
            print(f"{indent}  Right Child ( > {threshold:.2f} ):")
            print_tree_structure(node.get('right_child'), indent + "    ")
        else:  # Discrete
            branches = node.get('branches', {})
            for value, child_node in branches.items():
                print(f"{indent}  Branch for value: {value}")
                print_tree_structure(child_node, indent + "    ")

    print_tree_structure(dt_classifier.tree)  # This can be very verbose

    # --- Make predictions on training data itself for a quick check ---
    y_train_pred = dt_classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred == y_train)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")

    # --- Preprocess test data and evaluate ---
    # We need gender_submission.csv for actual test labels
    try:
        submission_df = pd.read_csv('gender_submission.csv')
        # Load fresh test data for eval
        test_df_for_eval = pd.read_csv('test.csv')

        # Create a copy for preprocessing
        processed_test_df = test_df_for_eval.copy()

        # Drop PassengerId for preprocessing, but keep it for submission later if needed
        passenger_ids_test = processed_test_df['PassengerId']
        processed_test_df = processed_test_df.drop(
            ['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, errors='ignore')

        processed_test_df['Age'] = processed_test_df['Age'].fillna(
            mean_age)  # Use train_mean_age
        if 'Fare' in processed_test_df.columns:  # Fare can be missing in test
            processed_test_df['Fare'] = processed_test_df['Fare'].fillna(
                train_df['Fare'].mean())  # Use train_fare_mean

        processed_test_df['Embarked'] = processed_test_df['Embarked'].fillna(
            mode_embarked)  # Use train_mode_embarked

        processed_test_df['Sex'] = processed_test_df['Sex'].map(
            {'male': 0, 'female': 1}).astype(int)

        # Apply the *training* embarked_map. Handle new values if any.
        current_embarked_map = {label: idx for idx, label in enumerate(
            train_df['Embarked'].fillna(mode_embarked).unique())}
        processed_test_df['Embarked'] = processed_test_df['Embarked'].map(
            current_embarked_map)
        # If test set has embarked values not in training, they will become NaN. Fill them.
        # A simple strategy: fill with a new category or the mode.
        processed_test_df['Embarked'] = processed_test_df['Embarked'].fillna(
            max(current_embarked_map.values()) + 1 if current_embarked_map else 0)

        # Ensure same feature order and set
        X_test = processed_test_df[feature_names].values
        y_test_actual = submission_df['Survived'].values

        if X_test.shape[0] == len(y_test_actual):
            y_test_pred = dt_classifier.predict(X_test)
            test_accuracy = np.mean(y_test_pred == y_test_actual)
            print(f"Test Accuracy: {test_accuracy:.4f}")
        else:
            print(
                "Test data and submission labels count mismatch. Cannot calculate test accuracy.")
            print(
                f"X_test shape: {X_test.shape}, y_test_actual length: {len(y_test_actual)}")

    except FileNotFoundError:
        print("\n'gender_submission.csv' not found. Skipping test set evaluation.")
    except Exception as e:
        print(f"\nError during test set evaluation: {e}")
        import traceback
        traceback.print_exc()
