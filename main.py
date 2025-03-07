import pandas as pd
from machine_learning import MachineLearning
from data_analysis import FraudAnalysis


class Main:
    def __init__(self):
        self.data = pd.read_csv("creditcard.csv")


    def process(self):
        fa = FraudAnalysis(data=self.data)
        print(f"Number of rows of the dataset: {fa.get_total_rows()}")
        print(f"Number of frauds present in the dataset: {fa.get_total_fraud('Class')}")
        print(f"Number of non-frauds present in the dataset: {fa.get_total_non_fraud('Class')}")
        print(f"Percentage of frauds present in the dataset: {fa.get_percentage_fraud('Class')}%")
        print(f"Percentage of non-frauds present in the dataset: {fa.get_percentage_non_fraud('Class')}%")

        # Inicia a instância da classe Machine Learning
        model = MachineLearning()

        # Inicia o validador com apenas 1 divisão, 10% de teste e a seed randômica = 0
        validator = model.get_validator(n_splits = 1, test_size = 0.1, random_state = 0)

        # Exclui a coluna 'target' da base original
        feature = self.data.drop('Class', axis=1).values

        # Separa a coluna Y em uma variável
        target = self.data['Class'].values

        # Divide os resultados do split em 4 variáveis, 2 de treino e 2 de teste
        feature_train, feature_test, target_train, target_test = model.get_split(validator, feature, target)

        # Inicializa a árvore de decisão
        decision_tree_classifier = model.get_decision_tree_classifier(max_depth=10, random_state=0, min_samples_leaf=10)

        # Treina o modelo com os dados que recuperamos do split
        decision_tree = model.fit_decision_tree(decision_tree=decision_tree_classifier, feature_train=feature_train,
                                                target_train=target_train)

        # Faz a predição para testar o modelo
        target_prediction = model.tree_predict(decision_tree=decision_tree, feature_test=feature_test)

        # Verifica acurácia do modelo
        print(model.get_tree_score(target_test=target_test, target_predict=target_prediction))
        print(model.get_confusion_matrix(target_test=target_test, target_predict=target_prediction))
        print(model.get_precision_score(target_test=target_test, target_predict=target_prediction))
        print(model.get_recall_score(target_test=target_test, target_predict=target_prediction))

        random_tree_classifier = model.get_random_forest_classifier(n_estimators=50, random_state = 0, max_depth=10)
        random_forest = model.fit_decision_tree(decision_tree=random_tree_classifier, feature_train=feature_train,
                                                target_train=target_train)
        random_forest_prediction = model.tree_predict(decision_tree=random_forest, feature_test=feature_test)

        print(model.get_precision_score(target_test=target_test, target_predict=random_forest_prediction))
        print(model.get_recall_score(target_test=target_test, target_predict=random_forest_prediction))

        ada_boost_classifier = model.get_ada_boost_classifier(n_estimators=200, random_state = 0)
        ada_boost = model.fit_decision_tree(decision_tree=ada_boost_classifier, feature_train=feature_train,
                                            target_train=target_train)
        ada_boost_prediction = model.tree_predict(decision_tree=ada_boost, feature_test=feature_test)

        print(model.get_precision_score(target_test=target_test, target_predict=ada_boost_prediction))
        print(model.get_recall_score(target_test=target_test, target_predict=ada_boost_prediction))

        # Salva um arquivo .png com a árvore de decisão
        model.save_tree_plot(decision_tree=decision_tree_classifier, name="arquivo1.png", filled=True, fontsize=14)




if __name__ == "__main__":
    main = Main()
    main.process()