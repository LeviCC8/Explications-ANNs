import docplex.mp.model as mp
from anchor import anchor_tabular
from lime import lime_tabular
import shap
from teste import insert_output_constraints_tjeng, get_miminal_explanation
from milp import codify_network
import tensorflow as tf
import pandas as pd
import os
import numpy as np


def get_anchor_explainer(class_names, feature_names, train_data, categorical_names):
    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=class_names,
        feature_names=feature_names,
        train_data=train_data,
        categorical_names=categorical_names)
    return explainer


def get_anchor_explanation(network_input, model, explainer, class_names, feature_names):
    predict_fn = lambda x: [tf.argmax(model.predict(x)[0]).numpy()]
    exp = explainer.explain_instance(network_input,
                                     predict_fn,
                                     #num_features=len(feature_names),
                                     #top_labels=1,
                                     #labels=list(range(len(class_names)))
                                     )
    return sorted(list(set(exp.exp_map['feature'])))


def get_lime_explainer(class_names, feature_names, train_data, categorical_names):
    explainer = lime_tabular.LimeTabularExplainer(
        class_names=class_names,
        feature_names=feature_names,
        training_data=train_data,
        categorical_names=categorical_names)
    return explainer


def get_lime_explanation(network_input, model, explainer, n_features, class_names, feature_names):
    predict_fn = lambda x: model.predict(np.expand_dims(x, 0))[0]
    exp = explainer.explain_instance(network_input[0],
                                     predict_fn,
                                     num_features=n_features,
                                     top_labels=1,
                                     #labels=list(range(len(class_names)))
                                     )
    return sorted(list(dict(exp.local_exp[exp.top_labels[0]]).keys()))


def get_shap_explainer(model, train_data):
    explainer = shap.Explainer(model, train_data)
    return explainer


def get_shap_explanation(network_input, explainer, n_features, predicted_class):
    # ACHO QUE NÃO PRECISO TRATAR DADOS CATEGÓRICOS SEM ONE HOT
    shap_values = explainer(network_input)[-1]
    sum_values = shap_values
    sorted_by_abs_sum_values = np.argsort(sum_values.abs.values, axis=0)[::-1][:, predicted_class]
    return sorted(sorted_by_abs_sum_values[:n_features])


def validate_heuristic_explanation(mdl, heuristic_explanation, network_input, network_output, n_classes, output_bounds):
    output_variables = [mdl.get_var_by_name(f'o_{i}') for i in range(n_classes)]
    mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{i}')  == network_input[i] for i in heuristic_explanation], names='input')

    binary_variables = mdl.binary_var_list(n_classes - 1, name='b')
    mdl.add_constraint(mdl.sum(binary_variables) >= 1)

    mdl = insert_output_constraints_tjeng(mdl, output_variables, network_output, binary_variables, output_bounds)
    mdl.solve(log_output=False)
    if mdl.solution is None:
        return None
    else:
        return np.array([mdl.solution.get_value(f'x_{i}') for i in range(len(network_input))])


def repair_heuristic_explanation(mdl, heuristic_explanation, network_input, network_output, n_classes, output_bounds):
    heuristic_explanation_complement = [i for i in range(len(network_input)) if i not in heuristic_explanation]
    im1_constraints = mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{i}') == network_input[i] for i in heuristic_explanation_complement],
        names='im1_')
    im2_constraints = mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{i}') == network_input[i] for i in heuristic_explanation],
        names='im2_')

    output_variables = [mdl.get_var_by_name(f'o_{i}') for i in range(n_classes)]

    binary_variables = mdl.binary_var_list(n_classes - 1, name='b')
    mdl.add_constraint(mdl.sum(binary_variables) >= 1)
    mdl = insert_output_constraints_tjeng(mdl, output_variables, network_output, binary_variables, output_bounds)

    for constraint in im1_constraints:
        mdl.remove_constraint(constraint)

        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)

    for constraint in im2_constraints:
        mdl.remove_constraint(constraint)

        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)

    return mdl.find_matching_linear_constraints('im1_')+mdl.find_matching_linear_constraints('im2_')


def repair_heuristic_explanation2(mdl, heuristic_explanation, network_input, network_output, n_classes, output_bounds):
    heuristic_explanation_complement = [i for i in range(len(network_input)) if i not in heuristic_explanation]
    output_variables = [mdl.get_var_by_name(f'o_{i}') for i in range(n_classes)]
    mdl.add_constraints(
        [mdl.get_var_by_name(f'x_{i}') == network_input[i] for i in heuristic_explanation], names='input')

    binary_variables = mdl.binary_var_list(n_classes - 1, name='b')
    mdl.add_constraint(mdl.sum(binary_variables) >= 1)

    mdl = insert_output_constraints_tjeng(mdl, output_variables, network_output, binary_variables, output_bounds)

    for feature in heuristic_explanation_complement:
        mdl.add_constraint(mdl.get_var_by_name(f'x_{feature}') == network_input[feature], ctname=f'constraint{feature}')
        mdl.solve(log_output=False)
        if mdl.solution is None:
            break

    return mdl.find_matching_linear_constraints('input')+mdl.find_matching_linear_constraints('constraint')


def refine_heuristic_explanation(mdl, heuristic_explanation, network_input, network_output, n_classes, output_bounds):
    return get_miminal_explanation(mdl, tf.constant(network_input), network_output, n_classes=n_classes,
                                method='tjeng', output_bounds=output_bounds, initial_explanation=heuristic_explanation)


if __name__ == '__main__':
    # dir_path = 'glass2'
    # feature_names = ['Refractive Index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium',
    #                  'Iron']
    # class_names = [0, 1]
    # categorical_names = {}
    dir_path = 'cleveland'
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    class_names = [0, 1, 2, 3, 4]
    categorical_names = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    categorical_names = {k: 'abcdefghijklmnopqrs' for k, v in dict(enumerate(feature_names)).items() if v in categorical_names}
    model_path = f'datasets\\{dir_path}\\model_2layers_{dir_path}.h5'
    model = tf.keras.models.load_model(model_path)

    data_test = pd.read_csv(f'datasets\\{dir_path}\\test.csv')
    data_train = pd.read_csv(f'datasets\\{dir_path}\\train.csv')
    df_data = data_train.append(data_test)
    data = df_data.to_numpy()
    network_input = data[60, :-1] # 59 é válida com SHAP
    network_input = np.array(tf.reshape(tf.constant(network_input), (1, -1)))
    predicted_class = np.argmax(model(network_input)[0])

    mdl, output_bounds = codify_network(model, df_data, method='tjeng', relaxe_constraints=False)
    explanation = get_miminal_explanation(mdl.clone(), tf.constant(network_input), predicted_class,
                                          n_classes=len(class_names), method='tjeng', output_bounds=output_bounds)
    n_features = len(explanation)

    # explainer = get_anchor_explainer(class_names, feature_names, data[:, :-1], categorical_names)
    # heuristic_explanation = get_anchor_explanation(network_input, model, explainer, class_names, feature_names)
    # explainer = get_lime_explainer(class_names, feature_names, data[:, :-1], categorical_names)
    # heuristic_explanation = get_lime_explanation(network_input, model, explainer, n_features, class_names, feature_names)
    explainer = get_shap_explainer(model, data[:, :-1])
    heuristic_explanation = get_shap_explanation(network_input, explainer, n_features, predicted_class)

    counter_example = validate_heuristic_explanation(mdl.clone(), heuristic_explanation, network_input[0], predicted_class, len(class_names), output_bounds)
    if counter_example is not None:
        new_explanation = repair_heuristic_explanation2(mdl.clone(), heuristic_explanation, network_input[0],
                                                       predicted_class, len(class_names), output_bounds)
    else:
        new_explanation = refine_heuristic_explanation(mdl.clone(), heuristic_explanation, network_input, predicted_class,
                                          n_classes=len(class_names), output_bounds=output_bounds)
