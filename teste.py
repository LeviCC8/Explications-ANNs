import docplex.mp.model as mp
import numpy as np
import tensorflow as tf
from milp import codify_network
import pandas as pd
from time import time
from statistics import mean


def insert_input_output_constraints(mdl, network_input, network_output, indexes, n_classes):

    input_variables = [mdl.get_var_by_name(f'x_{i}') for i in range(len(network_input[0]))]
    output_variables = [mdl.get_var_by_name(f'o_{i}') for i in range(n_classes)]

    mdl.add_constraints([input_variables[i] == feature.numpy() for i, feature in enumerate(network_input[0]) if i in indexes])

    variable_output = output_variables[network_output]
    indicator_variables = mdl.binary_var_list(len(output_variables) - 1, name='p')
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            p = indicator_variables[aux_var]
            aux_var += 1
            mdl.add_indicator(p, variable_output <= output, 1)

    mdl.add_constraint(mdl.sum(indicator_variables) >= 1)

    return mdl


def get_miminal_explanation(mdl, network_input, network_output, n_classes):

    indexes_to_keep = list(range(len(network_input[0])))
    for i in range(len(network_input[0])):
        indexes = indexes_to_keep.copy()
        indexes.remove(i)
        mdl_aux = insert_input_output_constraints(mdl.clone(), network_input, network_output, indexes, n_classes)
        mdl_aux.solve(log_output=False)
        if mdl_aux.solution is None:
            indexes_to_keep.remove(i)

    return [mdl.get_var_by_name(f'x_{i}') for i in range(len(network_input[0])) if i in indexes_to_keep]


if __name__ == '__main__':
    model_path = 'datasets\\voting\\model_voting.h5'
    model = tf.keras.models.load_model(model_path)
    mdl = codify_network(model, domain_input='I', bounds_input=[0, 2])

    data_test = pd.read_csv('datasets\\voting\\test.csv').to_numpy()
    data_train = pd.read_csv('datasets\\voting\\train.csv').to_numpy()

    data = np.append(data_test, data_train, axis=0)

    time_list = []
    len_list = []
    for i in range(data.shape[0]):
        print(i)
        network_input = data[i, 1:]

        network_input = tf.reshape(tf.constant(network_input), (1, 16))
        network_output = model.predict(tf.constant(network_input))[0]
        network_output = tf.argmax(network_output)

        start = time()
        explanation = get_miminal_explanation(mdl, network_input, network_output, n_classes=2)
        time_list.append(time() - start)

        len_list.append(len(explanation))
    print(f'Explication sizes:\nm: {min(len_list)}\na: {mean(len_list)}\nM: {max(len_list)}')
    print(f'Time:\nm: {min(time_list)}\na: {mean(time_list)}\nM: {max(time_list)}')
