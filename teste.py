import docplex.mp.model as mp
import numpy as np
import tensorflow as tf
from milp import codify_network_fischetti, codify_network_tjeng, get_domain_and_bounds_inputs
import pandas as pd
from time import time
from statistics import mean


def insert_input_output_constraints_fischetti(mdl, network_input, network_output, indexes, n_classes):

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


def insert_input_output_constraints_tjeng(mdl, network_input, network_output, indexes, n_classes, output_bounds):

    input_variables = [mdl.get_var_by_name(f'x_{i}') for i in range(len(network_input[0]))]
    output_variables = [mdl.get_var_by_name(f'o_{i}') for i in range(n_classes)]

    mdl.add_constraints([input_variables[i] == feature.numpy() for i, feature in enumerate(network_input[0]) if i in indexes])

    variable_output = output_variables[network_output]
    upper_bounds_diffs = output_bounds[network_output][1] - np.array(output_bounds)[:, 0]  # Output i: oi - oj <= u1 = ui - lj
    binary_variables = mdl.binary_var_list(n_classes - 1, name='b')
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            ub = upper_bounds_diffs[i]
            mdl.add_constraint(variable_output <= output + mdl.sum([(1-b)*(ub/2) if j == aux_var else b*(ub/2)
                                                                    for j, b in enumerate(binary_variables)]))
            aux_var += 1

    mdl.add_constraint(mdl.sum(binary_variables) == 1)

    return mdl


def get_miminal_explanation(mdl, network_input, network_output, n_classes, method, output_bounds=None):
    assert not (method == 'tjeng' and output_bounds == None), 'If the method tjeng is chosen, output_bounds must be passed.'

    indexes_to_keep = list(range(len(network_input[0])))
    for i in range(len(network_input[0])):
        indexes = indexes_to_keep.copy()
        indexes.remove(i)
        if method == 'tjeng':
            mdl_aux = insert_input_output_constraints_tjeng(mdl.clone(), network_input, network_output, indexes, n_classes, output_bounds)
        else:
            mdl_aux = insert_input_output_constraints_fischetti(mdl.clone(), network_input, network_output, indexes, n_classes)
        mdl_aux.solve(log_output=False)
        if mdl_aux.solution is None:
            indexes_to_keep.remove(i)

    return [mdl.get_var_by_name(f'x_{i}') for i in range(len(network_input[0])) if i in indexes_to_keep]


if __name__ == '__main__':
    dir_path = 'backache'
    n_classes = 2
    method = 'tjeng'

    data_test = pd.read_csv(f'datasets\\{dir_path}\\test.csv')
    data_train = pd.read_csv(f'datasets\\{dir_path}\\train.csv')

    data = data_train.append(data_test)

    model_path = f'datasets\\{dir_path}\\model_{dir_path}.h5'
    model = tf.keras.models.load_model(model_path)
    domain, bounds = get_domain_and_bounds_inputs(data)
    print('Domain: ', domain)
    print('Bounds: ', bounds)

    if method == 'tjeng':
        mdl, output_bounds = codify_network_tjeng(model, data)
    else:
        mdl = codify_network_fischetti(model, domain_input=domain, bounds_input=bounds)

    time_list = []
    len_list = []
    data = data.to_numpy()
    for i in range(data.shape[0]):
        print(i)
        network_input = data[i, :-1]

        network_input = tf.reshape(tf.constant(network_input), (1, -1))
        network_output = model.predict(tf.constant(network_input))[0]
        network_output = tf.argmax(network_output)

        start = time()
        if method == 'tjeng':
            explanation = get_miminal_explanation(mdl, network_input, network_output,
                                                  n_classes=n_classes, method='tjeng', output_bounds=output_bounds)
        else:
            explanation = get_miminal_explanation(mdl, network_input, network_output, n_classes=n_classes,
                                                  method='fischetti')

        time_list.append(time() - start)

        len_list.append(len(explanation))
    print(f'Explication sizes:\nm: {min(len_list)}\na: {mean(len_list)}\nM: {max(len_list)}')
    print(f'Time:\nm: {min(time_list)}\na: {mean(time_list)}\nM: {max(time_list)}')
