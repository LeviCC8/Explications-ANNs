import docplex.mp.model as mp
import numpy as np
import tensorflow as tf
from milp import codify_network
import pandas as pd
from time import time
from statistics import mean
import cProfile


def insert_input_output_constraints_fischetti(mdl, output_variables, network_output, binary_variables):
    to_remove = []

    variable_output = output_variables[network_output]
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            p = binary_variables[aux_var]
            aux_var += 1
            to_remove.append(mdl.add_indicator(p, variable_output <= output, 1))

    return mdl, to_remove


def insert_output_constraints_tjeng(mdl, output_variables, network_output, binary_variables, output_bounds):
    to_remove = []

    variable_output = output_variables[network_output]
    upper_bounds_diffs = output_bounds[network_output][1] - np.array(output_bounds)[:, 0]  # Output i: oi - oj <= u1 = ui - lj
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            ub = upper_bounds_diffs[i]
            to_remove.append(mdl.add_constraint(variable_output <= output + mdl.sum([(1-b)*(ub/2) if j == aux_var else
                                                                    b*(ub/2) for j, b in enumerate(binary_variables)])))
            aux_var += 1

    return mdl, to_remove


def get_miminal_explanation(mdl, network_input, network_output, n_classes, method, output_bounds=None):
    assert not (method == 'tjeng' and output_bounds == None), 'If the method tjeng is chosen, output_bounds must be passed.'

    input_variables = [mdl.get_var_by_name(f'x_{i}') for i in range(len(network_input[0]))]
    output_variables = [mdl.get_var_by_name(f'o_{i}') for i in range(n_classes)]
    input_constraints = mdl.add_constraints([input_variables[i] == feature.numpy() for i, feature in enumerate(network_input[0])], names='input')
    binary_variables = mdl.binary_var_list(n_classes - 1, name='b')

    if method == 'tjeng':
        mdl.add_constraint(mdl.sum(binary_variables) == 1)
    else:
        mdl.add_constraint(mdl.sum(binary_variables) >= 1)

    for i in range(len(network_input[0])):
        mdl.remove_constraint(input_constraints[i])
        if method == 'tjeng':
            mdl, to_remove = insert_output_constraints_tjeng(mdl, output_variables, network_output, binary_variables, output_bounds)
        else:
            mdl, to_remove = insert_input_output_constraints_fischetti(mdl, output_variables, network_output, binary_variables)

        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(input_constraints[i])

        mdl.remove_constraints(to_remove)

    return mdl.find_matching_linear_constraints('input')


def main():
    dir_path = 'spect'
    n_classes = 2
    #method = 'fischetti'
    method = 'tjeng'

    data_test = pd.read_csv(f'datasets\\{dir_path}\\test.csv')
    data_train = pd.read_csv(f'datasets\\{dir_path}\\train.csv')

    data = data_train.append(data_test)

    model_path = f'datasets\\{dir_path}\\model_{dir_path}.h5'
    model = tf.keras.models.load_model(model_path)

    mdl, output_bounds = codify_network(model, data, method, relaxe_constraints=True)

    time_list = []
    len_list = []
    data = data.to_numpy()
    for i in range(data.shape[0]):
        print(i)
        network_input = data[i, :-1]

        network_input = tf.reshape(tf.constant(network_input), (1, -1))
        network_output = model.predict(tf.constant(network_input))[0]
        network_output = tf.argmax(network_output)

        mdl_aux = mdl.clone()
        start = time()
        if method == 'tjeng':
            explanation = get_miminal_explanation(mdl_aux, network_input, network_output,
                                                  n_classes=n_classes, method='tjeng', output_bounds=output_bounds)
        else:
            explanation = get_miminal_explanation(mdl_aux, network_input, network_output, n_classes=n_classes,
                                                  method='fischetti')

        time_list.append(time() - start)

        len_list.append(len(explanation))
    print(f'Explication sizes:\nm: {min(len_list)}\na: {mean(len_list)}\nM: {max(len_list)}')
    print(f'Time:\nm: {min(time_list)}\na: {mean(time_list)}\nM: {max(time_list)}')


if __name__ == '__main__':
    #cProfile.run('main()', sort='time')
    main()
