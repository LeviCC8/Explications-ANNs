import docplex.mp.model as mp
import numpy as np
import tensorflow as tf
from milp import codify_network
import pandas as pd
from time import time
from statistics import mean
import cProfile
import pandas as pd


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
            #to_remove.append(mdl.add_constraint(variable_output <= output + mdl.sum([(1-b)*(ub/2) if j == aux_var else
            #                                                        b*(ub/2) for j, b in enumerate(binary_variables)])))
            z = binary_variables[aux_var]
            to_remove.append(mdl.add_constraint(variable_output - output - ub*(1 - z) <= 0))
            aux_var += 1

    return mdl, to_remove


def get_miminal_explanation(mdl, network_input, network_output, n_classes, method, output_bounds=None):
    assert not (method == 'tjeng' and output_bounds == None), 'If the method tjeng is chosen, output_bounds must be passed.'

    input_variables = [mdl.get_var_by_name(f'x_{i}') for i in range(len(network_input[0]))]
    output_variables = [mdl.get_var_by_name(f'o_{i}') for i in range(n_classes)]
    input_constraints = mdl.add_constraints([input_variables[i] == feature.numpy() for i, feature in enumerate(network_input[0])], names='input')
    binary_variables = mdl.binary_var_list(n_classes - 1, name='b')

    #if method == 'tjeng':
    #    mdl.add_constraint(mdl.sum(binary_variables) == 1)
    #else:
    #    mdl.add_constraint(mdl.sum(binary_variables) >= 1)
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
    datasets = [#{'dir_path': 'australian', 'n_classes': 2}, {'dir_path': 'auto', 'n_classes': 5},
                #{'dir_path': 'backache', 'n_classes': 2}, {'dir_path': 'breast-cancer', 'n_classes': 2},
                #{'dir_path': 'cleve', 'n_classes': 2}, {'dir_path': 'cleveland', 'n_classes': 5},
                {'dir_path': 'glass', 'n_classes': 5},]
                #{'dir_path': 'glass2', 'n_classes': 2},
                #{'dir_path': 'heart-statlog', 'n_classes': 2}, {'dir_path': 'hepatitis', 'n_classes': 2},
                #{'dir_path': 'spect', 'n_classes': 2}, {'dir_path': 'voting', 'n_classes': 2}]

    configurations = [#{'method': 'fischetti', 'relaxe_constraints': True},
                      #{'method': 'fischetti', 'relaxe_constraints': False},
                      #{'method': 'tjeng', 'relaxe_constraints': True},
                      {'method': 'tjeng', 'relaxe_constraints': False}]

    df = {'fischetti': {True: {'size': [], 'milp_time': [], 'build_time': []},
                        False: {'size': [], 'milp_time': [], 'build_time': []}},
          'tjeng': {True: {'size': [], 'milp_time': [], 'build_time': []},
                    False: {'size': [], 'milp_time': [], 'build_time': []}}}

    for dataset in datasets:
        dir_path = dataset['dir_path']
        n_classes = dataset['n_classes']

        for config in configurations:
            print(dataset, config)

            method = config['method']
            relaxe_constraints = config['relaxe_constraints']

            data_test = pd.read_csv(f'datasets\\{dir_path}\\test.csv')
            data_train = pd.read_csv(f'datasets\\{dir_path}\\train.csv')

            data = data_train.append(data_test)

            model_path = f'datasets\\{dir_path}\\model_4layers_{dir_path}.h5'
            model = tf.keras.models.load_model(model_path)

            codify_network_time = []
            for _ in range(5):
                start = time()
                mdl, output_bounds = codify_network(model, data, method, relaxe_constraints)
                codify_network_time.append(time() - start)
                print(codify_network_time[-1])

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

                explanation = get_miminal_explanation(mdl_aux, network_input, network_output,
                                                      n_classes=n_classes, method=method, output_bounds=output_bounds)

                time_list.append(time() - start)

                len_list.append(len(explanation))

            df[method][relaxe_constraints]['size'].extend([min(len_list), mean(len_list), max(len_list)])
            df[method][relaxe_constraints]['milp_time'].extend([min(time_list), mean(time_list), max(time_list)])
            df[method][relaxe_constraints]['build_time'].extend([min(codify_network_time), mean(codify_network_time), max(codify_network_time)])

            print(f'Explication sizes:\nm: {min(len_list)}\na: {mean(len_list)}\nM: {max(len_list)}')
            print(f'Time:\nm: {min(time_list)}\na: {mean(time_list)}\nM: {max(time_list)}')
            print(f'Build Time:\nm: {min(codify_network_time)}\na: {mean(codify_network_time)}\nM: {max(codify_network_time)}')
            'a'+1

    df = {'fischetti_relaxe_size': df['fischetti'][True]['size'],
          'fischetti_relaxe_time': df['fischetti'][True]['milp_time'],
          'fischetti_relaxe_build_time': df['fischetti'][True]['build_time'],
          'fischetti_not_relaxe_size': df['fischetti'][False]['size'],
          'fischetti_not_relaxe_time':  df['fischetti'][False]['milp_time'],
          'fischetti_not_relaxe_build_time': df['fischetti'][False]['build_time'],
          'tjeng_relaxe_size': df['tjeng'][True]['size'],
          'tjeng_relaxe_time': df['tjeng'][True]['milp_time'],
          'tjeng_relaxe_build_time': df['tjeng'][True]['build_time'],
          'tjeng_not_relaxe_size': df['tjeng'][False]['size'],
          'tjeng_not_relaxe_time': df['tjeng'][False]['milp_time'],
          'tjeng_not_relaxe_build_time': df['tjeng'][False]['build_time']}

    index_label = []
    for dataset in datasets:
        index_label.extend([f"{dataset['dir_path']}_m", f"{dataset['dir_path']}_a", f"{dataset['dir_path']}_M"])
    df = pd.DataFrame(data=df, index=index_label)
    df.to_csv('teste.csv')


if __name__ == '__main__':
    #cProfile.run('main()', sort='time')
    main()
