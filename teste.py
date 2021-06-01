import docplex.mp.model as mp
import numpy as np
import tensorflow as tf
from milp import codify_network
import pandas as pd


def insert_input_output_constraints(mdl, input_variables, output_variables, network_input, network_output):

    mdl.add_constraints([input_variables[i] == feature.numpy() for i, feature in enumerate(network_input[0])])

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


if __name__ == '__main__':
    model_path = 'datasets\\voting\\model_voting.h5'
    model = tf.keras.models.load_model(model_path)
    mdl, input_variables, output_variables = codify_network(model, domain_input='I', bounds_input=[0, 2])

    data_test = pd.read_csv('datasets\\voting\\test.csv').to_numpy()
    network_input = data_test[2, 1:]

    network_input = tf.reshape(tf.constant(network_input), (1, 16))
    network_output = model.predict(tf.constant(network_input))[0]
    network_output = tf.argmax(network_output)

    #network_input = tf.reshape(tf.constant([2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1]), (1, 16))
    #network_output = model.predict(tf.constant(network_input))[0]
    print('Input: ', network_input)
    print('Output: ', network_output)

    mdl = insert_input_output_constraints(mdl, input_variables, output_variables, network_input, network_output)

    print(mdl.export_to_string())

    print("\n\nSolving model....\n")

    solution = mdl.solve(log_output=True)

    print(mdl.get_solve_status())
    print(solution)
