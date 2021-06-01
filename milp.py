import docplex.mp.model as mp
import numpy as np
import tensorflow as tf


def codify_network(model, domain_input=None, bounds_input=None):

    if domain_input is None:
        domain_input = 'C'
    if bounds_input is None:
        bounds_input = [None, None]

    layers = model.layers

    mdl = mp.Model()

    num_features = layers[0].get_weights()[0].shape[0]

    if domain_input in ['C', 'I', 'B']:
        domain_input = [domain_input]*num_features

    if type(bounds_input[0]) != list:
        bounds_input = [bounds_input]*num_features

    assert len(domain_input) == num_features, \
        f"The domains list must have length of {num_features}, but {len(domain_input)} received"

    assert len(bounds_input) == num_features, \
        f"The bounds list must have length of {num_features}, but {len(bounds_input)} received"

    input_variables = []
    for i in range(len(domain_input)):
        lb, ub = bounds_input[i]
        if domain_input[i] == 'C':
            input_variables.append(mdl.continuous_var(lb=lb, ub=ub, name=f'x_{i}'))
        elif domain_input[i] == 'I':
            input_variables.append(mdl.integer_var(lb=lb, ub=ub, name=f'x_{i}'))
        elif domain_input[i] == 'B':
            input_variables.append(mdl.binary_var(name=f'x_{i}'))

    intermediate_variables = []
    auxiliary_variables = []
    indicator_variables = []
    for i in range(len(layers)-1):
        weights = layers[i].get_weights()[0]
        intermediate_variables.append(mdl.continuous_var_list(weights.shape[1], lb=0, name='y', key_format=f"_{i}_%s"))
        auxiliary_variables.append(mdl.continuous_var_list(weights.shape[1], lb=0, name='s', key_format=f"_{i}_%s"))
        indicator_variables.append(mdl.binary_var_list(weights.shape[1], name='z', key_format=f"_{i}_%s"))

    output_variables = mdl.continuous_var_list(layers[-1].get_weights()[0].shape[1], name='o')

    for i in range(len(layers)):
        A = layers[i].get_weights()[0].T
        b = layers[i].bias.numpy()
        x = input_variables if i == 0 else intermediate_variables[i-1]
        y = intermediate_variables[i] if i != len(layers) - 1 else output_variables

        if i != len(layers) - 1:
            s = auxiliary_variables[i]
            z = indicator_variables[i]
            mdl.add_constraints([A[j, :]@x + b[j] == y[j] - s[j] for j in range(A.shape[0])])
            mdl.add_indicators(z, [y[j] <= 0 for j in range(len(z))], 1)
            mdl.add_indicators(z, [s[j] <= 0 for j in range(len(z))], 0)
        else:
            mdl.add_constraints([A[j, :] @ x + b[j] == y[j] for j in range(A.shape[0])])

    return mdl, input_variables, output_variables


if __name__ == '__main__':
    model = tf.keras.models.load_model('datasets\\voting\\model_voting.h5')
    mdl, input_variables, output_variables = codify_network(model, domain_input='I', bounds_input=[0, 2])
    print(mdl.export_to_string())

# X ---- E
# x1 == 1 /\ x2 == 3 /\ F /\ ~E    INSATISFÁTIVEL
# x1 >= 0 /\ x1 <= 100 /\ x2 == 3 /\ F /\ ~E    INSATISFÁTIVEL -> x1 n é relevante,  SATISFÁTIVEL -> x1 é relevante
'''
print("\n\nSolving model....\n")

msol = mdl.solve(log_output=True)
print(mdl.get_solve_status())
'''