import docplex.mp.model as mp
from cplex import infinity
import numpy as np
import tensorflow as tf
import pandas as pd


def codify_network_fischetti(mdl, layers, input_variables, auxiliary_variables, intermediate_variables, decision_variables, output_variables):
    output_bounds = []

    for i in range(len(layers)):
        A = layers[i].get_weights()[0].T
        b = layers[i].bias.numpy()
        x = input_variables if i == 0 else intermediate_variables[i-1]
        if i != len(layers) - 1:
            s = auxiliary_variables[i]
            a = decision_variables[i]
            y = intermediate_variables[i]
        else:
            y = output_variables

        for j in range(A.shape[0]):

            if i != len(layers) - 1:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j] - s[j], ctname=f'c_{i}_{j}')
                mdl.add_indicator(a[j], y[j] <= 0, 1)
                mdl.add_indicator(a[j], s[j] <= 0, 0)

                mdl.maximize(y[j])
                mdl.solve()
                ub_y = mdl.solution.get_objective_value()
                mdl.remove_objective()

                mdl.maximize(s[j])
                mdl.solve()
                ub_s = mdl.solution.get_objective_value()
                mdl.remove_objective()

                y[j].set_ub(ub_y)
                s[j].set_ub(ub_s)


            else:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f'c_{i}_{j}')
                mdl.maximize(y[j])
                mdl.solve()
                ub = mdl.solution.get_objective_value()
                mdl.remove_objective()

                mdl.minimize(y[j])
                mdl.solve()
                lb = mdl.solution.get_objective_value()
                mdl.remove_objective()

                y[j].set_ub(ub)
                y[j].set_lb(lb)
                output_bounds.append([lb, ub])

    return mdl, output_bounds


def codify_network_tjeng(mdl, layers, input_variables, auxiliary_variables, intermediate_variables, decision_variables, output_variables):
    output_bounds = []

    for i in range(len(layers)):
        A = layers[i].get_weights()[0].T
        b = layers[i].bias.numpy()
        x = input_variables if i == 0 else intermediate_variables[i-1]
        if i != len(layers) - 1:
            s = auxiliary_variables[i]
            a = decision_variables[i]
            y = intermediate_variables[i]
        else:
            s = output_variables

        for j in range(A.shape[0]):

            mdl.add_constraint(A[j, :] @ x + b[j] == s[j], ctname=f'c_{i}_{j}')
            mdl.maximize(s[j])
            mdl.solve()
            ub = mdl.solution.get_objective_value()
            mdl.remove_objective()

            if ub <= 0 and i != len(layers) - 1:
                mdl.remove_constraint(f'c_{i}_{j}')
                mdl.add_constraint(y[j] == 0, ctname=f'c_{i}_{j}')
                continue

            mdl.minimize(s[j])
            mdl.solve()
            lb = mdl.solution.get_objective_value()
            mdl.remove_objective()

            if lb >= 0 and i != len(layers) - 1:
                mdl.remove_constraint(f'c_{i}_{j}')
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f'c_{i}_{j}')
                continue

            s[j].set_ub(ub)
            s[j].set_lb(lb)

            if i != len(layers) - 1:
                mdl.add_constraint(y[j] <= s[j] - lb * (1 - a[j]))
                mdl.add_constraint(y[j] >= s[j])
                mdl.add_constraint(y[j] <= ub * a[j])
            else:
                output_bounds.append([lb, ub])

    return mdl, output_bounds


def codify_network(model, dataframe, method, relaxe_constraints):
    layers = model.layers
    num_features = layers[0].get_weights()[0].shape[0]
    mdl = mp.Model()

    domain_input, bounds_input = get_domain_and_bounds_inputs(dataframe)
    bounds_input = np.array(bounds_input)

    if relaxe_constraints:
        input_variables = mdl.continuous_var_list(num_features, lb=bounds_input[:, 0], ub=bounds_input[:, 1], name='x')
    else:
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
    decision_variables = []

    for i in range(len(layers)-1):
        weights = layers[i].get_weights()[0]
        intermediate_variables.append(mdl.continuous_var_list(weights.shape[1], lb=0, name='y', key_format=f"_{i}_%s"))

        if method == 'tjeng':
            auxiliary_variables.append(mdl.continuous_var_list(weights.shape[1], lb=-infinity, name='s', key_format=f"_{i}_%s"))
        else:
            auxiliary_variables.append(mdl.continuous_var_list(weights.shape[1], lb=0, name='s', key_format=f"_{i}_%s"))

        if relaxe_constraints and method == 'tjeng':
            decision_variables.append(mdl.continuous_var_list(weights.shape[1], name='a', lb=0, ub=1, key_format=f"_{i}_%s"))
        else:
            decision_variables.append(mdl.binary_var_list(weights.shape[1], name='a', lb=0, ub=1, key_format=f"_{i}_%s"))

    output_variables = mdl.continuous_var_list(layers[-1].get_weights()[0].shape[1], lb=-infinity, name='o')

    if method == 'tjeng':
        mdl, output_bounds = codify_network_tjeng(mdl, layers, input_variables, auxiliary_variables,
                                                  intermediate_variables, decision_variables, output_variables)
    else:
        mdl, output_bounds = codify_network_fischetti(mdl, layers, input_variables, auxiliary_variables,
                                                  intermediate_variables, decision_variables, output_variables)

    if relaxe_constraints:
        # Tighten domain of variables 'a'
        for i in decision_variables:
            for a in i:
                a.set_vartype('Integer')

        # Tighten domain of input variables
        for i, x in enumerate(input_variables):
            if domain_input[i] == 'I':
                x.set_vartype('Integer')
            elif domain_input[i] == 'B':
                x.set_vartype('Binary')
            elif domain_input[i] == 'C':
                x.set_vartype('Continuous')

    return mdl, output_bounds


def get_domain_and_bounds_inputs(dataframe):
    domain = []
    bounds = []
    for column in dataframe.columns[:-1]:
        if len(dataframe[column].unique()) == 2:
            domain.append('B')
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])
        elif np.any(dataframe[column].unique().astype(np.int64) != dataframe[column].unique().astype(np.float64)):
            domain.append('C')
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])
        else:
            domain.append('I')
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])

    return domain, bounds


if __name__ == '__main__':
    path_dir = 'voting'
    model = tf.keras.models.load_model(f'datasets\\{path_dir}\\model_{path_dir}.h5')

    data_test = pd.read_csv(f'datasets\\{path_dir}\\test.csv')
    data_train = pd.read_csv(f'datasets\\{path_dir}\\train.csv')
    data = data_train.append(data_test)

    mdl, bounds = codify_network_tjeng(model, data)
    print(mdl.export_to_string())
    print(bounds)

# X ---- E
# x1 == 1 /\ x2 == 3 /\ F /\ ~E    INSATISFÁTIVEL
# x1 >= 0 /\ x1 <= 100 /\ x2 == 3 /\ F /\ ~E    INSATISFÁTIVEL -> x1 n é relevante,  SATISFÁTIVEL -> x1 é relevante
'''
print("\n\nSolving model....\n")

msol = mdl.solve(log_output=True)
print(mdl.get_solve_status())
'''
