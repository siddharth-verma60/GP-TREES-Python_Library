import numpy as np
import copy
import math
import random
import Fitness
import Functions
from collections import deque
from operator import attrgetter


class GP_Tree():
    """Genetic-Programming Syntax Tree created for performing
the common GP operations on trees. This tree is traversed
and represented according to Depth-First-search (DFS) traversal.

Paramaters and functions used to describe the tree are described
as follows:"""

    # Node class of the tree which contains the terminal or the function with its children.
    class _Node:

        def __init__(self, data):

            # Data is the function or the terminal constant
            self.data = data

            # Parent of every node will also be provided (Not implemented yet).
            self.parent = None

            # Its size would be equal to the arity of the function. For the terminals, it would be none
            self.children = None

        # This is overriden to define the representation of every node. The function is called recursively
        # to build the whole representation of the tree.
        def __str__(self):

            # "retval" is the final string that would be returned. Here the content of the node is added.
            retval = str(self.data) + "=>"

            # Content of the children of the node is added here in retval.
            if (self.children != None):
                for child in self.children:
                    retval += str(child.data) + ", "

            retval += "END\n"  # After every node and its children, string is concatenated with "END"

            # Recursive calls to all the nodes of the tree.
            if (self.children != None):
                for child in self.children:
                    retval += child.__str__()

            return retval

    def __init__(self, function_set, terminal_set, num_features, min_depth, max_depth, fitness_metric):
        '''The constructor accepts the max and min depth values of
        the tree.'''

        self.function_set = function_set
        # A list of functions to be used. Custom functions can be created.'

        self.terminal_set = terminal_set
        # List of floating point or zero arity functions acting as the terminals
        # of the tree

        self.num_features = num_features
        # Specifies the num of features in the input file

        self.min_depth = min_depth
        # Specifies the minimum depth of the tree.

        self.max_depth = max_depth
        # Specifies the maximum depth of the tree.

        self.fitness_metric = fitness_metric
        # The fitness function metric to be used to calculate fitness.

        ###################################################################
        # Other parameters :
        ###################################################################

        self.root = None
        # This is the root of the tree. It is from here, that the tree is traversed by every member function.

        self.fitness = None
        # The fitness value of the syntax tree

        self.number_of_terminals = 0
        self.number_of_functions = 0
        # These parameters are required for calculating the "terminal_ratio" in generation methods.

        self._add_features_in_terminal_set(prefix="X")
        # Features are added in the final terminal_set

    # this returns the string representation of the root which builds the representation of the whole tree recursively.
    def __str__(self):
        return self.root.__str__()

    @property
    def terminal_ratio(self):
        # Returns the ratio of the number of terminals to the number of all the functions in the tree.
        return self.number_of_terminals / float(self.number_of_terminals + self.number_of_functions)

    # Adds the number of arguments as specified in num_features in the syntax tree. The arguments is prefixed
    # with "X" followed by the index number. Eg: X0, X1, X2 ....
    def _add_features_in_terminal_set(self, prefix):

        temp_list = []
        for i in range(self.num_features):
            feature_str = "{prefix}{index}".format(prefix=prefix, index=i)
            temp_list.append(feature_str)

        temp_list.extend(self.terminal_set)
        self.terminal_set = temp_list

    #####################################################################################
    #                            Tree Generation Methods                                #
    #####################################################################################

    # The main tree generation function. Recursive function that starts building the tree from the root and returns the root of the constructed tree.
    def _generate(self, condition, depth, height):

        node = None  # The node that would be returned and get assigned to the root of the tree.
        # See functions: 'generate_full' and 'generate_grow' for assignment to the root of the tree.

        # Condition to check if currently function is to be added. If the condition is false, then the terminal
        # is not yet reached and a function should be inserted.
        if (condition(depth, height) == False):
            node_data = random.choice(self.function_set)  # Randomly choosing a function from the function set
            node_arity = Functions.get_arity(
                node_data)  # Getting the arity of the function to determine the node's children

            node = GP_Tree._Node(node_data)  # Creating the node.
            self.number_of_functions += 1

            node.children = []  # Creating the empty children list
            for _ in range(node_arity):
                child = self._generate(condition, depth + 1, height)
                child.parent = node
                node.children.append(child)  # Children are added recursively.

        else:  # Now the terminal should be inserted
            node_data = random.choice(self.terminal_set)  # Choosing the terminal randomly.
            node = GP_Tree._Node(node_data)  # Creating the terminal node
            self.number_of_terminals += 1
            node.children = None  # Children is none as the arity of the terminals is 0.

        return node  # Return the node created.

    def generate_full(self):
        # The method constructs the full tree. Note that only the function 'condition' is different in the
        # 'generate_grow()' and 'generate_full()' methods.

        def condition(depth, height):
            return depth == height

        height = random.randint(self.min_depth, self.max_depth)
        self.root = self._generate(condition, 0, height)

    def generate_grow(self):
        # The method constructs a grown tree.

        def condition(depth, height):
            return depth == height or (depth >= self.min_depth and random.random() < self.terminal_ratio)

        height = random.randint(self.min_depth, self.max_depth)
        self.root = self._generate(condition, 0, height)

    def generate_half_and_half(self):
        # Half the time, the expression is generated with 'generate_full()', the other half,
        # the expression is generated with 'generate_grow()'.

        # Selecting grow or full method randomly.
        method = random.choice((self.generate_grow, self.generate_full))
        # Returns either a full or a grown tree
        method()

    #####################################################################################
    #    Tree Traversal Methods: Different ways of representing the tree expression     #
    #####################################################################################

    # Depth-First-Traversal of the tree. It first reads the node, then the left child and then the right child.
    def tree_expression_DFS(self):

        expression = []  # expression to be built and returned

        # Recursive function as an helper to this function.
        self._tree_expression_DFS_helper(self.root, expression)

        return expression

    # Helper recursive function needed by the function "tree_expression_DFS()".
    def _tree_expression_DFS_helper(self, node, expression):

        expression.append(node.data)  # Expression to be built.

        if (node.children != None):
            for child in node.children:  # Adding children to the expression recursively.
                self._tree_expression_DFS_helper(child, expression)

        return

        # Breadth-First-Traversal of the tree. It first reads the left child, then the node itself and then the right child.

    def tree_expression_BFS(self):
        q = deque()  # BFS is implemented using a queue (FIFO)
        expression = []  # Expression to be built and returned.

        # Adding root to the queue
        node = self.root
        q.append(node)

        while (q):
            popped_node = q.popleft()
            if (popped_node.children != None):
                for child in popped_node.children:
                    q.append(child)

            expression.append(popped_node.data)

        return expression

    #####################################################################################
    #                            Tree Evaluation Methods                                #
    #####################################################################################

    # This function evaluates the syntax tree and returns the value evaluated by the tree.
    def evaluate(self, X_Data):

        # X_Data : shape = [n_samples, num_features]
        # Training vectors, where n_samples is the number of samples and num_features is the number of features.

        # Return: Y_Pred: shape = [n_samples]
        # Evaluated value of the n_samples.

        Y_Pred = []

        for features in X_Data:
            if features.size != self.num_features:
                raise ValueError("Number of input features in X_Data is not equal to the parameter: 'num_features'.")

            Y_Pred.append(self._evaluate_helper(self.root, features))

        return np.array(Y_Pred)

    # Helper function for the func: "evaluate()". This makes recursive calls for the evaluation.
    def _evaluate_helper(self, node, X_Data):

        # Terminal nodes
        if (node.children == None):

            if isinstance(node.data, str):
                feature_name = node.data
                index = int(feature_name[1:])
                return X_Data[index]

            else:
                return node.data

        args = []  # Will contain the input arguments i.e the children of the function in the tree.

        for child in node.children:
            args.append(self._evaluate_helper(child, X_Data))  # Evaluation by the recursive calls.

        func = Functions.get_function(node.data)  # Get the function from the alias name
        return func(*args)  # Return the computed value

    #####################################################################################
    #                            Tree Fitness Measure Method                            #
    #####################################################################################

    # Function to evaluate the fitness of the tree using the specified metric for it.
    def evaluate_fitness(self, X_Data, Y_Data, Weights=None):

        # X_Data: np-array. Shape = (num_features, n_samples). Specifies the features of the samples.
        # Y_Data: np-array. Shape = (n_samples). These are the target values of n_samples.
        # Weights: np-array. Shape = (n_sample). Weights applied to individual sample

        Y_Pred = self.evaluate(X_Data)
        # Predicted values from the tree.

        fitness_function = Fitness.get_fitness_metric(self.fitness_metric)
        fitness = fitness_function(Y_Data, Y_Pred, Weights)

        self.fitness = Fitness.get_fitness_sign(self.fitness_metric) * fitness

        return self.fitness

#####################################################################################################
''' The Definition of the functions that are related to the population of the syntax trees.'''
#####################################################################################################

#####################################################################################
#                            Population Initialization                              #
#####################################################################################

def initialize_population(function_set, terminal_set, num_features, min_depth, max_depth, fitness_metric,
                          population_size):
    '''This function initializes the population of the trees. This needs to be changed according to the covering mechanism.
    Currently, it takes a parameter 'population_size' and returns the population of randomly created trees.'''

    population = []
    for _ in range(population_size):
        tree = GP_Tree(function_set, terminal_set, num_features, min_depth, max_depth, fitness_metric)
        tree.generate_half_and_half()
        population.append(tree)

    return population


#####################################################################################
#           Population Evaluation: Calculating fitness of each individual           #
#####################################################################################

def evaluate_population(population, X_Data, Y_Data, Weights=None):
    # The whole population is evaluated with the input X_Data and the fitness of each individual tree is calculated.

    for individual_tree in population:
        individual_tree.evaluate_fitness(X_Data, Y_Data, Weights)


#####################################################################################
#               Parent selection method for applying genetic operators              #
#####################################################################################

def tournament_selection(population, k, tournament_size):
    '''This function selects the best individual (based on fitness values) among 'tournament_size' randomly chosen
    individual trees, 'k' times. The list returned contains references to the syntax tree objects.

    population: A list of syntax trees to select from.
    k: The number of individuals to select.
    tournament_size: The number of individual trees participating in each tournament.
    returns: A list of selected individual trees.
    '''
    selected = []
    for _ in range(k):
        selected_aspirants = [random.choice(population) for i in range(tournament_size)]
        selected.append(max(selected_aspirants, key=attrgetter("fitness")))
    return selected


#####################################################################################
#                            GP Cross-over: One-point crossover                     #
#####################################################################################

def crossover_onepoint(population):
    ''' This method performs the one point crossover in the population. It selects the parents from the population
    through the selection method and then prepare a replica copy of those parents. It then searches for the nodes
    in the common subtrees of the selected parents and then choose a random nodes from both the trees and then swap
    them between esch other.'''

    # Parents are selected through the tournament selection. 2 parents are selected from the tournament size of 5.
    parents = tournament_selection(population, k=2, tournament_size=5)

    # Copy of the parents is created. It is in these copies, where we will perform the crossover and return them as
    # the offsprings.
    parent1 = copy.deepcopy(parents[0])
    parent2 = copy.deepcopy(parents[1])

    # Roots of the parents.
    root_parent1 = parent1.root
    root_parent2 = parent2.root

    # These lists will contain the nodes that are present in the common subtree. These are passed to a function:
    # "crossover_onepoint_helper" that checks for the common subtree in terms of the shape and populates these lists
    # with the common nodes.
    common_subtree1_nodes = []
    common_subtree2_nodes = []

    # Common subtrees are checked in parent1 and parent2 and the above lists are populated.
    crossover_onepoint_helper(root_parent1, root_parent2, common_subtree1_nodes, common_subtree2_nodes)

    if len(common_subtree1_nodes) > 1:  # Have to clear this doubt. What if no common subtree is there? Right now I
        # am returning the parents if such a scenario occurs.

        # Select the random node index to make a slice. This index would be common for both the trees.
        slice_index = random.randint(1, len(common_subtree1_nodes) - 1)

        # Select the nodes with the help of the slice_index.
        slice_node1 = common_subtree1_nodes[slice_index]
        slice_node2 = common_subtree2_nodes[slice_index]

        # "ancestor_node1" and "ancestor_node2" are the parent nodes of the sliced node. These will be needed while
        # applying the actual crossover.
        ancestor_node1 = slice_node1.parent
        ancestor_node2 = slice_node2.parent

        # Finding the selected node to slice in its parent node
        for i in range(len(ancestor_node1.children)):
            if ancestor_node1.children[i] == slice_node1:
                ancestor_node1.children[i] = slice_node2  # Putting the subtree selected from the other tree.
                slice_node2.parent = ancestor_node1  # Making the parent reference
                break  # Can come out of the loop from here.

        # Same thing happening for the other tree.
        for i in range(len(ancestor_node2.children)):
            if ancestor_node2.children[i] == slice_node2:
                ancestor_node2.children[i] = slice_node1
                slice_node1.parent = ancestor_node2
                break

    # offsprings are referenced here.
    offspring1 = parent1
    offspring2 = parent2

    return offspring1, offspring2


def crossover_onepoint_helper(node1, node2, common_subtree1_nodes, common_subtree2_nodes):
    ''' This is a helper function for crossover that recursively checks for the common subtrees between two GP trees according
    to the common shapes. It populates the lists in its arguments with the common nodes in the respective trees.
    '''

    # Base condition in case of a terminal nodes
    if (node1.children == None or node2.children == None):  # arity==0
        return

    # Base condition occuring on the nodes from where dissimilarity in the tree shape starts occuring
    if (len(node1.children) != len(node2.children)):
        common_subtree1_nodes.append(node1)
        common_subtree2_nodes.append(node2)
        return

    common_subtree1_nodes.append(node1)
    common_subtree2_nodes.append(node2)

    # Recursive calls to all the children. Reach here only when the length of children (arity) of both the nodes is same.
    for _ in range(len(node1.children)):
        crossover_onepoint_helper(node1.children[_], node2.children[_], common_subtree1_nodes, common_subtree2_nodes)


#####################################################################################
#                                 GP Mutation                                       #
#####################################################################################

def mutation_NodeReplacement(population):
    '''Replaces a randomly chosen node from the individual tree by a randomly chosen node with the same number
    of arguments from the attribute: "arity" of the individual node. It takes the input "population" and selects
    the parent to mutate.'''

    # Parent is selected through the tournament selection. 1 parent is selected from the tournament size of 5.
    parents = tournament_selection(population, k=1, tournament_size=5)

    # Copy of the parent is created. It is in this copy, where we will perform the mutation and return it as
    # the offspring.
    parent = copy.deepcopy(parents[0])

    # Root of the parent.
    root_parent = parent.root

    # List to store all the nodes in the parent chosen. This list is populated using the function "mutation_helper".
    all_nodes = []

    # Populating the list "all_nodes".
    mutation_helper(root_parent, all_nodes)

    mutation_point = random.choice(all_nodes)  # Choosing the mutation point

    if (mutation_point.children == None):  # Case 1: Mutation point is a terminal.
        new_terminal_data = random.choice(parent.terminal_set)
        mutation_point.data = new_terminal_data

    else:  # Case 2: Mutation point is a function
        mutation_point_arity = Functions.get_arity(mutation_point.data)

        while True:  # Finding the same arity function.
            new_function_data = random.choice(parent.function_set)
            if (Functions.get_arity(new_function_data) == mutation_point_arity):
                mutation_point.data = new_function_data
                break

    offspring = parent
    return offspring


def mutation_Uniform(population, random_subtree_root):
    '''Randomly select a mutation point in the individual tree, then replace the subtree at that point
    as a root by the "random_subtree_root" that was generated using one of the initialization methods.'''

    # Parent is selected through the tournament selection. 1 parent is selected from the tournament size of 5.
    parents = tournament_selection(population, k=1, tournament_size=5)

    # Copy of the parent is created. It is in this copy, where we will perform the mutation and return it as
    # the offspring.
    parent = copy.deepcopy(parents[0])

    # Root of the parent.
    root_parent = parent.root

    # List to store all the nodes in the parent chosen. This list is populated using the function "mutation_helper".
    all_nodes = []

    # Populating the list "all_nodes".
    mutation_helper(root_parent, all_nodes)

    mutation_point = random.choice(all_nodes)  # Choosing the mutation point randomly.
    ancestor_mutation_point = mutation_point.parent  # Saving the parent node of the mutation point

    if (ancestor_mutation_point == None):  # What to do if mutation point is root itself? Need to ask this too.
        # Right now, returning the tree itself.
        return parent

    # Performing the mutation.
    for i in range(len(ancestor_mutation_point.children)):
        if (ancestor_mutation_point.children[i] == mutation_point):
            ancestor_mutation_point.children[i] = random_subtree_root
            del (mutation_point)
            random_subtree_root.parent = ancestor_mutation_point
            break

    offspring = parent

    return offspring


def mutation_helper(node, all_nodes):
    ''' Helper function to store all the nodes in a tree and return the list of the stored nodes. Used by mutation
    methods.'''

    if (node.children == None):
        all_nodes.append(node)
        return

    all_nodes.append(node)

    for child in node.children:
        mutation_helper(child, all_nodes)



if __name__ == "__main__":

    ## Example to show the formation and representation of a single gp-tree.

    tree = GP_Tree(function_set=("add", "mul", "sub", "div", "cos", "sqrt", "absolute", "sin", "tan"),
                   terminal_set=(4.0, 9.0, 11.0, 0.0), num_features=5, min_depth=2, max_depth=4,
                   fitness_metric="Root_Mean_Square_Error")
    tree.generate_half_and_half()  # Generating the tree through ramped half and half method

    print(tree.tree_expression_DFS())  # Printing the tree in a Depth first order
    print(tree.tree_expression_BFS())  # Printing the tree in Breadth first order
    print()
    print(tree)  # Printing the tree. The tree is printed in the form of the actual tree. It starts with the root,
    # followed by an arrow with its children and an "END". Then the subsequent lines show the children nodes traversed down
    # along the depth. Terminal nodes are just ended by an "END".


    # Making the population of the syntax trees
    function_set = ("add", "mul", "sub", "div", "cos", "sqrt", "absolute", "sin", "tan")
    terminal_set = (5.0, 8.5, 2.0, 0.0)

    population = initialize_population(function_set, terminal_set, num_features=1, min_depth=2, max_depth=4,
                                       fitness_metric="Root_Mean_Square_Error", population_size=100)

    # Generating random data-points for testing
    X_Data = np.arange(0, 1, 0.01).reshape(100, 1)
    Y_Data = X_Data ** 4 + X_Data ** 3 + X_Data ** 2 + X_Data

    # Evaluating the whole population
    evaluate_population(population, X_Data, Y_Data)


    # Performing the mutation operation on the GP tree through Uniform mutation method.
    # Generating a random subtree for mutation.
    mut_subtree = GP_Tree(function_set, terminal_set, 1, 2, 3, "Root_Mean_Square_Error")
    mut_subtree.generate_full()
    root_mut_subtree = mut_subtree.root

    # Passing the population and the random subtree to the mutation function
    offspring = mutation_Uniform(population, root_mut_subtree)
    # Evaluating the offspring for updating the new fitness.
    offspring.evaluate_fitness(X_Data, Y_Data)

    # Performing the mutation operation on the GP tree through Node Replacement method.
    offspring = mutation_NodeReplacement(population)
    # Evaluating the offspring for updating the new fitness.
    offspring.evaluate_fitness(X_Data, Y_Data)


    # Performing the crossover operation on the GP tree through one point crossover method.
    offsprings = crossover_onepoint(population)

    # Evaluating the new fitness values of the new offsprings generated by the crossover method.
    offsprings[0].evaluate_fitness(X_Data, Y_Data)
    offsprings[1].evaluate_fitness(X_Data, Y_Data)

