import streamlit as st

from optimization.genetic_algorithm import launch_optimization

st.title("Optimization")

st.write("This is a test")

modes = ["global", "local"]
selected_launch_mode = st.sidebar.selectbox("Select mode (focus)", modes)
limite_poste_livraisons = st.sidebar.slider("Limite Poste de livraisons (PL)", 1, 19, 19, 1)
limit_maisons_par_pl = st.sidebar.slider("Limite nnombre de maisons par PL", 1, 200, 10, 5)
# Parameters for the genetic algorithm
# Number of individuals in each generation, slider
pop_size = st.slider("Population size", 25, 200, 50, 25)
# Number of generations, slider
num_generations = st.slider("Number of generations", 10, 100, 50, 10)

## 2 col in streamlit body
col1, col2, col3 = st.columns(3)
# Selection retain rate, slider
selection_retain_rate = col1.slider("Selection retain rate", 0.0, 1.0, 0.5, 0.1)
# Selection rate, slider
selection_rate = col2.slider("Selection rate", 0.0, 1.0, 0.5, 0.1)
# crossover rate, slider
crossover_rate = col3.slider("Crossover rate", 0.0, 1.0, 0.5, 0.1)
# mutation rate, slider
mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.5, 0.1)

# Launch optimization
if st.button("Launch optimization"):

    # Launch optimization and display loading bar
    with st.spinner("Optimization in progress...") as spinner:

        result = launch_optimization(selected_launch_mode,
                                     limite_poste_livraisons=3, limit_maisons_par_pl=10,
                                     max_generations=1, population_size=5,
                                     mutation_rate=0.1, crossover_rate=0.7, selection_retain_rate=0.1,
                                     selection_rate=0.4,
                                     plot_with_streamlit=True, verbose=True)

    # Display results and dont quit the spinner
    if selected_launch_mode == "global":
        st.write("Global mode")
        st.write("Result :", result)
        spinner.text("Optimization done !")

        best_solution = result["global_best_reseau"]

        # select box
        selected_logement_name = st.selectbox("Selectionner une maison",
                                              [s.logement_name for s in best_solution.get_leaves_schedules()])

        # subtitle
        st.subheader("Maison {}".format(selected_logement_name))
        st.write(best_solution.get_leaf_schedule_by_name(selected_logement_name).to_df())
