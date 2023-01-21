import streamlit as st

from optimization.genetic_algorithm import launch_optimization

st.title("Optimization ⚙️")

modes = ["global", "local"]
selected_launch_mode = st.sidebar.selectbox("Select mode (focus)", modes)
limite_poste_livraisons = st.sidebar.slider("Limit number of PL (Poste de livraisons)", 1, 19, 3, 1)
limit_maisons_par_pl = st.sidebar.slider("Limit number of houses per PL", 1, 200, 3, 5)

col1, col2 = st.columns(2)

# Parameters for the genetic algorithm
# Number of individuals in each generation, slider
pop_size = col1.slider("Population size", 25, 200, 50, 25)
# Number of generations, slider, default 5
num_generations = col2.selectbox("Number of generations", [2, 5, 10, 20, 50, 100])

## 2 col in streamlit body
col1, col2, col3, col4 = st.columns(4)
# Selection retain rate, slider
selection_retain_rate = col1.selectbox("Selection retain rate", [0.1, 0.2, 0.3])
# Selection rate, slider
selection_rate = col2.selectbox("Selection rate", [0.4, 0.5, 0.6])
# crossover rate, slider
crossover_rate = col3.selectbox("Crossover rate", [0.4, 0.5, 0.5, 0.6, 0.7], index=2)
# mutation rate, slider
mutation_rate = col4.selectbox("Mutation rate", [0.1, 0.2, 0.3, 0.4])

# Launch optimization
# the button stay pressed until the end of the optimization
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False


def callback():
    st.session_state.button_clicked = True


if st.button("Launch optimization", on_click=callback):

    # Launch optimization and display loading bar
    with st.spinner("Optimization in progress...") as spinner:
        result = launch_optimization(
            selected_launch_mode,
            limite_poste_livraisons=limite_poste_livraisons,
            limit_maisons_par_pl=limit_maisons_par_pl,
            max_generations=num_generations,
            population_size=pop_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            selection_retain_rate=selection_retain_rate,
            selection_rate=selection_rate,
            display_plot=selected_launch_mode == "global",
            plot_with_streamlit=selected_launch_mode == "global",
            verbose=True,
        )

    # Display results and dont quit the spinner

    if selected_launch_mode == "global":
        st.write("Result :", result)
        best_solution = result["global_best_reseau"]
        st.header("📶 Poste source ({:.1f}kW)".format(best_solution.get_global_consommation()))

        for s in best_solution.get_leaves_schedules():
            # subtitle
            st.subheader("🏠 Maison {} ({:.1f}kW)".format(s.logement_name, s.consommation))
            st.write(s.logement_equipements)
            st.write(best_solution.get_leaf_schedule_by_name(s.logement_name).to_df())

    else:
        st.write("Result :", result)

        best_reseaux_pl = result["local_best_reseaux"]
        for poste_livraison in best_reseaux_pl.keys():
            best_solution = best_reseaux_pl[poste_livraison]
            # title h2
            st.header(
                "📶 Poste de livraison {} ({:.1f}kW)".format(poste_livraison, best_solution.get_global_consommation()))
            for s in best_solution.get_leaves_schedules():
                # h3
                st.subheader("🏠 Maison {} ({:.1f}kW)".format(s.logement_name, s.consommation))
                st.write(s.logement_equipements)
                #
                st.write(best_solution.get_leaf_schedule_by_name(s.logement_name).to_df())
