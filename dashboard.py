
import streamlit as st
from simulation import Simulation

if __name__ == '__main__':
    # create a streamlit app to run the simulation
    # and display the results
    # add a selector bar to select the simulation parameters
    # and a button to run the simulation
    st.sidebar.title("Setup")
    st.sidebar.markdown("Select the simulation parameters and click on the button to run the simulation")
    simulation_time = st.sidebar.slider("Simulation time", 1, 100, 50)
    simulation_size = st.sidebar.slider("Simulation size", 1, 100, 50)
    # add a slider to select initial_knowledge_asymmetry
    initial_knowledge_asymmetry = st.sidebar.slider("Initial knowledge symmetry", 0, 100, 1)
    # add a slider to select initial_economic_asymmetry
    initial_economic_asymmetry = st.sidebar.slider("Initial economic symmetry", 0, 100, 1)
    # add a button to run the simulation
    if st.sidebar.button("Run simulation"):
        s = Simulation(simulation_time=simulation_time,
                    simulation_size=simulation_size,
                    initial_economic_asymmetry=initial_economic_asymmetry,
                    initial_knowledge_asymmetry=initial_knowledge_asymmetry)
        s.go(verbose=False)
        # create a streamlit app to visualize results

        # use a grid layout
        c1 = st.container()
        col1, col2 = c1.columns(2)
        col1.subheader("# companies alive")
        col1.line_chart(s.companies_s)
        c2 = st.container()
        col1, col2 = c2.columns(2)
        col1.subheader("Economic gini")
        col1.line_chart(s.economic_gini_s)#.title("Economic gini")
        # add a line chart to visualize knowledge gini
        # make the title of the chart "Knowledge gini"
        col2.subheader("Knowledge gini")
        col2.line_chart(s.knowledge_gini_s)#.title("Knowledge gini")
        # create a histogram to visualize the distribution of _E
        col1.subheader("Economic distribution")
        col1.bar_chart(s._E)#.title("Economic distribution")
        # create a histogram to visualize the distribution of _K
        col2.subheader("Knowledge distribution")
        col2.bar_chart(s._K)#.title("Knowledge distribution")