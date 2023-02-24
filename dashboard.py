
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from simulations import SimulationCos, SimulationNK
  
if __name__ == '__main__':
    # create a streamlit app to run the simulation
    # and display the results

    simulation_classes = [SimulationCos, SimulationNK]
    simulation_class = st.sidebar.selectbox("Select simulation class", simulation_classes)
    # create an instance of the selected simulation class in order to inspect its member variables
    simulation_instance = simulation_class()
    # inspect simulation_instance and create a slider for each member variable
    member_variables = [attr for attr in dir(simulation_instance) 
        if not callable(getattr(simulation_instance, attr)) 
        and not attr.startswith("__")
        and not attr.startswith("_")
        and not attr.endswith("_s")]

    # create a sidebar to select the value of the member variables
    st.sidebar.title("Simulation parameters")
    st.sidebar.markdown("Select the simulation parameters and click on the button to run the simulation")
    
    values = {}
    for member_variable in member_variables:
        # get the value of the member variable
        value = getattr(simulation_instance, member_variable)
        # get the type of the member variable
        value_type = type(value)
        # if the type is int, the slider will have integer values from 0 to 100
        if value_type == int:
            values[member_variable] = st.sidebar.slider(member_variable, 0, 100, value)
        # if the type is float, the slider will have float values from 0.0 to 2.0
        elif value_type == float:
            values[member_variable] = st.sidebar.slider(member_variable, 0.0, 2.0, value)
        # if the type is bool, the slider will have boolean values
        elif value_type == bool:
            values[member_variable] = st.sidebar.checkbox(member_variable, value)
        # if the type is list, the slider will have a list of values
        elif value_type == list:
            values[member_variable] = st.sidebar.multiselect(member_variable, value, value)
        # if the type is str, the slider will have a string value
        elif value_type == str:
            values[member_variable] = st.sidebar.text_input(member_variable, value)
        # if the type is not recognized, the slider will have a string value
        else:
            values[member_variable] = st.sidebar.text_input(member_variable, str(value))  
    
    if st.sidebar.button("Run simulation"):
        # create a simulation instance with the selected values
        s = simulation_class(**values) 
        s.go()
        # create a streamlit app to visualize results
        # use a grid layout
        c1 = st.container()
        col1, col2 = c1.columns(2)
        col1.subheader(f"# companies alive = {s.companies_s[-1]}")
        col1.line_chart(s.companies_s)
        col2.subheader("# location on the knowledge landscape")
        try:
            df = pd.DataFrame(np.array([s._D, s._K]).transpose(), columns=['distance', 'knowledge'])
            chart1 = alt.Chart(df).mark_circle().encode(
                x='distance', y='knowledge', tooltip=['distance', 'knowledge'])
            mx = np.max(s._D) + 1
            xs = np.arange(0, mx, 0.1)
            ys = s.fitness(xs)
            df2 = pd.DataFrame(np.array([xs, ys]).transpose(), columns=['distance', 'knowledge'])
            chart2 = alt.Chart(df2).mark_line(color='black', opacity=0.5).encode(
                x='distance', y='knowledge')
            chart = chart1 + chart2
            col2.altair_chart(chart, use_container_width=True)
        except:
            pass
        c2 = st.container()
        col1, col2 = c2.columns(2)
        col1.subheader("Economic gini")
        col1.line_chart(s.economic_gini_s)
        # add a line chart to visualize knowledge gini
        # make the title of the chart "Knowledge gini"
        col2.subheader("Knowledge gini")
        col2.line_chart(s.knowledge_gini_s)
        # create a histogram to visualize the distribution of _E
        col1.subheader("Economic distribution")
        col1.bar_chart(s._E)
        # drop the NaN values from _E and sum the remaining values
        col1.markdown(f"### Market cap: {np.nansum(s._E):.0f}")
        # create a histogram to visualize the distribution of _K
        col2.subheader("Knowledge distribution")
        col2.bar_chart(s._K)