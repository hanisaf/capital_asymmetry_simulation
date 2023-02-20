
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
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
    initial_knowledge_location = st.sidebar.slider("Initial knowledge location", 0, 10, 0)
    # add a slider to select initial_economic_asymmetry
    initial_economic_stock = st.sidebar.slider("Initial economic stock", 0, 100, 50)
    exploration_norms = st.sidebar.slider("Exploration norms", 0, 25, 0 )
    # add a button to run the simulation
    if st.sidebar.button("Run simulation"):
        s = Simulation(simulation_time=simulation_time,
                    simulation_size=simulation_size,
                    initial_economic_stock=initial_economic_stock,
                    initial_knowledge_location=initial_knowledge_location,
                    exploration_norms=exploration_norms)  
        s.go(verbose=False)
        # create a streamlit app to visualize results
    
        # use a grid layout
        c1 = st.container()
        col1, col2 = c1.columns(2)
        col1.subheader("# companies alive")
        col1.line_chart(s.companies_s)
        col2.subheader("# location on the knowledge landscape")

        df = pd.DataFrame(np.array([s._D, s._K]).transpose(), columns=['distance', 'knowledge'])
        chart1 = alt.Chart(df).mark_circle().encode(
            x='distance', y='knowledge', tooltip=['distance', 'knowledge'])
        mx = np.max(s._D) + 1
        xs = np.arange(0, mx, 0.1)
        ys = s.landscape(xs)
        df2 = pd.DataFrame(np.array([xs, ys]).transpose(), columns=['distance', 'knowledge'])
        chart2 = alt.Chart(df2).mark_line(color='black', opacity=0.5).encode(
            x='distance', y='knowledge')
        chart = chart1 + chart2
        col2.altair_chart(chart, use_container_width=True)
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
        col1.markdown(f"### Market cap: {s._E.sum():.0f}")
        # create a histogram to visualize the distribution of _K
        col2.subheader("Knowledge distribution")
        col2.bar_chart(s._K)#.title("Knowledge distribution")