"""
Author: TOM BUTLER

Python script for a streamlit dashboard showcasing the effects of imbalanced
training data on final predictions with the Titanic dataset.

"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")


class Dashboard:
    def __init__(self):
        self.data = None

        self.load_data()
        self.title()

    def title(self):
        st.title("Imbalanced Training Data")

    def load_data(self):
        # TODO load in the scores from each model
        self.data = pd.read_csv(...)

    def plot_scores(self):
        # TODO matplotlib plot of the scores for a given model
        pass


if __name__ == "__main__":
    dashboard = Dashboard()
