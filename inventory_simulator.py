import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import norm


class InventorySimulator:
    def __init__(self, yearly_demand_mean: float,
                 yearly_demand_deviation: float,
                 vendor_lead_time_mean: float,
                 vendor_lead_time_deviation: float,
                 single_transaction_fixed_cost: float,
                 variable_transaction_cost: float,
                 holding_cost: float,
                 min_order_size: float,
                 stockout_cost: float, review_periods: list[int], confidence_level=0.95):
        """Creates a new InventorySimulator with all it´s parameters for a single product
        :param yearly_demand_mean: mean yearly demand, mu D  [unit]
        :param yearly_demand_deviation: yearly demand deviation,
            sigma D [unit]
        :param vendor_lead_time_mean: mean vendor lead time, mu L [days]
        :param vendor_lead_time_deviation: vendor lead time, sigma L [days]
        :param single_transaction_fixed_cost: the cost of buying a single
            unit, k [$/order]
        :param variable_transaction_cost: the cost of buying a unit,
            Ct [$/unit]
        :param holding_cost: average cost of having a unit in inventory
            for a whole year, h [$/(unit*year)]
        :param stockout_cost:  the capital lost from inventory that has
            become unavailable for the customer to purchase, [$/unit]
        :param confidence_level: 1 - alfa
            be included in the target inventory level calculation
        :param min_order_size: the order sizes must be multiples of this number. E.g:If I need 950 units, and packages
            come in increments of 200 I will order 1000
        :param review_periods: how often we will review stock. If we review every week place: [7], if we review every
            wednesday and sunday put [3,4] (alternating between 3 and 4 days)
        """

        self.daily_demand_mean = yearly_demand_mean / 365.0  # mu d [unit]
        self.daily_demand_deviation = yearly_demand_deviation / 365.0  # sigma d [unit]
        self.vendor_lead_time_mean = vendor_lead_time_mean
        self.vendor_lead_time_deviation = vendor_lead_time_deviation
        self.single_transaction_fixed_cost = single_transaction_fixed_cost
        self.variable_transaction_cost = variable_transaction_cost
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.review_periods = review_periods
        self.confidence_level = confidence_level
        self.min_order_size = min_order_size

        # Calculate z alpha [] using the confidence level
        self.z_score = norm.ppf(confidence_level)

        # Calculate review period R [day]
        review_period = np.mean(self.review_periods)

        # Calculate SS, Security Stock  [unit]
        self.security_stock = np.sqrt(
            (self.vendor_lead_time_mean + review_period)
            * (self.daily_demand_deviation ** 2)
            + (self.daily_demand_mean * self.vendor_lead_time_deviation) ** 2
        )

        # Calculate S, target inventory level [unit]
        self.target_inventory_level = self.daily_demand_mean \
            * (review_period + self.vendor_lead_time_mean) \
            + self.security_stock

        self.num_by_col = {'Inventory': 0, 'Demand': 1, 'Sales': 2,
                           'Lost Sales': 3, 'Order Arrivals': 4, 'Order Requests': 5}

        # Simulation variables
        self.simulation_data = np.zeros([len(self.num_by_col), 0])
        self.days_since_last_review = 0
        self.current_review_period_index = 0
        self.current_review_period = 0
        self.update_review_period()

    def update_review_period(self) -> None:
        """Updates the review period based on the current review period index"""
        self.current_review_period = self.review_periods[self.current_review_period_index]

    def move_to_next_review_period(self) -> None:
        """Gets the next review period and updated in the current_review period variable"""
        self.current_review_period_index += 1
        if self.current_review_period_index >= len(self.review_periods):
            self.current_review_period_index = 0

        self.days_since_last_review = 0
        self.update_review_period()

    def initialize_sim_variables(self, sim_days: int) -> None:
        """
        Updates the simulation variables to be able to simulate again
        :param sim_days: days that you will run the simulation
        """
        # Initialize variables
        self.simulation_data = np.zeros([len(self.num_by_col), sim_days])
        self.days_since_last_review = 0
        self.current_review_period_index = 0
        self.update_review_period()

        # The initial inventory will be set as the target inventory level
        self.simulation_data[self.num_by_col['Inventory']][0] = self.target_inventory_level

    def simulate(self, sim_days: int) -> None:
        """
        Iterates over all the days and calculates inventory, demand, sales, lost sales, order arrivals and order
            requests
        :param sim_days: days that you will run the simulation
        """

        self.initialize_sim_variables(sim_days)

        for day in range(1, sim_days):

            # Update days since last review of stock
            self.days_since_last_review += 1

            # Update inventory using past inventory and order arrivals
            inventory = \
                self.simulation_data[self.num_by_col['Inventory']][day - 1] \
                + self.simulation_data[self.num_by_col['Order Arrivals']][day]

            # If today is multiple of review period, check stock, and reorder
            if self.days_since_last_review == self.current_review_period:
                required_inventory = max(0, self.target_inventory_level - inventory)
                order_size = required_inventory

                # Re-adjust order based on min order size
                amount_to_order = (order_size // self.min_order_size + 1) * self.min_order_size

                # Place order
                if amount_to_order > 0:
                    lead_time = np.random.normal(
                        self.vendor_lead_time_mean, self.vendor_lead_time_deviation)
                    lead_time = int(max(1, round(lead_time)))  # Orders cannot arrive yesterday (min: 1)
                    self.simulation_data[self.num_by_col['Order Requests']][day] += amount_to_order

                    arrival_day = day + lead_time
                    if arrival_day < sim_days:
                        self.simulation_data[self.num_by_col['Order Arrivals']][arrival_day] += amount_to_order
                self.move_to_next_review_period()

            # Reduce inventory by today's demand
            demand = np.random.normal(self.daily_demand_mean, self.daily_demand_deviation)
            demand = max(0, demand)

            self.simulation_data[self.num_by_col['Demand']][day] = demand

            sales = min(inventory, demand)
            lost_sales = demand - sales
            self.simulation_data[self.num_by_col['Lost Sales']][day] = lost_sales
            self.simulation_data[self.num_by_col['Sales']][day] = sales

            # Set the inventory at the end of the day
            self.simulation_data[self.num_by_col['Inventory']][day] = inventory - sales

    def calculate_total_service_level(self) -> float:
        """
        Calculate the total service level (TSL), which is the percentage of review periods without a stockout
        :return: the current TSL
        """
        days_array = self.simulation_data[self.num_by_col['Inventory']]

        # Split the array into subsets of length of the review periods
        period_lengths = self.review_periods * (len(days_array) // sum(self.review_periods))
        periods = np.split(days_array, np.cumsum(period_lengths)[:-1])

        # Count the subsets without a zero
        count_without_zero = sum(not np.any(subset == 0) for subset in periods)

        # Calculate the percentage
        return count_without_zero / len(periods)

    def calculate_metrics(self) -> tuple[float, float, float]:
        """
        Calculates the pertinent KPIs
        :return: fill rate, TSL, daily stockout
        """
        fill_rate = self.simulation_data[self.num_by_col['Sales']].sum() \
            / self.simulation_data[self.num_by_col['Demand']].sum()

        total_service_level = self.calculate_total_service_level()

        daily_stock_out = np.count_nonzero(self.simulation_data[self.num_by_col['Inventory']]) \
            / len((self.simulation_data[self.num_by_col['Inventory']]))

        return fill_rate, total_service_level, daily_stock_out

    def show_sim_results_as_df(self) -> pd.DataFrame:
        """
        Based on the current simulation results inventory, demand, sales, lost sales, order arrivals and order
        requests create a df with the results
        :return: dataframe containing all the simulation data
        """
        return pd.DataFrame(self.simulation_data.T, columns=list(self.num_by_col.keys()))

    def plot_results(self) -> None:
        """Based on the current simulation results plots the inventory and stock outs"""

        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(range(len(self.simulation_data[0])),
                self.simulation_data[0], label='Inventory level')
        ax.plot(range(len(self.simulation_data[3])),
                self.simulation_data[3], label='Lost Sales')
        ax.set_xlabel('Day')
        ax.set_ylabel('Units')
        ax.legend()
        ax.axhspan(0, self.security_stock,
                   facecolor='gray', alpha=0.2, label="SS")

        plt.legend()
        plt.show()
