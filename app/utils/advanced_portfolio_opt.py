import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cvxopt
from scipy.optimize import minimize
import gurobipy as gp
from gurobipy import GRB

@dataclass
class MultiPeriodResult:
    weights: List[np.ndarray]
    expected_returns: List[float]
    risks: List[float]
    transaction_costs: List[float]
    metadata: Dict

class AdvancedPortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, transaction_costs: Optional[np.ndarray] = None,
                 risk_free_rate: float = 0.02):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.transaction_costs = transaction_costs or np.zeros(len(returns.columns))
        self.cov_matrix = returns.cov().values
        self.mean_returns = returns.mean().values

    def optimize_cvar(self, alpha: float = 0.05, lambda_risk: float = 0.5) -> np.ndarray:
        """
        Optimize portfolio using Conditional Value at Risk (CVaR)
        
        Args:
            alpha: Confidence level for CVaR
            lambda_risk: Risk aversion parameter
        """
        n_assets = len(self.returns.columns)
        n_scenarios = len(self.returns)
        
        # Create optimization model
        model = gp.Model('CVaR_Optimization')
        
        # Variables
        weights = model.addVars(n_assets, lb=0, ub=1, name='weights')
        var = model.addVar(name='VaR')
        z = model.addVars(n_scenarios, lb=0, name='z')
        
        # Objective: Maximize return - λ * CVaR
        scenario_returns = self.returns.values
        obj_return = gp.quicksum(weights[i] * self.mean_returns[i] 
                                for i in range(n_assets))
        obj_cvar = var + (1 / (alpha * n_scenarios)) * gp.quicksum(z)
        model.setObjective(obj_return - lambda_risk * obj_cvar, GRB.MAXIMIZE)
        
        # Constraints
        model.addConstr(gp.quicksum(weights) == 1, 'budget')
        
        for s in range(n_scenarios):
            model.addConstr(
                gp.quicksum(-scenario_returns[s, i] * weights[i] 
                           for i in range(n_assets)) - var <= z[s]
            )
        
        # Solve
        model.optimize()
        
        return np.array([weights[i].X for i in range(n_assets)])

    def optimize_multi_period(self, horizon: int, 
                            return_forecasts: List[np.ndarray],
                            risk_forecasts: List[np.ndarray],
                            constraints: Dict) -> MultiPeriodResult:
        """
        Multi-period portfolio optimization with transaction costs
        
        Args:
            horizon: Number of periods to optimize for
            return_forecasts: List of return forecasts for each period
            risk_forecasts: List of risk forecasts for each period
            constraints: Dictionary of constraints
        """
        n_assets = len(self.returns.columns)
        
        # Create optimization model
        model = gp.Model('Multi_Period_Optimization')
        
        # Variables
        weights = {}
        trades = {}
        for t in range(horizon):
            weights[t] = model.addVars(n_assets, lb=0, ub=1, name=f'weights_{t}')
            if t > 0:
                trades[t] = model.addVars(n_assets, lb=-GRB.INFINITY, 
                                        name=f'trades_{t}')
        
        # Objective: Maximize utility across periods
        utility = 0
        discount_factor = 0.95  # Time discount factor
        
        for t in range(horizon):
            # Expected return
            period_return = gp.quicksum(weights[t][i] * return_forecasts[t][i]
                                      for i in range(n_assets))
            
            # Risk penalty
            risk_penalty = gp.quicksum(weights[t][i] * risk_forecasts[t][i]
                                     for i in range(n_assets))
            
            # Transaction costs
            if t > 0:
                transaction_cost = gp.quicksum(
                    self.transaction_costs[i] * abs(trades[t][i])
                    for i in range(n_assets)
                )
            else:
                transaction_cost = 0
            
            utility += (discount_factor ** t) * (
                period_return - 0.5 * risk_penalty - transaction_cost
            )
        
        model.setObjective(utility, GRB.MAXIMIZE)
        
        # Constraints
        for t in range(horizon):
            # Budget constraint
            model.addConstr(gp.quicksum(weights[t]) == 1, f'budget_{t}')
            
            # Trading constraints
            if t > 0:
                for i in range(n_assets):
                    model.addConstr(
                        weights[t][i] - weights[t-1][i] == trades[t][i],
                        f'trade_balance_{t}_{i}'
                    )
            
            # Position limits
            if 'position_limits' in constraints:
                for i in range(n_assets):
                    model.addConstr(
                        weights[t][i] <= constraints['position_limits'][i],
                        f'position_limit_{t}_{i}'
                    )
            
            # Turnover constraints
            if 'max_turnover' in constraints and t > 0:
                model.addConstr(
                    gp.quicksum(abs(trades[t][i]) for i in range(n_assets)) <=
                    constraints['max_turnover'],
                    f'turnover_{t}'
                )
        
        # Solve
        model.optimize()
        
        # Extract results
        optimal_weights = []
        for t in range(horizon):
            optimal_weights.append(
                np.array([weights[t][i].X for i in range(n_assets)])
            )
        
        # Calculate metrics
        expected_returns = []
        risks = []
        transaction_costs_list = []
        
        for t in range(horizon):
            expected_returns.append(
                np.sum(optimal_weights[t] * return_forecasts[t])
            )
            risks.append(
                np.sum(optimal_weights[t] * risk_forecasts[t])
            )
            if t > 0:
                transaction_costs_list.append(
                    np.sum(np.abs(optimal_weights[t] - optimal_weights[t-1]) *
                          self.transaction_costs)
                )
            else:
                transaction_costs_list.append(0)
        
        return MultiPeriodResult(
            weights=optimal_weights,
            expected_returns=expected_returns,
            risks=risks,
            transaction_costs=transaction_costs_list,
            metadata={
                'objective_value': model.ObjVal,
                'solver_status': model.Status
            }
        )

    def optimize_robust_cvar(self, alpha: float = 0.05,
                           uncertainty_set: Dict[str, float]) -> np.ndarray:
        """
        Robust CVaR optimization with uncertainty sets
        
        Args:
            alpha: Confidence level for CVaR
            uncertainty_set: Dictionary defining uncertainty bounds
        """
        n_assets = len(self.returns.columns)
        n_scenarios = len(self.returns)
        
        # Create optimization model
        model = gp.Model('Robust_CVaR')
        
        # Variables
        weights = model.addVars(n_assets, lb=0, ub=1, name='weights')
        var = model.addVar(name='VaR')
        z = model.addVars(n_scenarios, lb=0, name='z')
        
        # Uncertainty sets
        mean_uncertainty = uncertainty_set.get('mean', 0.1)
        cov_uncertainty = uncertainty_set.get('covariance', 0.1)
        
        # Worst-case expected returns
        worst_case_returns = self.mean_returns - mean_uncertainty * \
                           np.abs(self.mean_returns)
        
        # Worst-case covariance
        worst_case_cov = self.cov_matrix * (1 + cov_uncertainty)
        
        # Objective: Minimize worst-case CVaR
        obj = var + (1 / (alpha * n_scenarios)) * gp.quicksum(z)
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        model.addConstr(gp.quicksum(weights) == 1, 'budget')
        
        # Return target constraint
        model.addConstr(
            gp.quicksum(weights[i] * worst_case_returns[i] 
                       for i in range(n_assets)) >= self.risk_free_rate,
            'return_target'
        )
        
        # CVaR constraints with worst-case scenarios
        for s in range(n_scenarios):
            scenario_return = gp.quicksum(
                weights[i] * (self.returns.iloc[s, i] - 
                            mean_uncertainty * abs(self.returns.iloc[s, i]))
                for i in range(n_assets)
            )
            model.addConstr(-scenario_return - var <= z[s])
        
        # Solve
        model.optimize()
        
        return np.array([weights[i].X for i in range(n_assets)])
