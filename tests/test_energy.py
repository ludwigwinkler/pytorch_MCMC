# import torch
# from mcmc.energy import LinearRegressionEnergy


# class TestLinearRegressionEnergy:
#     def test_least_squares_recovery(self):
#         torch.manual_seed(42)
#         # True parameters
#         m_true = 1.0
#         b_true = -1.0
#         sigma_true = 0.7
#         n = 1000
#         energy = LinearRegressionEnergy(m=m_true, b=b_true, sigma=sigma_true)
#         x, y = energy.sample(num_samples=n)
#         # Analytical least squares solution
#         X = torch.cat([x, torch.ones_like(x)], dim=1)  # shape (n, 2)
#         solution = torch.linalg.lstsq(X, y)
#         beta_hat = solution.solution  # shape (2, 1)
#         m_est = beta_hat[0, 0].item()
#         b_est = beta_hat[1, 0].item()
#         y_pred = m_est * x + b_est
#         sigma_est = torch.sqrt(((y - y_pred) ** 2).mean()).item()
#         # Assert close to true values
#         assert abs(m_est - m_true) < 0.05, f"m: {m_est} vs {m_true}"
#         assert abs(b_est - b_true) < 0.05, f"b: {b_est} vs {b_true}"
#         assert abs(sigma_est - sigma_true) < 0.05, \
# f"sigma: {sigma_est} vs {sigma_true}"
