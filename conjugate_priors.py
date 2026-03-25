import numpy as np
import matplotlib.pyplot as plt


def gaussian_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
	"""Return N(x | mean, variance) for array x."""
	coeff = 1.0 / np.sqrt(2.0 * np.pi * variance)
	exponent = -0.5 * ((x - mean) ** 2) / variance
	return coeff * np.exp(exponent)


def posterior_parameters(
	data: np.ndarray, prior_mean: float, prior_variance: float, obs_variance: float
) -> tuple[float, float]:
	"""Compute posterior mean/variance for Gaussian likelihood with known variance."""
	n = data.size
	sample_mean = float(np.mean(data))

	prior_precision = 1.0 / prior_variance
	data_precision = n / obs_variance
	posterior_variance = 1.0 / (prior_precision + data_precision)
	posterior_mean = posterior_variance * (
		prior_precision * prior_mean + data_precision * sample_mean
	)
	return posterior_mean, posterior_variance


def main() -> None:
	# Example observed data (known observation variance model)
	data = np.array([2.7, 2.9, 3.2, 2.8, 3.0, 3.1, 2.6, 2.95])

	# Prior: mu ~ N(mu0, tau0^2)
	mu0 = 0.5
	tau0_sq = 1.0

	# Likelihood model: x_i | mu ~ N(mu, sigma^2), with known sigma^2
	sigma_sq = 0.49

	# Likelihood as a function of mu is proportional to N(mu | x_bar, sigma^2 / n)
	n = data.size
	x_bar = float(np.mean(data))
	like_mean = x_bar
	like_var = sigma_sq / n

	post_mean, post_var = posterior_parameters(data, mu0, tau0_sq, sigma_sq)

	# Plot range chosen to include all three curves cleanly
	grid_min = min(mu0 - 3.5 * np.sqrt(tau0_sq), x_bar - 3.5 * np.sqrt(like_var))
	grid_max = max(mu0 + 3.5 * np.sqrt(tau0_sq), x_bar + 3.5 * np.sqrt(like_var))
	mu_grid = np.linspace(grid_min, grid_max, 1200)

	prior_pdf = gaussian_pdf(mu_grid, mu0, tau0_sq)
	likelihood_pdf = gaussian_pdf(mu_grid, like_mean, like_var)
	posterior_pdf = gaussian_pdf(mu_grid, post_mean, post_var)

	plt.style.use("seaborn-v0_8-whitegrid")
	fig, ax = plt.subplots(figsize=(11, 6.5))

	ax.plot(mu_grid, prior_pdf, color="#1f77b4", linewidth=2.8, label="Prior")
	ax.plot(
		mu_grid,
		likelihood_pdf,
		color="#ff7f0e",
		linewidth=2.8,
		label="Likelihood (as a function of mu)",
	)
	ax.plot(mu_grid, posterior_pdf, color="#2ca02c", linewidth=3.2, label="Posterior")

	ax.fill_between(mu_grid, posterior_pdf, color="#2ca02c", alpha=0.15)

	ax.set_title(
		"Gaussian Prior + Gaussian Likelihood => Gaussian Posterior",
		fontsize=14,
		pad=14,
	)

	ax.set_xlabel("Parameter mu", fontsize=12)
	ax.set_ylabel("Density", fontsize=12)
	ax.legend(loc="upper left", frameon=True)

	output_path = "conjugate_priors_plot.png"
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	fig.tight_layout()
	plt.show()
	print(f"Saved plot image to: {output_path}")


if __name__ == "__main__":
	main()
