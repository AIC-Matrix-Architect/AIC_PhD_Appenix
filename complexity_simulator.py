# File: complexity_simulator.py

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import skew, kurtosis
from itertools import combinations


class ComplexitySimulator:
    def __init__(self, entities):
        """Initialize with list of entities"""
        self.entities = entities
        self.interactions = [f"{a}-{b}" for a, b in combinations(entities, 2)]
        self.n_interactions = len(self.interactions)

    def generate_probabilities(self, dist_type, head_indices=None):
        """Generate probability distribution for given complexity type.

        Parameters
        ----------
        dist_type : str
            One of 'normal', 'longtail', or 'uniform'.
        head_indices : array-like or None
            Indices treated as head interactions for the longtail distribution.
            When None, the first 20% of interactions are used as the head.

        Returns
        -------
        np.ndarray
            Normalised probability array of length ``self.n_interactions``.
        """
        n = self.n_interactions
        probs = np.zeros(n)

        if dist_type == "normal":
            # Top 20% interactions get 80% of probability mass (Pareto principle).
            # Remaining 80% of interactions receive 20% via exponential decay.
            n_head = max(1, int(0.2 * n))
            # Assign 80% mass to the head interactions with exponential decay
            head_weights = np.exp(-np.arange(n_head))
            head_weights = head_weights / head_weights.sum() * 0.80
            probs[:n_head] = head_weights

            # Remaining interactions get 20% with steeper exponential decay
            n_tail = n - n_head
            if n_tail > 0:
                tail_weights = np.exp(-np.arange(n_tail) * 3)
                tail_weights = tail_weights / tail_weights.sum() * 0.20
                probs[n_head:] = tail_weights

        elif dist_type == "longtail":
            # Top 20% interactions get 50% of probability mass.
            # Remaining 50% distributed across the tail with power-law decay.
            n_head = max(1, int(0.2 * n))
            if head_indices is None:
                head_indices = np.arange(n_head)

            head_weights = np.exp(-np.arange(n_head))
            head_weights = head_weights / head_weights.sum() * 0.50

            tail_indices = np.array([i for i in range(n) if i not in set(head_indices)])
            n_tail = len(tail_indices)
            if n_tail > 0:
                # Power-law decay: weight ∝ (rank + 1)^(-1.5)
                tail_weights = np.array([(r + 1) ** (-1.5) for r in range(n_tail)])
                tail_weights = tail_weights / tail_weights.sum() * 0.50
            else:
                tail_weights = np.array([])

            for idx, w in zip(head_indices, head_weights):
                probs[idx] = w
            for idx, w in zip(tail_indices, tail_weights):
                probs[idx] = w

        elif dist_type == "uniform":
            probs = np.ones(n) / n

        else:
            raise ValueError(f"Unknown dist_type: {dist_type!r}")

        # Ensure the distribution sums to exactly 1 (guard against floating-point drift)
        probs = probs / probs.sum()
        return probs

    def simulate(self, dist_type, n_steps=10000):
        """Simulate interaction occurrences over time.

        For the long-tail distribution the head interactions shift every 1000
        time steps to model dynamic complexity.

        Parameters
        ----------
        dist_type : str
            One of 'normal', 'longtail', or 'uniform'.
        n_steps : int
            Number of time steps to simulate.

        Returns
        -------
        np.ndarray
            Integer array of occurrence counts per interaction.
        """
        n = self.n_interactions
        counts = np.zeros(n, dtype=int)

        if dist_type == "longtail":
            n_head = max(1, int(0.2 * n))
            all_indices = np.arange(n)
            # Simulate in 1000-step windows with shifting head
            step = 0
            while step < n_steps:
                # Re-sample head interactions for this window
                head_indices = np.random.choice(all_indices, size=n_head, replace=False)
                probs = self.generate_probabilities("longtail", head_indices=head_indices)
                window_size = min(1000, n_steps - step)
                samples = np.random.choice(n, size=window_size, p=probs)
                for s in samples:
                    counts[s] += 1
                step += window_size
        else:
            probs = self.generate_probabilities(dist_type)
            samples = np.random.choice(n, size=n_steps, p=probs)
            for s in samples:
                counts[s] += 1

        return counts

    def plot_distributions(self, normal_counts, longtail_counts, uniform_counts):
        """Plot side-by-side histograms of the three distribution types and save to PNG."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        data = [
            ("Normal", normal_counts, "steelblue"),
            ("Long-tailed", longtail_counts, "darkorange"),
            ("Uniform", uniform_counts, "seagreen"),
        ]

        for ax, (title, counts, color) in zip(axes, data):
            sorted_counts = np.sort(counts)[::-1]
            skewness = skew(counts)
            kurt = kurtosis(counts)

            ax.bar(range(len(sorted_counts)), sorted_counts, color=color, alpha=0.8)
            ax.set_title(
                f"{title} Distribution\nSkewness={skewness:.2f}, Kurtosis={kurt:.2f}",
                fontsize=12,
            )
            ax.set_xlabel("Interaction Rank")
            ax.set_ylabel("Occurrence Count")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.suptitle("Complexity Distribution Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        output_path = "complexity_distributions.png"
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Static comparison saved: {output_path}")

    def animate(self, dist_type, n_steps=10000, n_frames=100):
        """Create GIF animation of accumulating interaction occurrences.

        Each frame accumulates ``n_steps // n_frames`` time steps worth of
        interactions and updates the bar chart accordingly.

        Parameters
        ----------
        dist_type : str
            One of 'normal', 'longtail', or 'uniform'.
        n_steps : int
            Total number of time steps to simulate.
        n_frames : int
            Number of animation frames.
        """
        steps_per_frame = max(1, n_steps // n_frames)
        n = self.n_interactions
        cumulative_counts = np.zeros(n, dtype=int)

        # Pre-compute all samples so animation is deterministic
        if dist_type == "longtail":
            n_head = max(1, int(0.2 * n))
            all_indices = np.arange(n)
            all_samples = []
            step = 0
            while step < n_steps:
                head_indices = np.random.choice(all_indices, size=n_head, replace=False)
                probs = self.generate_probabilities("longtail", head_indices=head_indices)
                window_size = min(1000, n_steps - step)
                window_samples = np.random.choice(n, size=window_size, p=probs)
                all_samples.extend(window_samples.tolist())
                step += window_size
        else:
            probs = self.generate_probabilities(dist_type)
            all_samples = np.random.choice(n, size=n_steps, p=probs).tolist()

        # Determine sort order from final counts for consistent x-axis
        temp_counts = np.zeros(n, dtype=int)
        for s in all_samples:
            temp_counts[s] += 1
        sort_order = np.argsort(temp_counts)[::-1]
        sorted_labels = [self.interactions[i] for i in sort_order]

        fig, ax = plt.subplots(figsize=(max(10, n * 0.7), 5))

        bar_container = ax.bar(range(n), np.zeros(n), color="steelblue")
        ax.set_xticks(range(n))
        ax.set_xticklabels(sorted_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Interaction Pair")
        ax.set_ylabel("Cumulative Count")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        title_obj = ax.set_title("")

        # Colour thresholds will be set dynamically each frame
        def bar_colors(counts_sorted):
            max_c = counts_sorted.max() if counts_sorted.max() > 0 else 1
            colors = []
            for c in counts_sorted:
                ratio = c / max_c
                if ratio >= 0.66:
                    colors.append("red")
                elif ratio >= 0.33:
                    colors.append("yellow")
                else:
                    colors.append("blue")
            return colors

        dist_labels = {"normal": "Normal", "longtail": "Long-tailed", "uniform": "Uniform"}
        dist_label = dist_labels[dist_type]

        # Frame update function
        sample_idx = 0

        def update(frame):
            nonlocal sample_idx
            start = sample_idx
            end = min(start + steps_per_frame, len(all_samples))
            for s in all_samples[start:end]:
                cumulative_counts[s] += 1
            sample_idx = end

            counts_sorted = cumulative_counts[sort_order]
            colors = bar_colors(counts_sorted)
            for bar, h, c in zip(bar_container, counts_sorted, colors):
                bar.set_height(h)
                bar.set_color(c)

            current_step = min(end, n_steps)
            s_val = skew(cumulative_counts) if cumulative_counts.sum() > 0 else 0.0
            k_val = kurtosis(cumulative_counts) if cumulative_counts.sum() > 0 else 0.0
            title_obj.set_text(
                f"{dist_label} Distribution | Step {current_step}/{n_steps} | "
                f"Skewness={s_val:.2f}, Kurtosis={k_val:.2f}"
            )
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            return list(bar_container) + [title_obj]

        anim = FuncAnimation(fig, update, frames=n_frames, blit=False, repeat=False)
        output_path = f"complexity_animation_{dist_type}.gif"
        writer = PillowWriter(fps=10)
        anim.save(output_path, writer=writer)
        plt.close(fig)
        print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw = input("Enter entities (comma-separated): ")
    entities = [e.strip() for e in raw.split(",") if e.strip()]

    simulator = ComplexitySimulator(entities)

    print("\nSimulating Normal Distribution...")
    normal_counts = simulator.simulate("normal", n_steps=10000)
    print(f"  Skewness: {skew(normal_counts):.2f}")
    print(f"  Excess Kurtosis: {kurtosis(normal_counts):.2f}")
    simulator.animate("normal", n_steps=10000, n_frames=100)

    print("\nSimulating Long-Tailed Distribution...")
    longtail_counts = simulator.simulate("longtail", n_steps=10000)
    print(f"  Skewness: {skew(longtail_counts):.2f}")
    print(f"  Excess Kurtosis: {kurtosis(longtail_counts):.2f}")
    simulator.animate("longtail", n_steps=10000, n_frames=100)

    print("\nSimulating Uniform Distribution...")
    uniform_counts = simulator.simulate("uniform", n_steps=10000)
    print(f"  Skewness: {skew(uniform_counts):.2f}")
    print(f"  Excess Kurtosis: {kurtosis(uniform_counts):.2f}")
    simulator.animate("uniform", n_steps=10000, n_frames=100)

    simulator.plot_distributions(normal_counts, longtail_counts, uniform_counts)
