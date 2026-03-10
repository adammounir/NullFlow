"""
Page-Hinkley Drift Detector for task-free continual learning.

Monitors the loss stream to detect distribution shifts (task boundaries)
without explicit task labels. When a drift is detected, the system triggers
the Sleep phases (NREM + REM) for consolidation.

The Page-Hinkley test is a sequential analysis technique that detects a
change in the average of a sequence of observations.
"""


class PageHinkleyDetector:
    """
    Page-Hinkley Test for change detection in streaming data.

    The test monitors the cumulative deviation of observations from their
    running mean. A drift is signaled when the maximum deviation exceeds
    the current deviation by more than a threshold λ.

    Statistic:
        m_t = Σ_{i=1}^{t} (x_i - x̄_t - δ)
        M_t = max_{i=1..t} m_i
        Drift detected when: M_t - m_t > λ

    Where:
        - δ (delta): tolerance for acceptable change magnitude
        - λ (threshold): detection threshold (larger = less sensitive)
        - warmup: minimum observations before detection is active
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        warmup: int = 100,
    ):
        """
        Args:
            delta: Tolerance parameter. Smaller values make the detector
                   more sensitive to small changes.
            threshold: Detection threshold λ. Larger values require
                       bigger distributional shifts to trigger.
            warmup: Minimum number of observations before detection starts.
        """
        self.delta = delta
        self.threshold = threshold
        self.warmup = warmup

        # Internal state
        self.n: int = 0          # Number of observations
        self.sum: float = 0.0    # Cumulative sum
        self.mean: float = 0.0   # Running mean
        self.m_t: float = 0.0    # Page-Hinkley statistic
        self.M_t: float = 0.0    # Minimum of m_t over time

        # History for visualization
        self.values_history: list = []
        self.statistic_history: list = []   # m_t - M_t over time
        self.drift_points: list = []        # Indices where drift was detected

    def update(self, value: float) -> bool:
        """
        Process a new observation (e.g., current batch loss).

        Args:
            value: New observation value (typically the loss).

        Returns:
            True if a distributional drift is detected, False otherwise.
        """
        self.n += 1
        self.sum += value
        self.mean = self.sum / self.n

        # Update Page-Hinkley statistic:
        # Accumulate deviation from running mean minus tolerance
        self.m_t += value - self.mean - self.delta
        # Track the minimum cumulative sum
        self.M_t = min(self.M_t, self.m_t)

        # PH statistic is the gap between current and min
        ph_stat = self.m_t - self.M_t

        # Store history for visualization
        self.values_history.append(value)
        self.statistic_history.append(ph_stat)

        # Check for drift (only after warmup period)
        if self.n > self.warmup and ph_stat > self.threshold:
            self.drift_points.append(len(self.values_history) - 1)
            self.reset()
            return True

        return False

    def reset(self):
        """
        Reset all internal statistics after a drift is detected.

        This prepares the detector for monitoring the next segment of
        the data stream.
        """
        self.n = 0
        self.sum = 0.0
        self.mean = 0.0
        self.m_t = 0.0
        self.M_t = 0.0

    @property
    def history(self) -> list:
        """Alias for values_history for backward compatibility."""
        return self.values_history

    def get_current_statistic(self) -> float:
        """
        Get the current drift statistic (m_t - M_t).

        Returns:
            Current Page-Hinkley statistic value.
        """
        return self.m_t - self.M_t

    def get_history(self) -> dict:
        """
        Get the full history for visualization.

        Returns:
            Dictionary with 'values', 'statistics', and 'drift_points'.
        """
        return {
            "values": self.values_history,
            "statistics": self.statistic_history,
            "drift_points": self.drift_points,
        }

    def reset_history(self):
        """Clear all history (for memory management)."""
        self.values_history = []
        self.statistic_history = []
        self.drift_points = []
