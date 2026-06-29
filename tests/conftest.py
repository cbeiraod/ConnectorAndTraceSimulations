import os
from hypothesis import settings, HealthCheck

# Register a "dev" profile for fast, local development (e.g., 10 examples)
settings.register_profile(
    "dev",
    max_examples=10,
    # Sometimes fuzzing generation takes a bit if assume() rejects a lot of cases
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow]
)

# Register a "ci" profile for deep stress testing in CI pipelines
settings.register_profile(
    "ci",
    max_examples=500,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow]
    # We omit Phase.shrink here. If a test fails, it reports the first failure
    # it found immediately without spending 5 minutes trying to simplify it.
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target]
)

# Automatically load the "ci" profile if the CI environment variable is set
# (GitHub Actions, GitLab CI, etc. usually set this automatically).
# Otherwise, default to the fast "dev" profile.
settings.load_profile("ci" if os.getenv("CI") else "dev")