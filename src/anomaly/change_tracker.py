"""Change Persistence Tracker - Track price changes across scrape runs.

Tracks whether price changes have "persisted" (remained stable across multiple
scrape runs). This is used to filter out transient anomalies.

Key insight: Real price changes tend to persist. Scraper bugs often cause
one-off spikes that revert in the next run.

Usage:
    tracker = ChangePersistenceTracker()

    # On each scrape run
    for record in records:
        persistence_info = tracker.update(record)
        if persistence_info.persisted_runs >= 2:
            # Change has been stable - more likely real
            pass

Storage: In-memory state for local execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PriceChange:
    """Tracks a single price change event."""

    competitor_product_id: str
    competitor: str
    old_price: float | None
    new_price: float
    change_pct: float | None
    first_seen_at: datetime
    last_seen_at: datetime
    persisted_runs: int
    run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "competitor_product_id": self.competitor_product_id,
            "competitor": self.competitor,
            "old_price": self.old_price,
            "new_price": self.new_price,
            "change_pct": self.change_pct,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "persisted_runs": self.persisted_runs,
            "run_ids": self.run_ids[-5:],  # Keep last 5 run IDs
        }


@dataclass
class PersistenceInfo:
    """Information about price persistence for a product."""

    competitor_product_id: str
    competitor: str

    # Current price info
    current_price: float
    previous_price: float | None

    # Persistence tracking
    price_changed: bool
    persisted_runs: int  # How many consecutive runs with this price
    first_seen_at: datetime | None  # When this price was first observed
    is_new_change: bool  # True if this is the first observation of this price

    # Stability assessment
    is_stable: bool  # True if persisted >= threshold
    stability_threshold: int  # Threshold used

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "competitor_product_id": self.competitor_product_id,
            "competitor": self.competitor,
            "current_price": self.current_price,
            "previous_price": self.previous_price,
            "price_changed": self.price_changed,
            "persisted_runs": self.persisted_runs,
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "is_new_change": self.is_new_change,
            "is_stable": self.is_stable,
            "stability_threshold": self.stability_threshold,
        }


@dataclass
class ProductPriceState:
    """Internal state for tracking a product's price history."""

    competitor_product_id: str
    competitor: str
    current_price: float
    previous_price: float | None
    current_price_first_seen: datetime
    current_price_run_count: int
    last_run_id: str | None = None


class ChangePersistenceTracker:
    """Track price changes across scrape runs.

    Determines whether price changes have "persisted" (remained stable)
    across multiple scrape runs.

    Usage:
        tracker = ChangePersistenceTracker(persistence_threshold=2)

        # Process each record
        for record in records:
            info = tracker.update(
                competitor_product_id=record.competitor_product_id,
                competitor=record.competitor,
                price=record.price,
                run_id=run_health.run_id,
                timestamp=record.scraped_at,
            )

            if info.is_stable:
                # Price has persisted - more likely real change
                pass
            elif info.is_new_change:
                # New change - wait for persistence
                pass
    """

    def __init__(
        self,
        persistence_threshold: int = 2,
        price_change_threshold: float = 0.01,  # 1% to consider as changed
    ):
        """Initialize the tracker.

        Args:
            persistence_threshold: Number of runs for a change to be "stable".
            price_change_threshold: Minimum % change to consider price as changed.
        """
        self.persistence_threshold = persistence_threshold
        self.price_change_threshold = price_change_threshold

        # State: {(competitor_product_id, competitor): ProductPriceState}
        self._state: dict[tuple[str, str], ProductPriceState] = {}

    def update(
        self,
        competitor_product_id: str,
        competitor: str,
        price: float,
        run_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> PersistenceInfo:
        """Update tracking for a product and return persistence info.

        Args:
            competitor_product_id: Product identifier.
            competitor: Competitor identifier.
            price: Current price.
            run_id: Optional run identifier.
            timestamp: Optional timestamp.

        Returns:
            PersistenceInfo with current persistence status.
        """
        key = (competitor_product_id, competitor)
        now = timestamp or datetime.now(timezone.utc)

        if key not in self._state:
            # First observation - initialize state
            self._state[key] = ProductPriceState(
                competitor_product_id=competitor_product_id,
                competitor=competitor,
                current_price=price,
                previous_price=None,
                current_price_first_seen=now,
                current_price_run_count=1,
                last_run_id=run_id,
            )

            return PersistenceInfo(
                competitor_product_id=competitor_product_id,
                competitor=competitor,
                current_price=price,
                previous_price=None,
                price_changed=False,  # No previous price to compare
                persisted_runs=1,
                first_seen_at=now,
                is_new_change=False,
                is_stable=False,  # Not stable until threshold met
                stability_threshold=self.persistence_threshold,
            )

        state = self._state[key]

        # Skip duplicate updates from same run
        if run_id and run_id == state.last_run_id:
            return PersistenceInfo(
                competitor_product_id=competitor_product_id,
                competitor=competitor,
                current_price=price,
                previous_price=state.previous_price,
                price_changed=False,
                persisted_runs=state.current_price_run_count,
                first_seen_at=state.current_price_first_seen,
                is_new_change=False,
                is_stable=state.current_price_run_count >= self.persistence_threshold,
                stability_threshold=self.persistence_threshold,
            )

        # Check if price changed
        price_changed = self._is_price_changed(state.current_price, price)

        if price_changed:
            # Price changed - reset persistence counter
            previous_price = state.current_price
            state.previous_price = previous_price
            state.current_price = price
            state.current_price_first_seen = now
            state.current_price_run_count = 1
            state.last_run_id = run_id

            return PersistenceInfo(
                competitor_product_id=competitor_product_id,
                competitor=competitor,
                current_price=price,
                previous_price=previous_price,
                price_changed=True,
                persisted_runs=1,
                first_seen_at=now,
                is_new_change=True,
                is_stable=False,
                stability_threshold=self.persistence_threshold,
            )
        else:
            # Price unchanged - increment persistence counter
            state.current_price_run_count += 1
            state.last_run_id = run_id

            return PersistenceInfo(
                competitor_product_id=competitor_product_id,
                competitor=competitor,
                current_price=price,
                previous_price=state.previous_price,
                price_changed=False,
                persisted_runs=state.current_price_run_count,
                first_seen_at=state.current_price_first_seen,
                is_new_change=False,
                is_stable=state.current_price_run_count >= self.persistence_threshold,
                stability_threshold=self.persistence_threshold,
            )

    def _is_price_changed(self, old_price: float, new_price: float) -> bool:
        """Check if price has meaningfully changed."""
        if old_price == 0:
            return new_price != 0
        change_pct = abs(new_price - old_price) / old_price
        return change_pct > self.price_change_threshold

    def get_persistence(
        self, competitor_product_id: str, competitor: str
    ) -> PersistenceInfo | None:
        """Get current persistence info for a product.

        Args:
            competitor_product_id: Product identifier.
            competitor: Competitor identifier.

        Returns:
            PersistenceInfo if product is tracked, None otherwise.
        """
        key = (competitor_product_id, competitor)
        state = self._state.get(key)

        if state is None:
            return None

        return PersistenceInfo(
            competitor_product_id=competitor_product_id,
            competitor=competitor,
            current_price=state.current_price,
            previous_price=state.previous_price,
            price_changed=False,  # Current state
            persisted_runs=state.current_price_run_count,
            first_seen_at=state.current_price_first_seen,
            is_new_change=False,
            is_stable=state.current_price_run_count >= self.persistence_threshold,
            stability_threshold=self.persistence_threshold,
        )

    def get_unstable_products(self) -> list[PersistenceInfo]:
        """Get all products with recent unstable price changes.

        Returns:
            List of PersistenceInfo for products below stability threshold.
        """
        unstable = []
        for key, state in self._state.items():
            if state.current_price_run_count < self.persistence_threshold:
                info = PersistenceInfo(
                    competitor_product_id=state.competitor_product_id,
                    competitor=state.competitor,
                    current_price=state.current_price,
                    previous_price=state.previous_price,
                    price_changed=True,  # It changed recently
                    persisted_runs=state.current_price_run_count,
                    first_seen_at=state.current_price_first_seen,
                    is_new_change=state.current_price_run_count == 1,
                    is_stable=False,
                    stability_threshold=self.persistence_threshold,
                )
                unstable.append(info)
        return unstable

    def get_stats(self) -> dict[str, Any]:
        """Get tracking statistics.

        Returns:
            Dictionary with tracking stats.
        """
        total = len(self._state)
        stable = sum(
            1
            for s in self._state.values()
            if s.current_price_run_count >= self.persistence_threshold
        )
        unstable = total - stable

        return {
            "total_products_tracked": total,
            "stable_products": stable,
            "unstable_products": unstable,
            "persistence_threshold": self.persistence_threshold,
        }

    def clear(self) -> None:
        """Clear all tracking state."""
        self._state.clear()

    def remove_product(self, competitor_product_id: str, competitor: str) -> bool:
        """Remove a product from tracking.

        Args:
            competitor_product_id: Product identifier.
            competitor: Competitor identifier.

        Returns:
            True if product was removed, False if not found.
        """
        key = (competitor_product_id, competitor)
        if key in self._state:
            del self._state[key]
            return True
        return False
