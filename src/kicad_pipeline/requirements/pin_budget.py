"""MCU pin-budget tracker and validator.

:class:`PinBudgetTracker` is a mutable accumulator used during requirements
parsing.  Once all pins have been assigned, call :meth:`PinBudgetTracker.build`
to obtain the immutable :class:`~kicad_pipeline.models.requirements.MCUPinMap`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dc_field

from kicad_pipeline.exceptions import RequirementsError
from kicad_pipeline.models.requirements import MCUPinMap, PinAssignment, PinFunction

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mutable tracker
# ---------------------------------------------------------------------------


@dataclass
class PinBudgetTracker:
    """Mutable tracker for building MCU pin assignments.

    Use this during requirements parsing to accumulate pin assignments, then
    call :meth:`build` to obtain the immutable :class:`MCUPinMap`.

    Attributes:
        mcu_ref: Reference designator of the MCU (e.g. ``'U1'``).
        total_pins: Total number of I/O pins available on the MCU.
    """

    mcu_ref: str
    total_pins: int
    _assignments: dict[str, PinAssignment] = dc_field(default_factory=dict)

    def assign(
        self,
        pin_number: str,
        pin_name: str,
        function: PinFunction,
        net: str,
        notes: str | None = None,
    ) -> None:
        """Assign *pin_number* to *net* with the given *function*.

        Args:
            pin_number: Physical pin number or identifier (e.g. ``'GPIO4'``).
            pin_name: Logical pin name (e.g. ``'IO4'``).
            function: Logical function (:class:`~kicad_pipeline.models.requirements.PinFunction`).
            net: Net name this pin connects to (e.g. ``'SPI_CLK'``).
            notes: Optional free-text notes.

        Raises:
            RequirementsError: If *pin_number* is already assigned.
        """
        if pin_number in self._assignments:
            existing = self._assignments[pin_number]
            raise RequirementsError(
                f"Pin '{pin_number}' on {self.mcu_ref} is already assigned to net "
                f"'{existing.net}' (function {existing.function.value}); "
                f"cannot reassign to '{net}'."
            )
        assignment = PinAssignment(
            mcu_ref=self.mcu_ref,
            pin_number=pin_number,
            pin_name=pin_name,
            function=function,
            net=net,
            notes=notes,
        )
        self._assignments[pin_number] = assignment
        log.debug(
            "%s: assigned pin %s (%s) → %s [%s]",
            self.mcu_ref,
            pin_number,
            pin_name,
            net,
            function.value,
        )

    def assign_if_free(
        self,
        pin_number: str,
        pin_name: str,
        function: PinFunction,
        net: str,
        notes: str | None = None,
    ) -> bool:
        """Assign *pin_number* only if it is currently unassigned.

        Args:
            pin_number: Physical pin number or identifier.
            pin_name: Logical pin name.
            function: Logical function.
            net: Net name this pin connects to.
            notes: Optional free-text notes.

        Returns:
            ``True`` if the pin was free and has been assigned;
            ``False`` if it was already taken.
        """
        if pin_number in self._assignments:
            log.debug(
                "%s: pin %s already assigned — skipping assign_if_free for net '%s'",
                self.mcu_ref,
                pin_number,
                net,
            )
            return False
        self.assign(pin_number, pin_name, function, net, notes)
        return True

    def is_assigned(self, pin_number: str) -> bool:
        """Return ``True`` if *pin_number* has already been assigned.

        Args:
            pin_number: Physical pin number or identifier.
        """
        return pin_number in self._assignments

    def get_assignment(self, pin_number: str) -> PinAssignment | None:
        """Return the :class:`PinAssignment` for *pin_number*, or ``None``.

        Args:
            pin_number: Physical pin number or identifier.
        """
        return self._assignments.get(pin_number)

    def pins_by_function(self, function: PinFunction) -> list[PinAssignment]:
        """Return all assignments whose function matches *function*.

        Args:
            function: The :class:`~kicad_pipeline.models.requirements.PinFunction`
                      to filter by.
        """
        return [a for a in self._assignments.values() if a.function == function]

    def free_gpio_count(self) -> int:
        """Estimate the number of unassigned pins.

        Computes ``total_pins - len(assigned)``.  This is a rough estimate
        because power and ground pins are typically not tracked here.

        Returns:
            Non-negative integer count of apparently free pins.
        """
        used = len(self._assignments)
        free = max(0, self.total_pins - used)
        log.debug(
            "%s: %d / %d pins assigned; ~%d free",
            self.mcu_ref,
            used,
            self.total_pins,
            free,
        )
        return free

    def build(self) -> MCUPinMap:
        """Build and return the immutable :class:`MCUPinMap`.

        Returns:
            An :class:`MCUPinMap` containing all current assignments.
            The ``unassigned_gpio`` field is left empty — callers should
            populate it separately if needed.
        """
        assignments = tuple(self._assignments.values())
        pin_map = MCUPinMap(
            mcu_ref=self.mcu_ref,
            assignments=assignments,
            unassigned_gpio=(),
        )
        log.debug(
            "%s: built MCUPinMap with %d assignments",
            self.mcu_ref,
            len(assignments),
        )
        return pin_map


# ---------------------------------------------------------------------------
# Standalone validator
# ---------------------------------------------------------------------------


def validate_pin_map(pin_map: MCUPinMap) -> list[str]:
    """Validate an :class:`MCUPinMap` and return a list of warning strings.

    An empty list means no issues were found.

    Checks performed:

    * No two assignments share the same ``pin_number``.
    * USB pins: both ``USB_DP`` and ``USB_DM`` are present if either appears.

    Args:
        pin_map: The :class:`MCUPinMap` to validate.

    Returns:
        List of human-readable warning strings.  Empty list means valid.
    """
    warnings: list[str] = []

    # --- Duplicate pin numbers -------------------------------------------
    seen_pins: dict[str, PinAssignment] = {}
    for assignment in pin_map.assignments:
        pn = assignment.pin_number
        if pn in seen_pins:
            warnings.append(
                f"Duplicate assignment for pin '{pn}' on {pin_map.mcu_ref}: "
                f"net '{seen_pins[pn].net}' and '{assignment.net}'."
            )
        else:
            seen_pins[pn] = assignment

    # --- USB differential pair completeness -----------------------------
    usb_functions = {a.function for a in pin_map.assignments}
    has_dp = PinFunction.USB_DP in usb_functions
    has_dm = PinFunction.USB_DM in usb_functions
    if has_dp and not has_dm:
        warnings.append(
            f"{pin_map.mcu_ref}: USB_DP assigned but USB_DM is missing — "
            "USB differential pair is incomplete."
        )
    if has_dm and not has_dp:
        warnings.append(
            f"{pin_map.mcu_ref}: USB_DM assigned but USB_DP is missing — "
            "USB differential pair is incomplete."
        )

    return warnings
