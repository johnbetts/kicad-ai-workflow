#!/usr/bin/env python3
"""Generate the ADS1115 4-channel ADC schematic for testing."""
from __future__ import annotations

from pathlib import Path

from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinFunction,
    PinType,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.project_file import write_project_file
from kicad_pipeline.schematic.builder import build_schematic, write_schematic


def _make_passive_2pin(ref: str, value: str, fp: str = "R_0805") -> Component:
    """Create a 2-pin passive component (resistor, capacitor)."""
    return Component(
        ref=ref, value=value, footprint=fp,
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE),
        ),
    )


def _make_screw_terminal(ref: str) -> Component:
    """Create a 2-pin screw terminal connector."""
    return Component(
        ref=ref, value="Screw_Terminal_01x02", footprint="Conn_01x02",
        pins=(
            Pin(number="1", name="SIG", pin_type=PinType.PASSIVE),
            Pin(number="2", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
        ),
    )


def build_requirements() -> ProjectRequirements:
    """Build complete requirements for the 4-channel ADC design.

    Circuit:
    - 4 sensor channels: screw terminal → 100k/20k voltage divider → 100nF filter → ADS1115 AINx
    - ADS1115 16-bit I2C ADC, powered by +5V
    - DIP switch for I2C address selection (ADDR → GND/VDD/SDA/SCL)
    - Raspberry Pi 2x20 header for I2C + power
    - Bypass caps: 100nF + 10uF on +5V rail
    """
    # ----------------------------------------------------------------
    # Sensor channels: J + R_hi + R_lo + C per channel
    # ----------------------------------------------------------------
    sensor_comps: list[Component] = []
    sensor_nets: list[Net] = []
    gnd_connections: list[NetConnection] = []
    sensor_refs: list[str] = []

    r_num = 1
    c_num = 1
    for ch in range(4):
        j_ref = f"J{ch + 1}"
        r_hi = f"R{r_num}"
        r_lo = f"R{r_num + 1}"
        c_ref = f"C{c_num}"
        r_num += 2
        c_num += 1

        ain = f"AIN{ch}"
        sens = f"SENS{ch}"

        sensor_comps.extend([
            _make_screw_terminal(j_ref),
            _make_passive_2pin(r_hi, "100k"),
            _make_passive_2pin(r_lo, "20k"),
            _make_passive_2pin(c_ref, "100nF", "C_0805"),
        ])
        sensor_refs.extend([j_ref, r_hi, r_lo, c_ref])

        # Signal chain: J.SIG → R_hi.1, R_hi.2 → R_lo.1 → C.1 (= AIN node)
        sensor_nets.append(Net(
            name=sens,
            connections=(
                NetConnection(ref=j_ref, pin="1"),
                NetConnection(ref=r_hi, pin="1"),
            ),
        ))
        sensor_nets.append(Net(
            name=ain,
            connections=(
                NetConnection(ref=r_hi, pin="2"),
                NetConnection(ref=r_lo, pin="1"),
                NetConnection(ref=c_ref, pin="1"),
            ),
        ))

        # GND connections for this channel
        gnd_connections.extend([
            NetConnection(ref=j_ref, pin="2"),
            NetConnection(ref=r_lo, pin="2"),
            NetConnection(ref=c_ref, pin="2"),
        ])

    # ----------------------------------------------------------------
    # ADS1115 ADC (powered by +5V, not +3V3)
    # ----------------------------------------------------------------
    u1 = Component(
        ref="U1", value="ADS1115", footprint="MSOP-10", lcsc="C37593",
        pins=(
            Pin(number="1", name="ADDR", pin_type=PinType.INPUT),
            Pin(number="2", name="ALERT/RDY", pin_type=PinType.OUTPUT),
            Pin(number="3", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
            Pin(number="4", name="AIN0", pin_type=PinType.INPUT, function=PinFunction.ANALOG_IN),
            Pin(number="5", name="AIN1", pin_type=PinType.INPUT, function=PinFunction.ANALOG_IN),
            Pin(number="6", name="AIN2", pin_type=PinType.INPUT, function=PinFunction.ANALOG_IN),
            Pin(number="7", name="AIN3", pin_type=PinType.INPUT, function=PinFunction.ANALOG_IN),
            Pin(number="8", name="SDA", pin_type=PinType.BIDIRECTIONAL, function=PinFunction.I2C_SDA),
            Pin(number="9", name="SCL", pin_type=PinType.INPUT, function=PinFunction.I2C_SCL),
            Pin(number="10", name="VDD", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
        ),
    )

    # ----------------------------------------------------------------
    # DIP Switch (4-position) for I2C address selection
    # Pins 1-4: switch inputs (all connected to ADDR)
    # Pins 5-8: switch outputs (COM1=GND, COM2=VDD, COM3=SDA, COM4=SCL)
    # Only one switch should be ON to select address:
    #   SW1 ON → ADDR=GND  → 0x48
    #   SW2 ON → ADDR=VDD  → 0x49
    #   SW3 ON → ADDR=SDA  → 0x4A
    #   SW4 ON → ADDR=SCL  → 0x4B
    # ----------------------------------------------------------------
    sw1 = Component(
        ref="SW1", value="DIP_Switch_x04", footprint="DIP_Switch_x04",
        pins=(
            Pin(number="1", name="ADDR", pin_type=PinType.PASSIVE),
            Pin(number="2", name="ADDR", pin_type=PinType.PASSIVE),
            Pin(number="3", name="ADDR", pin_type=PinType.PASSIVE),
            Pin(number="4", name="ADDR", pin_type=PinType.PASSIVE),
            Pin(number="5", name="GND", pin_type=PinType.PASSIVE),
            Pin(number="6", name="VDD", pin_type=PinType.PASSIVE),
            Pin(number="7", name="SDA", pin_type=PinType.PASSIVE),
            Pin(number="8", name="SCL", pin_type=PinType.PASSIVE),
        ),
    )

    # ----------------------------------------------------------------
    # Raspberry Pi 2x20 stacking header
    # ----------------------------------------------------------------
    pi_pin_defs = [
        # Odd pins (left side)
        ("1", "3V3", PinType.POWER_OUT, PinFunction.VCC),
        ("3", "GPIO2_SDA1", PinType.BIDIRECTIONAL, PinFunction.I2C_SDA),
        ("5", "GPIO3_SCL1", PinType.BIDIRECTIONAL, PinFunction.I2C_SCL),
        ("7", "GPIO_PT7", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("9", "GND", PinType.POWER_IN, PinFunction.GND),
        ("11", "GPIO17", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("13", "GPIO_PT13", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("15", "GPIO_PT15", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("17", "GPIO_PT17", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("19", "GPIO_PT19", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("21", "GPIO_PT21", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("23", "GPIO_PT23", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("25", "GND", PinType.POWER_IN, PinFunction.GND),
        ("27", "GPIO_PT27", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("29", "GPIO_PT29", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("31", "GPIO_PT31", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("33", "GPIO_PT33", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("35", "GPIO_PT35", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("37", "GPIO_PT37", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("39", "GND", PinType.POWER_IN, PinFunction.GND),
        # Even pins (right side)
        ("2", "5V", PinType.POWER_OUT, PinFunction.VCC),
        ("4", "5V", PinType.POWER_OUT, PinFunction.VCC),
        ("6", "GND", PinType.POWER_IN, PinFunction.GND),
        ("8", "GPIO_PT8", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("10", "GPIO_PT10", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("12", "GPIO_PT12", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("14", "GND", PinType.POWER_IN, PinFunction.GND),
        ("16", "GPIO_PT16", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("18", "GPIO_PT18", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("20", "GND", PinType.POWER_IN, PinFunction.GND),
        ("22", "GPIO_PT22", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("24", "GPIO_PT24", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("26", "GPIO_PT26", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("28", "GND", PinType.POWER_IN, PinFunction.GND),
        ("30", "GPIO_PT28", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("32", "GPIO_PT32", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("34", "GND", PinType.POWER_IN, PinFunction.GND),
        ("36", "GPIO_PT36", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("38", "GPIO_PT38", PinType.BIDIRECTIONAL, PinFunction.GPIO),
        ("40", "GPIO_PT40", PinType.BIDIRECTIONAL, PinFunction.GPIO),
    ]
    j5 = Component(
        ref="J5", value="Conn_02x20_Stacking", footprint="Conn_02x20_Stacking",
        pins=tuple(
            Pin(number=num, name=name, pin_type=ptype, function=func)
            for num, name, ptype, func in pi_pin_defs
        ),
    )

    # Bypass capacitors
    c5 = _make_passive_2pin("C5", "100nF", "C_0805")
    c6 = _make_passive_2pin("C6", "10uF", "C_0805")

    # ----------------------------------------------------------------
    # All components (no R9 — DIP switch handles address selection)
    # ----------------------------------------------------------------
    all_comps = tuple(sensor_comps + [u1, sw1, j5, c5, c6])

    # ----------------------------------------------------------------
    # Nets
    # ----------------------------------------------------------------
    all_nets: list[Net] = list(sensor_nets)

    # ADC AIN connections (merge with existing AINx divider nets)
    for ch in range(4):
        ain = f"AIN{ch}"
        for idx, existing in enumerate(all_nets):
            if existing.name == ain:
                all_nets[idx] = Net(
                    name=ain,
                    connections=existing.connections + (
                        NetConnection(ref="U1", pin=str(4 + ch)),
                    ),
                )
                break

    # I2C bus
    all_nets.append(Net(name="I2C_SDA", connections=(
        NetConnection(ref="U1", pin="8"),
        NetConnection(ref="J5", pin="3"),   # Pi GPIO2_SDA1
        NetConnection(ref="SW1", pin="7"),  # DIP switch SDA option
    )))
    all_nets.append(Net(name="I2C_SCL", connections=(
        NetConnection(ref="U1", pin="9"),
        NetConnection(ref="J5", pin="5"),   # Pi GPIO3_SCL1
        NetConnection(ref="SW1", pin="8"),  # DIP switch SCL option
    )))

    # ADDR net — DIP switch inputs (pins 1-4) all connect to ADDR
    all_nets.append(Net(name="ADDR", connections=(
        NetConnection(ref="U1", pin="1"),
        NetConnection(ref="SW1", pin="1"),
        NetConnection(ref="SW1", pin="2"),
        NetConnection(ref="SW1", pin="3"),
        NetConnection(ref="SW1", pin="4"),
    )))

    # ALERT/RDY net
    all_nets.append(Net(name="ALERT", connections=(
        NetConnection(ref="U1", pin="2"),
        NetConnection(ref="J5", pin="11"),  # Pi GPIO17
    )))

    # +5V power — both Pi 5V pins + ADS1115 VDD + bypass caps + DIP VDD option
    all_nets.append(Net(name="+5V", connections=(
        NetConnection(ref="J5", pin="2"),   # Pi 5V pin 2
        NetConnection(ref="J5", pin="4"),   # Pi 5V pin 4
        NetConnection(ref="U1", pin="10"),  # ADS1115 VDD
        NetConnection(ref="SW1", pin="6"),  # DIP switch VDD option
        NetConnection(ref="C5", pin="1"),   # bypass cap
        NetConnection(ref="C6", pin="1"),   # bulk cap
    )))

    # GND — all ground connections
    gnd_connections.extend([
        NetConnection(ref="U1", pin="3"),   # ADS1115 GND
        NetConnection(ref="SW1", pin="5"),  # DIP switch GND option
        NetConnection(ref="J5", pin="9"),   # Pi GND pins
        NetConnection(ref="J5", pin="25"),
        NetConnection(ref="J5", pin="39"),
        NetConnection(ref="J5", pin="6"),
        NetConnection(ref="J5", pin="14"),
        NetConnection(ref="J5", pin="20"),
        NetConnection(ref="J5", pin="28"),
        NetConnection(ref="J5", pin="34"),
        NetConnection(ref="C5", pin="2"),   # bypass cap GND
        NetConnection(ref="C6", pin="2"),   # bulk cap GND
    ])
    all_nets.append(Net(name="GND", connections=tuple(gnd_connections)))

    # ----------------------------------------------------------------
    # Feature blocks
    # ----------------------------------------------------------------
    features = (
        FeatureBlock(
            name="Analog Sensors",
            description="4-channel voltage divider sensor inputs",
            components=tuple(sensor_refs),
            nets=("SENS0", "SENS1", "SENS2", "SENS3",
                  "AIN0", "AIN1", "AIN2", "AIN3"),
            subcircuits=(),
        ),
        FeatureBlock(
            name="ADC Core",
            description="ADS1115 16-bit ADC with DIP switch address select",
            components=("U1", "SW1"),
            nets=("I2C_SDA", "I2C_SCL", "ADDR", "ALERT"),
            subcircuits=(),
        ),
        FeatureBlock(
            name="Connector Interface",
            description="Raspberry Pi 2x20 stacking header",
            components=("J5",),
            nets=(),
            subcircuits=(),
        ),
        FeatureBlock(
            name="Power Supply",
            description="Bypass and bulk capacitors",
            components=("C5", "C6"),
            nets=("+5V",),
            subcircuits=(),
        ),
    )

    return ProjectRequirements(
        project=ProjectInfo(
            name="ADS1115_4CH_ADC",
            author="kicad-ai-pipeline",
            revision="v0.2",
            description="4-channel voltage divider ADC with Raspberry Pi interface",
        ),
        features=features,
        components=all_comps,
        nets=tuple(all_nets),
        mechanical=MechanicalConstraints(board_width_mm=80.0, board_height_mm=60.0),
        power_budget=PowerBudget(
            rails=(
                PowerRail(name="+5V", voltage=5.0, current_ma=500.0, source_ref="J5"),
            ),
            total_current_ma=500.0,
            notes=("Power from Pi header 5V pins",),
        ),
    )


def main() -> None:
    """Generate schematic and write to output directory."""
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    req = build_requirements()
    sch = build_schematic(req)

    sch_path = out_dir / "ADS1115_4CH_ADC.kicad_sch"
    write_schematic(sch, str(sch_path))
    print(f"Schematic written to {sch_path}")

    write_project_file("ADS1115_4CH_ADC", out_dir)
    print(f"Project file written to {out_dir / 'ADS1115_4CH_ADC.kicad_pro'}")


if __name__ == "__main__":
    main()
