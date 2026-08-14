"""Single source of truth for hardware parameters, with provenance.

===========================================================================
 @tandonmitul27  --  AUTHORED FILE (new)
===========================================================================

WHY THIS FILE EXISTS
    A memory-system study is only as trustworthy as the numbers fed into
    it, and those numbers otherwise scatter across simulator configs,
    shell commands and prose.  This module is the one place every
    hardware constant lives, and each carries a `source` and a
    `confidence` so a reader can tell -- without asking anyone -- which
    figures come from a standard, which from a vendor datasheet, which
    we measured, and which are still estimates.

        python sim/system_params.py     # prints the whole table

    The ESTIMATE entries are deliberately collected at the bottom, in
    NEEDS_SOURCE, so an unsourced number cannot quietly reach a result.
===========================================================================

System modelled (per project clarification): the **GPU is the CXL host**.

    GPU  ---- on-package ----  HBM        local memory
     |
     +---- CXL 2.0/3.0 link ---- CXL memory expander

One hop, not two.  Note this differs from shipping hardware, where CXL attaches
to the CPU and reaching the GPU costs a second PCIe traversal; GPU-side CXL is
emerging rather than deployed.  D1 must state which is assumed -- ours is the
forward-looking single-hop case, which is the optimistic one for CXL.

Every entry carries `source` and `confidence`:

    spec       arithmetic from a published standard (PCIe/CXL/JEDEC); exact
    datasheet  vendor-published figure
    measured   measured by us in this harness (see docs/VALIDATION.md)
    derived    computed from the above; formula given
    ESTIMATE   not sourced -- must be verified before use in the report

Nothing here is a magic constant: latency, bandwidth and capacity are all
sweep axes.  These are the anchor points the sweeps are centred on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Param:
    value: float
    unit: str
    source: str
    confidence: str
    note: str = ""


# ==========================================================================
# 1. PCIe / CXL link rates -- spec arithmetic, exact
# ==========================================================================
# PCIe 5.0: 32 GT/s/lane, 128b/130b encoding
#     per lane per direction = 32 * 128/130 / 8 = 3.938 GB/s
# PCIe 6.0: 64 GT/s/lane, PAM4 + FLIT/FEC
#     raw 64/8 = 8 GB/s per lane; FLIT overhead ~5% -> ~7.56 GB/s
# CXL 1.1/2.0 run on PCIe 5.0 electricals; CXL 3.0 on PCIe 6.0.

def _pcie5(lanes: int) -> float:
    return 32.0 * (128 / 130) / 8 * lanes


def _pcie6(lanes: int) -> float:
    return 64.0 / 8 * 0.945 * lanes          # FLIT/FEC overhead


LINK = {
    "cxl2_x4":  Param(_pcie5(4),  "GB/s", "spec: PCIe 5.0 128b/130b", "spec"),
    "cxl2_x8":  Param(_pcie5(8),  "GB/s", "spec: PCIe 5.0 128b/130b", "spec"),
    "cxl2_x16": Param(_pcie5(16), "GB/s", "spec: PCIe 5.0 128b/130b", "spec"),
    "cxl3_x8":  Param(_pcie6(8),  "GB/s", "spec: PCIe 6.0 PAM4+FLIT", "spec"),
    "cxl3_x16": Param(_pcie6(16), "GB/s", "spec: PCIe 6.0 PAM4+FLIT", "spec"),
}

# What we actually achieve on those links, measured in this harness
# (protocol path + expander backend included).
LINK_EFFICIENCY = {
    "pcie4_x16_26": Param(
        23.5, "GB/s", "measured: FloE check, 90%", "measured"),
    "cxl2_x16_63": Param(
        50.7, "GB/s", "measured: DDR5-6400 backend, 80%", "measured",
        "CXL-DMSim reports 82-83% of local DDR bandwidth for its ASIC "
        "part; different denominator, similar figure"),
    "cxl3_x16_121": Param(
        105.6, "GB/s", "measured with 128-entry device FIFOs, 87%",
        "measured",
        "REQUIRES deeper device buffering than the silicon-validated "
        "CXL 2.0 ASIC's 48 entries: 121 GB/s x ~280 ns RTT / 64 B ~ 530 "
        "outstanding; with 48-entry FIFOs the link caps at ~86 GB/s "
        "(71%). No shipping CXL 3.0 expander exists to validate against; "
        "state the buffering assumption whenever this point is used. "
        "PATTERN-DEPENDENT: this 105.6 figure is a steady-state mixed "
        "stream; long warm sequential streams reach ~113 GB/s (93%), so "
        "treating 105.6 as the rate is conservative by up to ~13%."),
}


# ==========================================================================
# 2. GPU HBM -- vendor datasheets
# ==========================================================================
# Capacity and aggregate bandwidth are datasheet figures. Per-stack bandwidth
# is derived (aggregate / stacks) and is BELOW the JEDEC maximum for the
# generation, because shipping parts clock HBM under its top bin.

GPU = {
    "H100_SXM": dict(
        hbm_gen="HBM3",
        capacity=Param(80, "GB", "datasheet: NVIDIA H100 SXM5", "datasheet"),
        bandwidth=Param(3350, "GB/s", "datasheet: NVIDIA H100 SXM5", "datasheet"),
        stacks=Param(5, "count", "datasheet: 5 HBM3 stacks", "datasheet"),
    ),
    "H200_SXM": dict(
        hbm_gen="HBM3e",
        capacity=Param(141, "GB", "datasheet: NVIDIA H200 SXM", "datasheet"),
        bandwidth=Param(4800, "GB/s", "datasheet: NVIDIA H200 SXM", "datasheet"),
        stacks=Param(6, "count", "datasheet: 6 HBM3e stacks", "datasheet"),
    ),
    "B200": dict(
        hbm_gen="HBM3e",
        capacity=Param(192, "GB", "datasheet: NVIDIA B200", "datasheet"),
        bandwidth=Param(8000, "GB/s", "datasheet: NVIDIA B200", "datasheet"),
        stacks=Param(8, "count", "datasheet: 8 HBM3e stacks", "datasheet"),
    ),
}


def gpu_per_stack_gbps(name: str) -> float:
    """Derived: datasheet aggregate / stack count."""
    g = GPU[name]
    return g["bandwidth"].value / g["stacks"].value


# ==========================================================================
# 3. HBM device models -- JESD238A (we hold the standard: papers/JESD238A.pdf)
# ==========================================================================
# What JESD238A FIXES (directly citable):
#   * geometry & addressing        Table 4  (densities/channel, 256b prefetch/PC)
#   * voltages                     Table 70 (VDDC=VDDQ=1.1 V, VDDQL=0.4 V,
#                                            VPP=1.8 V)
#   * speed bins                   Table 92 (4.8-6.4 Gbps/pin; fCK max 1.6 GHz)
#   * array timings scoped PER     Table 3 / s3.1.2 -- the citation for halving
#     PSEUDO CHANNEL                tFAW/tRRD when modelling at channel level
#   * IDD state definitions and    s9, Tables 83-89 (Table 90, the value table,
#     measurement loop patterns     is an EMPTY TEMPLATE by design)
#
# What JESD238A deliberately leaves to VENDOR datasheets:
#   * array timing values (tRAS, tRCD, tRC, tRFC...) -- "tRAS is the analog
#     value from the vendor datasheet" (s5, note to RAS MR field)
#   * IDD current values
#
# Consequently: timing values come from Ramulator2's HBM3_6400Mbps preset
# (vendor-representative; the spec's own worked example uses tRAS=33 ns vs our
# 28.1 ns), and IDD currents are calibrated to published pJ/bit using the
# spec's state definitions.  This is the maximum provenance available without
# a vendor NDA.
#
# DRAMsim3 clock-domain note: JESD238A tCK is the command clock (0.625 ns at
# 6.4 Gbps); our DRAMsim3 tCK=0.3125 ns is the data-strobe domain (2x fCK),
# so cycle counts differ from JEDEC's but nanosecond values match.

HBM = {
    "HBM3": dict(
        config="HBM3_16Gb_x64",
        channels=Param(16, "count", "spec: JESD238A (papers/JESD238A.pdf)", "spec"),
        channel_width=Param(64, "bit", "spec: JESD238", "spec"),
        banks_per_channel=Param(32, "count",
                        "spec: JESD238A s3.1.2, 2 pch x 4 BG x 4 banks",
                        "spec"),
        data_rate=Param(6.4, "Gbps/pin", "spec: JESD238A Table 92 top bin",
                        "spec"),
        peak=Param(819.2, "GB/s/stack", "derived: 64b x 6.4Gbps x 16 / 8",
                   "derived"),
        achieved=Param(707.8, "GB/s/stack", "measured: docs/CALIBRATION.md", "measured",
                       "86% of peak; H100 datasheet implies 670 GB/s/stack"),
        capacity=Param(16, "GB/stack", "spec: 8-high, 16Gb dies", "spec"),
    ),
    "HBM3e": dict(
        config="HBM3e_24Gb_x64",
        channels=Param(16, "count", "spec: JESD238", "spec"),
        channel_width=Param(64, "bit", "spec: JESD238", "spec"),
        banks_per_channel=Param(32, "count", "spec: JESD238", "spec"),
        data_rate=Param(9.6, "Gbps/pin", "spec: HBM3E top bin", "spec"),
        peak=Param(1228.8, "GB/s/stack", "derived: 64b x 9.6Gbps x 16 / 8",
                   "derived"),
        achieved=Param(1027.3, "GB/s/stack", "measured: docs/CALIBRATION.md",
                       "measured", "84% of peak"),
        capacity=Param(24, "GB/stack", "spec: 8-high, 24Gb dies", "spec"),
    ),
}


# ==========================================================================
# 4. CXL protocol path -- CXL-DMSim, calibrated against real silicon
# ==========================================================================
# CXL-DMSim (arXiv:2411.02282) Table III, ASIC configuration. Their device
# model was validated against CXL FPGA and ASIC hardware at ~3.4% mean error;
# our reproduction matches their measured added latency within 5%.

CXL_PATH = dict(
    bridge_lat=Param(50, "ns", "CXL-DMSim Table III", "datasheet"),
    host_proto_lat=Param(12, "ns", "CXL-DMSim Table III (ASIC)", "datasheet"),
    dev_proto_lat=Param(15, "ns", "CXL-DMSim Table III (ASIC)", "datasheet"),
    dev_proto_lat_fpga=Param(60, "ns", "CXL-DMSim Table III (FPGA)", "datasheet"),
    link_fifo=Param(128, "entries", "CXL-DMSim Table III", "datasheet"),
    dev_fifo=Param(48, "entries", "CXL-DMSim Table III", "datasheet"),
    switch_lat=Param(100, "ns", "CXL-DMSim Table III (switch-attached)",
                     "datasheet", "only when a CXL switch is in the path"),
    # Silicon measurements the model is checked against (their Fig. 9/Table II)
    silicon_ddr_l=Param(130, "ns", "CXL-DMSim Fig.9, LMbench random read",
                        "datasheet"),
    silicon_asic=Param(284, "ns", "CXL-DMSim Table II, real CXL ASIC",
                       "datasheet"),
    silicon_fpga=Param(375, "ns", "CXL-DMSim Table II, real CXL FPGA",
                       "datasheet"),
    our_added_asic=Param(161.2, "ns", "measured: docs/CALIBRATION.md", "measured",
                         "vs silicon 154 ns, +4.7%"),
)


# ==========================================================================
# 5. CXL expander backend -- production Type-3 device briefs (papers/)
# ==========================================================================
# Two independent production device families, both public product briefs:
#
#   Astera Labs Leo (all production SKUs, May 2025 portfolio brief):
#       CXL 1.1/2.0 x16 (PCIe 5.0)  ->  2ch DDR5 up to 5600 MT/s, 2 TB
#   Marvell Structera X 2504 (product brief + 2024 launch deck):
#       CXL 2.0 x16 (PCIe 5.0)      ->  4ch DDR5-6400, 200 GB/s, >4 TB
#
# Backend-to-link ratio: Leo 2x44.8/63.0 = 1.42, Structera 204.8/63.0 = 3.25.
# Production expanders provision DRAM bandwidth ABOVE the link in every SKU,
# so the LINK binds -- confirming the assumption our harness makes.  Our
# expander model uses a DDR5 backend (Step 2) sized so the link binds, which
# both vendors' briefs now justify.

CXL_BACKEND = {
    "leo_cm5162": dict(
        link=Param(16, "lanes", "Astera Leo portfolio brief, May 2025 "
                   "(papers/AsteraLabs_Leo_Portfolio_Brief_2025.pdf)",
                   "datasheet", "CXL 1.1/2.0 on PCIe 5.0"),
        backend=Param(2, "DDR5 ch", "same brief: '2ch DDR5 up to 5600MT/s, "
                      "2TB' for every production SKU", "datasheet"),
        backend_peak=Param(89.6, "GB/s", "derived: 2 x 5600 MT/s x 8 B",
                           "derived"),
    ),
    "structera_x2504": dict(
        link=Param(16, "lanes", "Marvell Structera X 2504 product brief "
                   "(papers/Marvell_StructeraX2504_Product_Brief.pdf)",
                   "datasheet", "CXL 2.0 / PCIe 5.0 1x16-port"),
        backend=Param(4, "DDR5 ch", "same brief: 4ch DDR5 up to 6400 MT/s",
                      "datasheet"),
        backend_peak=Param(204.8, "GB/s", "derived: 4 x 6400 MT/s x 8 B; "
                           "deck quotes '200 GB/s memory bandwidth'",
                           "derived"),
        device_power=Param(30, "W", "Marvell Structera launch deck (c) 2024: "
                           "'Typical power consumption of <30W'", "datasheet",
                           "controller device only; DIMM power excluded"),
    ),
}


# ==========================================================================
# 6. Memory-side energy anchors -- published figures the model calibrates to
# ==========================================================================
# The DRAM energy MODEL is DRAMsim3's IDD x VDD state machine (Li et al.,
# IEEE CAL 2020, sim/gem5/ext/dramsim3/DRAMsim3), which follows the Micron
# power-calculation methodology (TN-40-07 for DDR4; same structure for DDR5).
# JESD238A defines the HBM3 IDD states (s9) but leaves the current VALUES to
# vendor datasheets (Table 90 is an empty template), so absolute HBM energy is
# anchored to the peer-reviewed access-energy figure below.

ENERGY = {
    "hbm2_access_energy": Param(
        3.97, "pJ/bit", "O'Connor et al., 'Fine-Grained DRAM', MICRO'17 "
        "(papers/OConnor_MICRO17_FineGrainedDRAM.pdf)", "datasheet",
        "Detailed physical model + floorplan: activation 909 pJ/1KB-row "
        "(1.21 pJ/bit at app-average locality) + on-die data movement "
        "2.24 pJ/bit + interposer I/O 0.3 pJ/bit = 3.92 incl ECC; abstract "
        "quotes 3.97. The HBM-class access-energy anchor."),
    "hbm3_calibrated": Param(
        3.97, "pJ/bit", "calibrated: HBM2-template IDD state ratios scaled "
        "by one factor (HBM3 k=3.270, HBM3e k=4.648) at JESD238A Table 70 "
        "VDD=1.1 V so the measured total equals the O'Connor anchor",
        "derived",
        "one-target calibration, stated as such -- JESD238A leaves IDD "
        "values to vendor datasheets (Table 90 empty by design). HBM3e held "
        "EQUAL per bit (conservative; vendor claims better). No "
        "generation-relative HBM energy claims are made, by construction. "
        "UNITS: DRAMsim3 energy stats are V*mA*cycles, not pJ; "
        "multiply by tCK (see measure_energy.py)."),
    "ddr5_expander_measured": Param(
        11.85, "pJ/bit", "measured: sim/measure_energy.py, DDR5-6400 "
        "subchannel, sequential; IDD currents are Micron 16Gb DDR5 die "
        "datasheet values carried by gem5's DDR5 class (uncalibrated)",
        "measured",
        "sits in the DDR/GDDR device-class range (O'Connor Fig.1a: GDDR5 "
        "14 pJ/bit). Known small understatement: DRAMPower-style activate "
        "formula goes slightly negative with these currents (IDD0 measured "
        "single-bank vs IDD3N all-bank), and IPP (VPP) is not carried by "
        "the gem5 class."),
    "hbm3e_vendor_claim": Param(
        2.5, "x perf/W vs HBM2E", "Micron HBM3E product brief, Rev. C 10/2023 "
        "(papers/Micron_HBM3E_Product_Brief.pdf)", "datasheet",
        "qualitative only -- no vendor publishes HBM3E pJ/bit; brief confirms "
        "16 channels / 2 pseudo-channels / BL8 geometry we model."),
    "serdes_pcie5_phy": Param(
        11.4, "pJ/bit", "IEEE: 'A 32Gb/s NRZ 37dB SerDes in 10nm CMOS to "
        "Support PCI Express Gen 5 Protocol' (doi 10.1109/CICC48029.2020)",
        "datasheet",
        "long-reach PHY incl PLL+clocking, per direction. Short-reach "
        "on-package links go as low as 0.54 pJ/bit (GRS, cited in O'Connor "
        "ref [33] context) -- brackets the link-energy sweep."),
    "cxl_device_energy_ceiling": Param(
        18.75, "pJ/bit", "derived: Structera X 2504 <30 W typical / "
        "200 GB/s x 8 (both figures from Marvell's own material)", "derived",
        "whole expander device (PHY+ctrl+cache+DMA) at full backend "
        "bandwidth, DIMMs excluded. Upper bound for e_link + e_ctrl; "
        "vendor 'typical' power, so conservative."),
}


# ==========================================================================
# 7. Things that still need a datasheet
# ==========================================================================
# Listed explicitly so they cannot leak into the report unnoticed.

NEEDS_SOURCE = {
    "cxl_link_ctrl_energy_split": Param(
        0, "pJ/bit", "no vendor publishes the SerDes-vs-controller split",
        "ESTIMATE",
        "Bracketed by the ENERGY entries: short-reach PHYs ~0.5 pJ/bit/dir, "
        "a published long-reach PCIe5 PHY 11.4 pJ/bit, whole-device ceiling "
        "18.75 pJ/bit. SWEEP e_link+e_ctrl jointly over 2-19 pJ/bit; "
        "conclusions must hold across the range."),
    "gpu_cxl_lane_count": Param(
        16, "lanes", "no shipping GPU exposes CXL host ports", "ESTIMATE",
        "x16 assumed. GPU-side CXL is emerging (a 2026 kernel patch adds a "
        "CXL DVSEC readiness check for Blackwell-Next in NVIDIA's NVGrace-GPU "
        "VFIO driver), but no datasheet fixes the width. SWEEP THIS."),
    "hbm3e_array_timings": Param(
        0, "ns", "no HBM3E preset in Ramulator2; JESD238A covers HBM3 only",
        "ESTIMATE",
        "HBM3 array timings held constant in ns with only the PHY scaled. "
        "Same array generation, so defensible, but not sourced."),
}


def audit() -> None:
    """Print every parameter with its provenance."""
    def show(label, p: Param):
        flag = "  !!" if p.confidence == "ESTIMATE" else ""
        print(f"  {label:<28}{p.value:>10.4g} {p.unit:<12}"
              f"[{p.confidence}]{flag}")
        if p.note:
            print(f"      {p.note}")
        print(f"      src: {p.source}")

    print("=== CXL / PCIe link rates (spec arithmetic) ===")
    for k, p in LINK.items():
        show(k, p)
    print("\n=== effective link bandwidth (measured, this harness) ===")
    for k, p in LINK_EFFICIENCY.items():
        show(k, p)
    print(f"\n=== GPU HBM (vendor datasheets) ===")
    for name, g in GPU.items():
        print(f"  {name} ({g['hbm_gen']}): {g['capacity'].value:g} GB, "
              f"{g['bandwidth'].value:g} GB/s, {g['stacks'].value:g} stacks "
              f"-> {gpu_per_stack_gbps(name):.0f} GB/s/stack [datasheet]")
    print(f"\n=== HBM device models (JESD238) ===")
    for gen, h in HBM.items():
        print(f"  {gen}: peak {h['peak'].value:.0f}, achieved "
              f"{h['achieved'].value:.0f} GB/s/stack, "
              f"{h['capacity'].value:g} GB/stack")
    print(f"\n=== CXL protocol path (CXL-DMSim, silicon-calibrated) ===")
    for k, p in CXL_PATH.items():
        show(k, p)
    print(f"\n=== CXL expander backends (production device briefs) ===")
    for name, d in CXL_BACKEND.items():
        for k, p in d.items():
            show(f"{name}.{k}", p)
    print(f"\n=== energy anchors (published) ===")
    for k, p in ENERGY.items():
        show(k, p)
    print(f"\n=== NEEDS A DATASHEET -- do not use unverified ===")
    for k, p in NEEDS_SOURCE.items():
        show(k, p)


if __name__ == "__main__":
    audit()
