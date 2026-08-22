"""Memory-side energy composition and EDP.

Every coefficient below is measured from the calibrated DRAMsim3 configs or
bracketed by a cited public figure -- provenance in sim/system_params.py and
sim/README.md ("Memory energy model").  Composition per step/run:

    E = bytes_hbm_read x e_hbm_rd  +  bytes_hbm_fill x e_hbm_wr
      + bytes_cxl_read x (e_ddr5_rd + e_link_ctrl)
      + [P_bg_hbm x n_hbm_ch + P_bg_ddr5 x n_ddr5_subch] x T
    EDP = E x T

Coefficient derivation (re-derivable with `python sim/measure_energy.py`):

  * HBM3/HBM3e dynamic pJ/bit = activate + column-read components of the
    CALIBRATED sequential run (total pinned to O'Connor MICRO'17 3.97 pJ/bit).
    Sequential is the right operating point: expert weights are 12-336 MiB
    contiguous streams.
  * HBM write (cache fill) scales the column term by the calibrated IDD ratio
    (IDD4W - IDD3N) / (IDD4R - IDD3N).
  * Background power = refresh + standby energy rate of the same run --
    charged per channel x wall time, NOT per bit, so idle channels cost
    energy and EDP penalises slow configurations twice, as it should.
  * DDR5 expander coefficients come from Micron-datasheet IDD (via gem5's
    DDR5 class) with NO calibration; the 11.85 pJ/bit total landing in the
    DDR/GDDR class range is itself a check.
  * e_link_ctrl (SerDes + expander controller, pJ/bit) is not public for any
    device; it is BRACKETED [2, 19]: short-reach PHYs ~0.5/dir, published
    PCIe5 long-reach PHY 11.4, whole-device ceiling 18.75 (Structera X 2504
    <30 W / 200 GB/s).  Every conclusion must be reported across the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

PJ = 1e-12

# component split of the calibrated sequential run, pJ/bit (see module doc)
# Background power is STATE-RESOLVED, not a single number:
#
#     P_bg(u) = P_refresh + u * P_active_standby + (1-u) * P_precharge_standby
#
# where u is the fraction of wall time the device is actively streaming.
# The single `bg_w_per_ch` figure it replaces was the SATURATED endpoint
# (u = 1), so charging it unconditionally over-billed every stall-dominated
# run -- the regime the tiering results live in. gem5 reported 22.0 W against
# the old model's 25.3 W on a real decode window; the split below reproduces
# that. Coefficients come from DRAMsim3's own per-state CYCLE COUNTERS
# (rank_active_cycles / all_bank_idle_cycles / num_cycles), so each power is
# energy-in-state / cycles-in-state -- measured, not fitted.
HBM = {
    "HBM3":  dict(act=0.318, col_rd=2.943, ref=0.165, stb=0.547, wr_ratio=1.329,
                  bg_w_per_ch=0.252, channels_per_stack=16,
                  p_act_stb=0.1980, p_pre_stb=0.1441, p_ref=0.0630),
    "HBM3e": dict(act=0.440, col_rd=2.788, ref=0.208, stb=0.533, wr_ratio=1.328,
                  bg_w_per_ch=0.382, channels_per_stack=16,
                  # measured from an HBM3e_24Gb_x64_1ch run's own per-state
                  # cycle counters, NOT scaled from HBM3. (An earlier draft
                  # scaled the HBM3 values by the bg_w ratio and mislabelled
                  # them "measured"; that was wrong by -6% / -6% / +17%.)
                  p_act_stb=0.2816, p_pre_stb=0.2046, p_ref=0.1119),
}

DDR5 = dict(act=0.0,          # DRAMPower-style activate term goes slightly
                              # negative with Micron DDR5 currents; clamped,
                              # small understatement (documented)
            col_rd=8.30, ref=0.232, stb=3.39, wr_ratio=0.868,
            bg_w_per_subch=0.64,
            p_act_stb=0.6248, p_pre_stb=0.4048, p_ref=0.0449)

E_LINK_CTRL_SWEEP = (2.0, 10.0, 19.0)   # pJ/bit; cited bracket, see module doc


@dataclass
class EnergyModel:
    hbm_gen: str = "HBM3"
    n_hbm_channels: int = 80            # 5 stacks x 16 (H100)
    n_ddr5_subchannels: int = 8         # Structera X 2504 backend
    e_link_ctrl: float = 10.0           # pJ/bit, swept

    def __post_init__(self):
        h = HBM[self.hbm_gen]
        self.e_hbm_rd = h["act"] + h["col_rd"]                    # pJ/bit
        self.e_hbm_wr = h["act"] + h["col_rd"] * h["wr_ratio"]
        self.e_cxl_rd = (DDR5["act"] + DDR5["col_rd"]) + self.e_link_ctrl
        # saturated background, kept for backward compatibility / u = 1
        self.p_bg = (h["bg_w_per_ch"] * self.n_hbm_channels
                     + DDR5["bg_w_per_subch"] * self.n_ddr5_subchannels)

    def p_background(self, u_hbm: float = 1.0, u_ddr5: float = 1.0) -> float:
        """Watts at the given per-tier activity fractions (0..1)."""
        h = HBM[self.hbm_gen]
        u_hbm = min(1.0, max(0.0, u_hbm))
        u_ddr5 = min(1.0, max(0.0, u_ddr5))
        p_h = h["p_ref"] + u_hbm * h["p_act_stb"] + (1 - u_hbm) * h["p_pre_stb"]
        p_d = (DDR5["p_ref"] + u_ddr5 * DDR5["p_act_stb"]
               + (1 - u_ddr5) * DDR5["p_pre_stb"])
        return p_h * self.n_hbm_channels + p_d * self.n_ddr5_subchannels

    def energy(self, bytes_hbm_read: float, bytes_hbm_fill: float,
               bytes_cxl_read: float, t_total_s: float,
               u_hbm: float | None = None,
               u_ddr5: float | None = None) -> dict:
        """Pass u_hbm / u_ddr5 (fraction of wall time each tier is streaming)
        to get state-resolved background power. Omitting them keeps the old
        saturated-endpoint behaviour, which OVER-bills stall-dominated runs."""
        e_hbm = bytes_hbm_read * 8 * self.e_hbm_rd * PJ
        e_fill = bytes_hbm_fill * 8 * self.e_hbm_wr * PJ
        e_cxl = bytes_cxl_read * 8 * self.e_cxl_rd * PJ
        p_bg = (self.p_bg if u_hbm is None and u_ddr5 is None
                else self.p_background(1.0 if u_hbm is None else u_hbm,
                                       1.0 if u_ddr5 is None else u_ddr5))
        e_bg = p_bg * t_total_s
        tot = e_hbm + e_fill + e_cxl + e_bg
        return {
            "e_hbm_read_j": e_hbm,
            "e_hbm_fill_j": e_fill,
            "e_cxl_read_j": e_cxl,
            "e_background_j": e_bg,
            "e_total_j": tot,
            "avg_power_w": tot / max(1e-12, t_total_s),
            "edp_js": tot * t_total_s,
        }


def hbm_only(hbm_gen: str = "HBM3", n_hbm_channels: int = 80) -> EnergyModel:
    """HBM-only configuration: no expander, no link -- its background power
    and link term drop out entirely."""
    m = EnergyModel(hbm_gen, n_hbm_channels, n_ddr5_subchannels=0,
                    e_link_ctrl=0.0)
    return m
