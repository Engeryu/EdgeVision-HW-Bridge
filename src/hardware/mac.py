# ===========================================================
#  File    : mac.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-15
# ===========================================================

from amaranth import Elaboratable, Module, Signal, signed

from src.config import cfg


class MACUnit(Elaboratable):
    """
    Multiply-Accumulate (MAC) hardware unit.
    Designed to process quantized weights (int8) from the PyTorch model.
    """

    def __init__(self, bit_width: int = cfg.hw.bit_width):
        # --- 1. Component Pins (IO) Definition ---

        # Inputs
        self.pixel_in = Signal(signed(bit_width), name="pixel_in")
        self.weight_in = Signal(signed(bit_width), name="weight_in")

        self.clear = Signal(name="clear")

        # Output
        self.result_out = Signal(signed(bit_width * 3), name="result_out")

    def elaborate(self, platform) -> Module:
        """
        Physical description of the component's internal synchronous logic.

        Defines the cycle-by-cycle behavior of the MAC unit:
        - If the 'clear' signal is high, the accumulator resets to zero.
        - Otherwise, at each clock tick, the product of the incoming pixel
          and weight is added to the running total.

        Args:
            platform: The target FPGA/ASIC platform (unused in pure simulation).

        Returns:
            Module: The instantiated Amaranth hardware module.
        """
        m = Module()

        with m.If(self.clear):
            m.d.sync += self.result_out.eq(0)
        with m.Else():
            m.d.sync += self.result_out.eq(
                self.result_out + (self.pixel_in * self.weight_in)
            )

        return m


if __name__ == "__main__":
    from amaranth.back import verilog

    mac = MACUnit()
    with open("mac.v", "w") as f:
        f.write(
            verilog.convert(
                mac, ports=[mac.pixel_in, mac.weight_in, mac.clear, mac.result_out]
            )
        )

    print("The Verilog file 'mac.v' was successfully generated!")
