# ===========================================================
#  File    : mac.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-14
# ===========================================================

from amaranth import Elaboratable, Module, Signal, signed

from src.config import cfg


class MACUnit(Elaboratable):
    """
    Unité matérielle de calcul Multiply-Accumulate (MAC).
    Conçue pour traiter les poids quantifités (int8) issus du modèle PyTorch.
    """

    def __init__(self, bit_width: int = cfg.hw.bit_width):
        # --- 1. Définition des broches (Pins / IO) du composant ---

        # Inputs
        self.pixel_in = Signal(signed(bit_width), name="pixel_in")
        self.weight_in = Signal(signed(bit_width), name="weight_in")

        self.clear = Signal(name="clear")

        # Output
        self.result_out = Signal(signed(bit_width * 3), name="result_out")

    def elaborate(self, platform) -> Module:
        """Description physique des connexions internes du composant."""
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

    print("Le fichier Verilog 'mac.v' a été généré avec succès !")
