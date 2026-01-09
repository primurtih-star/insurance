def to_rupiah(amount):
    """
    Format angka menjadi Rupiah.
    Contoh: 25000 → Rp 25.000
    """
    amount = float(amount)
    return f"Rp {amount:,.0f}".replace(",", ".")


def to_usd(amount):
    """
    Format USD untuk dashboard.
    Contoh: 25000 → $25,000.00
    """
    amount = float(amount)
    return f"${amount:,.2f}"
