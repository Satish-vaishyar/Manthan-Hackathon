from bot.signals.overlays import min_hold_allows_exit


def test_negative_age_legacy_lot_allowed():
    # Legacy holdings seeded with wall-clock date vs past sim time -> age<0
    assert min_hold_allows_exit(-498, 3, "exit_target") is True


def test_young_lot_still_blocked():
    assert min_hold_allows_exit(1, 3, "exit_target") is False


def test_old_lot_allowed():
    assert min_hold_allows_exit(5, 3, "exit_target") is True


def test_stop_loss_always_allowed():
    assert min_hold_allows_exit(0, 3, "atr_stop") is True
