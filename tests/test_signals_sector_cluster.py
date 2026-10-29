"""tests/test_signals_sector_cluster.py — coverage for bot.signals.sector_cluster (P2-08)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from bot.config import SectorClusterSettings
from bot.signals.sector_cluster import SectorCluster, cluster_sectors


@dataclass
class FakeStock:
    ticker: str
    description: str


@pytest.fixture
def stocks():
    return [
        FakeStock("AAA", "Software company building enterprise SaaS products and developer tools"),
        FakeStock("BBB", "Pharmaceutical company focused on oncology drugs and vaccines"),
        FakeStock("CCC", "Bank offering retail and corporate lending, mortgages, wealth management"),
        FakeStock("DDD", "Steel manufacturer for automotive and construction industries"),
        FakeStock("EEE", "IT services firm providing cloud, DevOps, and consulting"),
        FakeStock("FFF", "Pharma firm developing generic drugs and pharmaceutical ingredients"),
        FakeStock("GGG", "Software tools for developer productivity and code review"),
        FakeStock("HHH", "Bank specializing in investment banking, advisory, and trading"),
    ]


def test_sector_cluster_fits_and_groups(stocks):
    sc = SectorCluster(SectorClusterSettings(n_clusters=3, random_seed=42))
    sc.fit(stocks)
    assert sc.fitted
    mapping = sc.to_dict()
    # Mapping is deterministic for the seed
    assert all(v.startswith("S") for v in mapping.values())


def test_sector_cluster_predict_one_known_ticker(stocks):
    sc = SectorCluster(SectorClusterSettings(n_clusters=3, random_seed=42))
    sc.fit(stocks)
    s_aaa = sc.predict_one("AAA")
    assert s_aaa.startswith("S")
    assert isinstance(s_aaa, str)


def test_sector_cluster_unknown_ticker_falls_back_to_s0(stocks):
    sc = SectorCluster(SectorClusterSettings(n_clusters=3, random_seed=42))
    sc.fit(stocks)
    assert sc.predict_one("UNKNOWN") == "S0"


def test_sector_cluster_too_few_tickers_uses_fallback():
    only_one = [FakeStock("AAA", "A software company")]
    sc = SectorCluster(SectorClusterSettings(min_n_clusters=5))
    sc.fit(only_one)
    assert sc.fitted
    assert sc._fallback is True
    assert sc.predict_one("AAA") == "S0"


def test_sector_cluster_is_deterministic(stocks):
    sc1 = SectorCluster(SectorClusterSettings(n_clusters=4, random_seed=7))
    sc2 = SectorCluster(SectorClusterSettings(n_clusters=4, random_seed=7))
    sc1.fit(stocks); sc2.fit(stocks)
    # Cluster ID assignments may not be identical because KMeans in sklearn
    # can permute labels, but the partitioning must be stable.
    mapping1 = sc1.to_dict()
    mapping2 = sc2.to_dict()
    assert set(mapping1.keys()) == set(mapping2.keys())
    # Same partition: tickers that share a label in mapping1 share it in 2.
    same_partition = True
    tickers = list(mapping1.keys())
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i + 1:]:
            same1 = mapping1[t1] == mapping1[t2]
            same2 = mapping2[t1] == mapping2[t2]
            if same1 != same2:
                same_partition = False
                break
    assert same_partition


def test_cluster_sectors_helper(stocks):
    out = cluster_sectors(stocks, SectorClusterSettings(n_clusters=3, random_seed=42))
    assert isinstance(out, dict)
    assert set(out.keys()) == {s.ticker for s in stocks}


def test_sector_cluster_predict_many(stocks):
    sc = SectorCluster(SectorClusterSettings(n_clusters=3, random_seed=42))
    sc.fit(stocks)
    out = sc.predict_many(["AAA", "BBB", "CCC"])
    assert set(out.keys()) == {"AAA", "BBB", "CCC"}
    assert all(v.startswith("S") for v in out.values())