from time import perf_counter
from analysis.core import I3File


def test_metadata_extraction():
    file = I3File("/data/i3store/users/blaufuss/data/alert_catalog_v2/sim_21002_alerts/Alertv2_IC86.2016_NuMu.021002.009999.i3.zst")
    file.metadata(21002)

if __name__ == "__main__":
    time_start = perf_counter()
    test_metadata_extraction()
    time_end = perf_counter()
    print(f"\t---\tRun complete in {time_end - time_start} seconds.\t---")
