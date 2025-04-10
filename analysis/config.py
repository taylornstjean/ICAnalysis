# Contains path configurations
import os
from glob import glob


RUN_GROUP_ID = 21220

NFILES = {
        21220: 9954,
        20878: 9995,
        21002: 9949
    }

BASE_DIR = "/data/i3home/tstjean/icecube/"

I3TESTFILE = "/data/i3store/users/tstjean/i3tests/Alertv2_IC86.2016_NuMu.021220.009999.i3.zst"

I3FILEDIR_NUMU_TEST = "/data/i3store/users/tstjean/i3tests"

I3FILEDIR_NUMU = {
    21220: "/data/i3store/users/blaufuss/data/alert_catalog_v2/sim_21220_alerts",
    20878: "/data/i3store/users/blaufuss/data/alert_catalog_v2/sim_20878_alerts",
    21002: "/data/i3store/users/blaufuss/data/alert_catalog_v2/sim_21002_alerts"
}

SAMPLE_GCD_FILE = "/data/i3store/users/blaufuss/data/IC86/Run1179XX/GCD_Run117921.i3"

DATA_DIR = "/data/i3store/users/tstjean/data"

YML_CONFIG_SCHEMA_FILE = os.path.join(BASE_DIR, "data/config/yml_config_schema.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DB_TEST_DIR = os.path.join(DATA_DIR, "backend/test")
# 1DB_TEST_FILE = glob(os.path.join(DATA_DIR, "backend/21220/Alertv2_IC86.2016_NuMu.021220.00000*.db"))[0]

