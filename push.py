import psana
import psana.pscalib.calib.MDBUtils as mu  # type: ignore
import psana.pscalib.calib.MDBWebUtils as wu  # type: ignore
import psana.detector.UtilsCalib as uc  # type: ignore
cc = wu.cc
import argparse

def main(args):
    ds = psana.DataSource(exp=args.experiment, run=args.run)
    runs = next(ds.runs())
    detname = args.detname
    out_file = args.geometry
    ctype = "geometry"
    dtype = "str"
    data = mu.data_from_file(out_file, ctype, dtype, verb="DEBUG")
    detector = runs.Detector(detname)
    longname: str = detector.raw._uniqueid
    shortname: str = uc.detector_name_short(longname)
    det_type: str = detector._dettype
    run_orig: int = args.run
    run_beg: int = args.run
    run_end: str = "end"
    run: int = run_beg
    kwa = {
        "iofname": out_file,
        "experiment": args.experiment,
        "ctype": ctype,
        "dtype": dtype,
        "detector": shortname,
        "shortname": shortname,
        "detname": detname,
        "longname": longname,
        "run": run,
        "run_beg": run_beg,
        "run_end": run_end,
        "run_orig": run_orig,
        "dettype": det_type,
    }
    if args.dbsuffix:
        kwa["dbsuffix"] = args.dbsuffix
    _ = wu.deploy_constants(
        data,
        args.experiment,
        longname,
        url=cc.URL_KRB,
        krbheaders=cc.KRBHEADERS,
        **kwa,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push geometry file to calibration database")
    parser.add_argument("-e", "--experiment", type=str, help="Experiment name, e.g. mfx100852324")
    parser.add_argument("-r", "--run", type=int, help="Run number, e.g. 298")
    parser.add_argument("-d", "--detname", type=str, help="Detector name, e.g. jungfrau")
    parser.add_argument("-g", "--geometry", type=str, help="Path to geometry file to be pushed")
    parser.add_argument("-b", "--dbsuffix", type=str, help="Database suffix, e.g. testgeom", default=None)
    args = parser.parse_args()
    main(args)
